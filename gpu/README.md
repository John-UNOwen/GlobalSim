# GlobalSim GPU spike

One OpenCL kernel of the restricted flat no-skills tick chain (eight ticks per
race), proven bit-identical to the Rust reference on an Intel Arc B580, plus
timing against a 20-thread Rust baseline. See `docs/GPU_HANDOFF.md` in the
engine repo for the plan this spike implements.

The crate is standalone (empty `[workspace]` table) and vendors the engine's
`src/retail_physics.rs`, `src/json.rs`, `src/replay_frames.rs`, `src/random_gen.rs`
and `src/retail_random.rs` in place through `#[path]` includes in `src/lib.rs`,
plus `src/retail_geometry.rs` behind the `geometry` feature (see "Geometry
port"). The engine files are never edited to make the GPU pass. License and
provenance: `NOTICE`.

## Prerequisites

- Rust 1.100 (on this machine: `%USERPROFILE%\.cargo\bin\cargo.exe`)
- An OpenCL 3.0 GPU that passes the exactness gate described below
- The Intel oneAPI runtime for linking: OpenCL.lib import library

## Build

Every cargo command that links OpenCL needs the oneAPI lib dir on `LIB`:

```powershell
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
$env:LIB = 'C:\Program Files (x86)\Intel\oneAPI\2026.1\lib;' + $env:LIB
Set-Location "<repo-checkout>\gpu"
cargo build --features gpu --release
```

Without `--features gpu` the crate is pure Rust (no OpenCL link, no `LIB`
needed), which covers the reference tests. `--features geometry` is pure Rust
too (see "Geometry port").

## Test

```powershell
cargo test --features gpu --release
```

Suites:

| Suite | Proves |
| --- | --- |
| `lib` (24 tests) | Vendored engine tests pass inside this crate |
| `reference_trace` | The in-crate runner reproduces `restricted_trace.json` from the Jax oracle byte-for-byte (CRLF normalized) |
| `layout` | The three `repr(C)` kernel structs are 104/36/244 bytes with the exact field offsets, and `kernel_params()` equals the engine's `Parameters` public methods for every scalar the kernel reads |
| `comparator` | No device needed: one injected wrong value in a fake GPU outcome reports exactly the injected tick, runner, field and both raw bit patterns, equal frames with a flag are not parity |
| `gpu_gate` | See below |
| `rng_gate` (gpu only) | See "RNG port" below |
| `geometry_gate` (gpu + geometry only) | See "Geometry port" below |
| `race_diff` (gpu only) | The whole comparison pipeline on `examples/v77-race.jsonl` line 1 with geometry 10504 and seed 7: the CPU engine runs the whole race (1463 ticks), the stub's ticks diverge, and the outcome carries the fallback flag |
| `solo_race_gate` (gpu + initialization + geometry + race) | See "Race loop port" below |

### What the gate proves

`gpu_gate` runs 6 cases x 8 ticks through the OpenCL `restricted_tick` kernel
and compares every row's `time/pos/speed/hp/delay/finish/phase/dash/finished`
against the Rust reference, as `to_bits()` for the floats. Any single
differencing bit fails the test with the case, row, field, and both bit
patterns. It passed on the B580 on 2026-11-18 (`single_fp_config 0xbf`,
`double_fp_config 0x3f`, `cl_khr_fp64`).

`Spike::open` refuses devices that lack `cl_khr_fp64`, f64 subnormals, f32
subnormals, or correctly-rounded f32 divide/sqrt, and names the refusals. The
program is built with `-cl-fp32-correctly-rounded-divide-sqrt` and the kernel
source carries `#pragma FP_CONTRACT OFF` (`src/kernels/restricted_tick.cl`).

## Bench

```powershell
cargo run --features gpu --release --bin bench
```

Prints per N in {1, 64, 256, 4096, 65536}: upload, average kernel run, read
back, total, GPU races/s, the Rust baseline (native threads,
`available_parallelism`, one chunk per thread) and the ratio, plus the f64 vs
f32 fma microkernel ratio. Fixed iteration counts, printed in the output
header: 8 ticks per race, 8 warmup plus 256 timed kernel launches per N, 4096
fma per chain x 4 independent chains per work item with 20 timed launches.

The bench also re-checks bit-exactness for all 32768 rows at N=4096, so the
timed runs are known to execute the chain the gate proved.

## Whole-race comparison tool (handoff steps 4 and 5)

The crate depends on the full engine as a path dependency
(`globalsim = { package = "race_engine_rs", path = ".." }`,
`external-geometry`, default features off). The tool adds no engine file
copies, so `NOTICE` needs no change.

`gpu-race-diff` runs the engine and the GPU race runner on one request and
seed, and prints the race summary, the first divergent tick with both raw
bit patterns, and the fallback flag:

```powershell
cargo run --features gpu --release --bin gpu-race-diff -- --request ../examples/v77-race.jsonl --geometry ../data/course_geometry/10504.json --seed 7
```

Output on the B580:

```text
cpu race: 1463 ticks, 9 runners, 9 finishers (course 10504, seed 7)
gpu race: tick 0 runner 0 field pos: cpu bits 0x0000000000000000 gpu bits 0x3fc9930c00000000
fallback: UnsupportedDomain("field/rng/geometry not ported")
```

The request file is JSONL and the first line is taken. Omit `--seed` to use
the request's own seed. `--engine cpu-only` skips the GPU side.

The CPU side (`src/dep_cpu.rs`) runs the engine through its public API only:
`register_course_geometry`, `PreparedRace::new` with `retainAllFrames: true`
forced, `run(seed)`, and `Simulation::frames()`.

### What the tool proves today

The GPU side is `race::RestrictedStub`. It maps the request's player lane to
one restricted kernel case, runs the gated `restricted_tick` kernel for its
8 ticks, and flags the race with
`UnsupportedDomain("field/rng/geometry not ported")` from the first tick:
initialization, the field, RNG and geometry are not ported. The emitted
ticks still take part in the comparison, and their first divergence is the
progress measure (handoff step 4). The sample line above says it: every
engine runner sits out its RNG start delay while the delay-free stub is
already 0.20 m down the track at tick 0.

### Fallback flag (handoff step 5)

`race::FallbackReason` names why a race cannot be trusted on the GPU:
`UnsupportedDomain` for a domain the kernel cannot guarantee,
`DeviceUnavailable` for no gated device, a build without the gpu feature, or
an OpenCL failure. The scheduler treats a set flag as its rerun-on-CPU
queue. `comparator::compare` returns `Ok` only when every compared field
equals bit for bit and no flag is set.

Step 5's acceptance, zero unflagged mismatches over a large seed sweep, is
trivially true today because every whole race carries the flag. That is not
yet asserted as a race for anything but the restricted subset.

### How a whole-race kernel plugs in

`race::RaceRunner::run_race(request, seed) -> RaceOutcome` is the seam. A
step-6 kernel runner implements the trait, fills `RaceOutcome.frames` with
every runner's rows per tick, and sets `fallback` only where the kernel
cannot guarantee exactness. The comparator and the tool's summary and flag
handling need no change. `RaceOutcome` carries no tick timestamps; a kernel
that emits its own timeline will need the outcome extended.

## RNG port (handoff step 6.1)

`src/kernels/rng.cl` holds one kernel, `rng`, and one work item per chain, with
a per-item `mode` selecting Rule30, Retail, or substream. `src/rng.rs` builds the
input and output buffers, runs the kernel through the `gpu::Spike` device gate
(`Spike::build_program` compiles `rng.cl` with the same exactness options), and
exposes pure-Rust reference wrappers over the `#[path]`-included engine RNGs. The
reference wrappers call `crate::random_gen::Rule30CARng` and
`crate::retail_random::RetailRandom` directly. Only `uniform_guarded` (the
capped rejection loop) and `substream_key` (the FNV-1a plus murmur3 hash) mirror
a loop or a hash, because the engine exposes no capped-uniform entry point and
returns no intermediate hash. `uniform_guarded` still draws from the engine's
`int32()`, so no stream is reimplemented.

### What `rng_gate` pins

All comparisons run on the gated B580, and every float is compared through
`to_bits()`.

- Engine parity: 1024 Rule30 chains (splitmix-spread `u32x2` seeds) match the
  pure-Rust chain, including final `hi`/`lo`, the `int32()` and `random()` draws,
  and `uniform()` over fractional, exactly-1, 0, and tiny-negative uppers. 8040
  retail series over 1005 seeds (the signed boundaries plus a splitmix spread)
  match, across `next_int`, `next_double`, `integer` narrow and wide, and `real`
  and `real_distance` with swapped bounds. 256 substream keys (seed, runner 0 to
  7, four tags including `RUN` and a skill-style tag) match the derived seed, the
  hash, and the derived retail chain.
- Fixture tie-in: for the six seeds in `tests/fixtures/retail_rng_dotnet.json`
  (independent .NET `System.Random` values), the kernel's `next_int`,
  `next_double`, and `integer(i32::MIN, i32::MAX)` sequences equal the fixture's
  `integers`, `doubles`, and `wide` arrays. This pins the kernel, not just the
  Rust reference, to that external reference.

### Two sharp edges handled

- RetailRandom state math. Rust's `state` is `[i32; 56]` with `wrapping_sub` and
  `wrapping_add(i32::MAX)`, and a `value == i32::MAX` special case. The kernel
  stores the state as `uint`, runs `+` and `-` as the u32 wraps Rust performs, and
  casts to `int` only for the `< 0` sign tests. `wrapping_add(i32::MAX)` becomes
  `+ 0x7FFFFFFF` and `value == i32::MAX` becomes `== 0x7FFFFFFF`. No signed add in
  the kernel can overflow, so there is no undefined behavior on the device.
- `uniform()` runs an uncapped rejection loop, and on a degenerate upper (below
  `i32::MIN`) it never returns. The kernel caps the loop at `UNIFORM_CAP` 4096 and
  writes a sentinel with the item's fallback flag set (the handoff step 5
  mechanism) instead of hanging. `rng_gate` forces one such item (`upper -3e9`) and
  checks the sentinel equals a guarded Rust run within the cap. The tiny-negative
  uppers take the signed return without hitting the cap, so the guard and the
  engine's `uniform` are asserted equal for every non-cap case.

`next_double` multiplies by a compile-time constant. The engine computes
`1.0/i32::MAX as f64` (raw bits `0x3e00000000200000`); the kernel embeds it as the
hex literal `0x1.0000000200000p-31`, and `rng_gate` ties that literal to the Rust
value through the bits assertion.

## Geometry port (handoff step 6, geometry slice)

`src/kernels/geometry.cl` holds one kernel, `geometry`, one work item per
query, and a per-item `mode` selecting `corner_state_at`, `lane_max_at`,
`position` or `advance` for one course, ported line for line from
`src/retail_geometry.rs` with the widths of each source expression preserved:
segment coordinates interpolate in f64 with one f32 store, the quaternion
lerp normalizes by `(dot(q,q) as f64).sqrt() as f32` with f32 divides, the
lane-offset chain keeps the f32 `num2..num11` products with one f32 store per
component, `advance` squares, sums and compares in f64 with single f32 stores,
and `corner_state_at` scans the sorted corner events in reverse, short
circuiting at the first event with `distance >= start`. The Rust reference
side is the engine module itself, `src/retail_geometry.rs`, included in this
crate behind the `geometry` feature. JSON parsing (`from_value`) stays
host-side, nothing to port.

### Engine modules with CARGO_MANIFEST_DIR tests must be gated out of `--lib`

`src/retail_geometry.rs` carries `#[cfg(test)]` tests that read
`env!("CARGO_MANIFEST_DIR")/data/...`. Included as is, those tests compile
into this crate's unit-test target and look for data under `gpu/`, so they
fail there even though they pass in the engine. The rule for any engine module
with tests like that: declare the include behind `#[cfg(all(feature = "...",
not(test)))]` in `src/lib.rs`, and put its gate in an integration test (the
`tests/` dir compiles against the lib without `cfg(test)`). `cargo test --lib`
then never sees the module, and the gate runs through
`cargo test --features gpu,geometry --test geometry_gate`.

The accessors that read the engine's private runtime state (`flatten_course`,
`drop_overrun` in `src/retail_geometry.rs`) must sit in the same module as the
engine code, because struct field privacy is module-scoped. They live in a
`gpu/src/retail_geometry.rs` that `include!`s the engine file. A direct
`include!` of the engine file fails (E0753): its leading `//!` doc comment is
never module-leading inside a macro expansion, so `build.rs` emits a copy with
those lines turned into plain `//` comments (text preserved) and the one
nested `include!("retail_geometry_assets.rs")` rewritten to resolve from
`OUT_DIR`. The engine file on disk is never modified, and `build.rs` panics if
that include line ever changes shape.

### Kernel data (one course, flat read-only buffers)

`flatten_course` copies the state `CourseGeometry::from_value` built, field
for field, into `FlatCourse`. For `C` courses and `N` items the kernel reads
and writes (`run` in `src/geometry.rs` uploads them; f32 values in the
integer arrays go as raw bits):

| Buffer | Type, length | Contents |
| --- | --- | --- |
| `cs` | u32[16 per course] | course table: `distance`, `distance_to_world`, `initial_lane_max`, overrun present + `distance`, `rotation_fix` (4 f32 bits), lane-max and corner block bases and counts, main and overrun segment block bases |
| `sg_x`, `sg_xd`, `sg_z`, `sg_zd` | f64[2002 per course] | `AnimationSegment` `x`, `x_delta`, `z`, `z_delta`: main 1001-sample block, then overrun block (zeros when the course has none) |
| `sg_q`, `sg_qd` | f32[8004 per course] | segment `rotation` and `rotation_delta`, f32 x4 per sample |
| `lm_b`, `lm_l` | f32[total lane changes] | `lane_max_changes` boundary and limit |
| `ce_b`, `ce_e`, `ce_i`, `ce_f` | f32, f32, u32, u32[total corner events] | `corner_events` start, end, id, flags (bit 0 `final_corner`, bit 1 `straight_clears`) |
| `it` | u32[8 per item] | course index, mode, ready, distance, lane, movement, previous world x and z |
| `out` | u32[8 per item] | fallback flag, corner id, final-corner flag, result slots A/B/C (f32 bits) |

Engine errors (bad distance/lane/movement/previous-world, or past the course
end on a course with no overrun) become the step 5 fallback instead of a
failure: `OS_FLAG` set and the result slots at the sentinel `f32::NAN` bits
`0x7fc00000`.

### What `geometry_gate` pins

All comparisons run on the gated B580, and every float is compared through
`to_bits()`. Verified 2026-11-18 (`single_fp_config 0xbf`,
`double_fp_config 0x3f`).

- Fixture tie-in: every row of the engine's independent captured-game fixture
  `tests/fixtures/retail_geometry_port.json` (engine root, read through the
  repo path) replays through the kernel: 40,440 rows across 119 courses, and
  the kernel's four outputs (next distance, world x, world z,
  `lane_max_at`) equal the fixture's `expected_bits`, 161,760 values. Same
  role as the retail RNG fixture pin.
- Engine parity: 247,103 items over 10504, 10604, 10613 and 11102: a
  course-length/4096 distance grid (including the `i == 1000` endpoint), six
  lanes per distance (0, ±1/18, 6/18, `lane_max_at(d)` itself, 1e6), corner
  queries with `ready` on and off, all three query modes at each grid
  distance, ULP-1/ULP/ULP+1 probes around every corner start, corner end and
  lane-width boundary, `advance` with `previous_world` from
  `position(d - 0.1, lane)` and movements `0`, f32-denormal-tiny, half and
  one times the world step and 10 times it and 1000 (clip branch and skip
  branch both hit), and the 10604 overrun: distances 2001 to 2200 and
  distances past the end by multiples of the overrun distance (the fmod
  wrap).
- Error/fallback: NaN, negative and past-end distances, infinite lanes, NaN,
  negative and infinite movements and previous-world coordinates each set
  the flag and write the sentinel, matching the engine's own `is_err()` item
  by item, with valid items in the same batch staying clean. No valid sweep
  sample shows the flag or the sentinel bit pattern anywhere.

### Engine behavior that needed a handling decision

- Every captured course ships an overrun animation, so "past the course end
  without overrun" is unreachable from `load` or `bundled`. The gate reaches
  it the way the engine's own test does: clone the geometry and clear the
  overrun through the `drop_overrun` accessor (private field, engine module
  scope).
- Rust's f32 `%` (the overrun wrap) is `fmod`, and the remainder of the wrap
  is exact for both operators, so the OpenCL `fmod` carries the same value.
- `(sample as usize).min(1000)` in `position` needs no saturation trick on
  the device: the error checks and the wrap leave the sample finite in
  `[0, 1000]`, so `min(convert_int_sat_rtz(sample), 1000)` matches.

## Initialization port (handoff step 6, initialization slice)

`src/kernels/initialization.cl` holds one kernel, `initialization`, one work
item per chain, with the per-item `mode` selecting the adjusted-stats chain,
the per-solver Rule30 seeding chain, the retail initialization stream or the
initial HP chain, ported line for line from `src/retail_initialization.rs`,
the seeding part of `src/race_solver_builder.rs` and the rng part of
`RaceSolver::new` in `src/race_solver_retail.rs`. The Rust reference side runs
the vendored engine modules in place (feature `initialization`), and every
comparison goes through `to_bits()`. The kernel is concatenated after
`rng_core.cl` (`INITIALIZATION_KERNEL_SOURCE` in `src/gpu.rs`), so the ported
`Rule30CARng` and `RetailRandom` helpers, including `uniform_guarded` (a
mirror of `Rule30CARng::uniform`: mask `0xFFFFFFFF >> ((v | 1) as u32)
.leading_zeros()`, signed i64 compare), are shared with the gated RNG port and
`rng_gate` stays green.

What each mode pins:

- `IM_STATS`: `adjust_overcap`, `team_rank_bonus` (f32 thresholds 6/11/16/21/
  26), `ranked_base_stat` as `initial_base_wisdom((adjust_overcap(raw) as f32
  + bonus) as f64, mood)` with motivator `1 + 0.02 * mood`, the f32
  accumulator of `course_speed_modifier` (0.05/0.1/0.15/0.2 per nonzero css
  entry at the 300/600/900/2000 distance boundaries), the ground-and-turf
  speed and power tables indexed `[surface][ground]`, and the strategy and
  distance aptitude tables.
- `IM_R30`: `seed_lo = (seed + i * 7919) as u32`, the murmur fmix32 race seed
  hi, the guarded `uniform` pair that seeds the solver and the discarded
  pacer, then the solver draws in `RaceSolver::new` order: gate roll
  `uniform(12252240)`, random lot `uniform(100)`, the start-delay random, the
  hill-pair seeds, the hill gate roll (only when the first slope starts at
  0.0, `if slope_per < 0.0 && roll < wisdom * 0.0004`), `kakari uniform(7)`,
  the check random, three section randoms and the 24 section randoms. The
  hp-seed draw precedes all of them.
- `IM_INIT`: the retail init stream of `initialize()` and
  `initialize_sections_and_rushing()`: the force-in-move pass, the wisdom
  chance (`100 - 9000 / base_wisdom`, floored at 20), the start-delay draws,
  the 24 section randoms with the f32-rooted base and log ranges, the rushing
  eligibility draw `chance > (0, 100] integer` and its lot-section draw, the
  paired mode where each runner's rushing lottery draws from its own
  `RetailRandom::substream` (FNV-1a of seed, runner and tag, then the
  shift-multiply avalanche), the f32 `kakari_start` feed the retail runtime
  writes over the f64 one, six stage draw counters and eight continuation
  `next_int` plus eight `next_double` draws that pin the shared stream state
  without needing the tick loop.
- `IM_HP`: `GameHpState::with_retail` and `init()`: `initial_hp` through the
  bounded-stat f64 chain with the single f32 store, `course_speed`, the
  `hp_guts_factor` (f64 `sqrt`, then f32) and the HP ground modifier table.

### log10, the one transcendentally rounded function in the chain

The section and rushing chains call `log10` on f64. `engine_log10` in the
kernel is a line-for-line OpenCL port of `libm` 0.2.16 `src/math/log10.rs`
(FreeBSD msun `s_log10.c`), the same algorithm the Rust std compiles for musl
and wasm targets, with the hex float constants read from the crate source on
disk. Operands get explicit `(double)` casts everywhere the Rust multiplies in
f64, because OpenCL evaluates `f32 * f32` in float. On this host the Rust std
`f64::log10` matched the port bit for bit at every intermediate stage (a
probe dumped `xr`, `k`, `s`, `r`, `hi`, `lo`, `val_hi`, `y`, `val_lo` and the
result against a Rust mirror of the same libm code) and across every value the
576-case retail sweep drew, so the engine side and the kernel side agree. A
host whose libc ships a different `log10` would move the reference side with
it. Two transcription slips found this way (a dropped leading mantissa nibble
of IVLN10HI and a wrong `IVLN10LO` exponent) each moved only the last f64 bits
and nothing else, which is why only a stage-by-stage probe, not the end value,
could localize them.

### Kernel data (flat buffers, one stride per case)

`run` in `src/initialization.rs` uploads (`f32` values inside the integer
arrays travel as raw bits, f64 slots as raw bits):

| Buffer | Type, stride | Contents |
| --- | --- | --- |
| `in_u` | u32[96] | mode at 0, scalars at 1..6 (seed, n, paired flag, paired seed, lot min, lot max), css at 7..10, per-runner triple (style, adjusted-wisdom bits, bonus bits) from 16, stride 3 |
| `in_l` | u64[48] | per mode: stats raws 0..4, r30 first-slope start, value and adjusted wisdom 0..2, init distance at 0, per-runner base wisdom 8..31, parameters 32..41, hp stamina..`hp_drain[4]` 0..14 |
| `out_u` | u32[992] | RetailRandom bank 0..59, paired RUSH bank 60..119, fallback flag at 120, six stage counters 121..126, r30 scalars 128..159, runner rows at `128 + 30 * r` (force, chance, delay, rush chance, lot section, rush present, 24 sections), continuation `next_int` at 848..855, hp guts and wisdom factors at 170..171, race-rng bank snapshot at 928..987 (see "Race loop port") |
| `out_l` | u64[64] | continuation `next_double` bits at 32..39, init `kakari_start` feed 0..23, hp ground modifier, base speed and max hp at 0..2 |

The first version used a 27-slot row stride, which silently overlapped each
runner's 30-slot row (6 scalars + 24 sections) into the next one, and
continuation slots at 160 landed inside runner 1's row. Kernel and host agreed
on the addresses, so only the written value (last write wins) betrayed it. The
raw-slot dump in a debug run localized it in one step.

### What `initialization_gate` pins

All comparisons run on the gated B580, and every value is compared through
`to_bits()`. Verified 2026-11-18 (`single_fp_config 0xbf`,
`double_fp_config 0x3f`).

- Adjusted stats: 704 cases, 8,448 values. Raw stats at every comparison
  boundary of `adjust_overcap` and `team_rank_bonus` (119/1200/1200.0000001,
  ranks around 6/11/16/21/26), moods -2/0/2, and random fills, compared on
  base and adjusted stats plus the raw-stamina and raw-power rewrites.
- Rule30 seeding: 38,326 values over 512 random cases and 256 hill-boundary
  cases (wisdom solved so `roll` sits exactly at `wisdom * 0.0004`, then
  ULP-1/ULP/ULP+1 probes on both sides). The hill gate fired in 530 cases.
  The cap flag was never set (guarded `uniform`, rng gate shared).
- Retail init stream: 576 cases, 224,867 values, 376 paired and 200 drawing
  a rush. Crafted parameter sets carry negative force coefficients, a negative
  and a zero delay max, `lot_min == lot_max == 0`, temptation -13, wisdom and
  bonus edges (10.0, 112.5, 20.000002), and paired seeds `None`, `Some(0)`,
  `Some(-1)`. Stage counters, all per-row draws, the 24 sections, the
  `kakari_start` feed and the continuation draws are each compared per runner.
- Initial HP: 2,880 values, 576 cases, crafted physics with a zero initial
  coefficient, drain edges 0 and 1, style coefficients 0.5/1/1.02, stamina and
  guts edges around the 1/2000 bounds including negative.
- v77 reply pin: the `v77-race.jsonl` line-1 request (course 10504, 9 runners,
  ground Good) driven through the real engine in-process for seeds 1..8, with
  the kernel's chains feeding the comparison. Pinned reply fields:
  `startDelayTime` (the kernel's f32 widened to the f64 the reply writes),
  `gutsOrder` and `wizOrder` (orders built from the kernel's adjusted stats),
  and the frame-0 hp column (the kernel's max hp). 4,131 values.
- Source pins: the needle test asserts the seven engine files the host mirrors
  and references still contain the exact lines the mirrors were read from.

### Seams and not ported in this slice

- Skill triggers and their rng streams stay out. With no skills in the entry
  the wisdom lottery is computed but never drawn, and the pacer rng
  (`race_solver_builder.rs:867`) is drawn into and discarded; its seed pair is
  pinned only for stream parity. The six stage counters would catch any stream
  shift if skills joined later.
- Request-level handling stays host-side: the `gate_perm` entry permutation
  (`src/api.rs`) and the experimental `startup_pass` (`src/lib.rs:828`, a tick,
  not initialization). The v77 request uses neither.
- The gate pins only the reply fields listed above. Everything a tick writes
  belongs to the loop slice (handoff step 6, remaining).

## Race loop port (handoff step 6, no-skill races, field of one)

`src/kernels/race.cl` holds the whole-race kernel, `race`, one work item per
race, with the tick loop inside the kernel (up to `RACE_MAX_TICKS` 6144). The
translation unit is `rng_core.cl`, then `initialization.cl`, then
`geometry.cl`, then `src/kernels/retail_core.cl`, then `race.cl`, concatenated
as `RACE_KERNEL_SOURCE` in `src/gpu.rs`, so the gated RNG, initialization and
geometry code is shared untouched. `retail_core.cl` carries the ported per-
runner state columns (f64 physics state, u32 flags, f32 scratch) and the
ported logic: phase-boundary and section targets, the slope-event physics with
the downhill branch, the rushing state machine per strategy (Nige/Sashi/Senkou
rules and the end-check lottery), the position-keep modes with their wisdom
and exit-gap draws, the spurt plan (30 m candidate grid, speed grid in 0.1
steps under `maximum_spurt_speed`, time-diff sort, per-candidate dice), the
lane target chain (final-corner outward roll, side room, block parameter
pick), lane movement and overlap resolution, the front-blocker sample and
speed cap, the HP rate table with the rush and downhill multipliers, and the
finish step with the `finishTimeRaw` interpolation. The Rust reference side is
the engine through its public API (`src/dep_cpu.rs`).

Host side is `src/race_exec.rs` behind the `race` feature. `run` takes one
spec per race (course id, distance, geometry, seed, horse, ground) and returns
frames plus finish data; `GpuRace` implements `race::RaceRunner`. Domain gates
run before any launch: skills, multi-runner fields (duel ranking and spot
struggle), paired mode, the startup pass and grid mode each return the race
flagged with its reason instead of launching wrong numbers.

The race's dice stream is the manager RNG cloned at `lib.rs:773`, which is
after `initialize()` and `initialize_sections_and_rushing()` have consumed
their draws. `initialization.cl` snapshots the bank at that point (`out_u`
slots 928..987) and `race.cl` restores it as the loop's stream, so the section
and rushing lottery draws never touch race dice and the later continuation
draws (which exist only to pin the stream in `initialization_gate`) stay out.
The one `powf` in the loop (`maximum_spurt_speed`'s guts term) is computed
host-side in an `IM_STATS` pre-pass and passed as raw f32 bits in parameter
slot 84.

### What `solo_race_gate` pins

Verified 2026-11-19 on the gated B580 (`single_fp_config 0xbf`,
`double_fp_config 0x3f`), one kernel launch for all 24 races.

- 24 no-skill v77-style races: courses 10504, 10604 and 11004 (distances
  2000, 2000, 2600) x seeds 1..8, field of one (strategy Senkou, skills
  empty). Every tick, every runner, every frame column (pos, speed, hp, lane,
  rush mode, blocker) compared through `to_bits()` against the CPU engine's
  frames, plus identical tick counts and identical `finishTimeRaw` bits, with
  zero fallback flags on every race.
- `v77_multi_runner_race_stays_flagged` (`#[ignore]`): the 9-runner sample
  request returns flagged ("multi-runner duel and spot are not ported"),
  documenting where the multi-runner port resumes.

### Rounding traps this port hit

- The `maximum_spurt_speed` guts term indexes the adjusted stats by GUTS
  (layout index 3). Reading index 0 (speed) instead shifts every spurt plan
  by a constant +0.0582 m/s.
- `hp_per_second_with_guts_factor` computes `(gap*gap)/hp_drain[1]` in f32 in
  Rust and only then widens. Doing that divide in f64 moves the per-tick cost
  by one f32 ulp once the rush rate multiplier is in play, visible within a
  few hundred ticks.
- The engine clones the shared RNG after the rushing lottery, not after
  `initialize()` alone, so the snapshot point sits after the per-runner
  section and rush draws.

## Not covered (step 6 blockers)

Engine behavior the race loop port does not reach, each needing its own port
and gate:

- Skills and their rng streams (trigger windows, additive modifiers into
  `acceleration` and `target_speed`, activation hooks)
- Multi-runner field interactions: duel ranking, spot struggle, field-level
  oikomi and oonige effects
- Paired mode, the experimental startup pass and grid mode (flagged, gated)
- Ground and terrain lookups beyond the slope and corner geometry events
- Replay frame encoding outside the gate's frame comparison
