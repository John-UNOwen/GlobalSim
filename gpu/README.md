# GlobalSim GPU spike

One OpenCL kernel of the restricted flat no-skills tick chain (eight ticks per
race), proven bit-identical to the Rust reference on an Intel Arc B580, plus
timing against a 20-thread Rust baseline. See `docs/GPU_HANDOFF.md` in the
engine repo for the plan this spike implements.

The crate is standalone (empty `[workspace]` table) and vendors the engine's
`src/retail_physics.rs`, `src/json.rs`, `src/replay_frames.rs`,
`src/binary_replay.rs`, `src/random_gen.rs` and `src/retail_random.rs` in place
through `#[path]` includes in `src/lib.rs`, plus `src/retail_geometry.rs`
behind the `geometry` feature (see "Geometry port"). The engine files are never
edited to make the GPU pass. License and provenance: `NOTICE`.

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
| `lib` (28 tests) | Vendored engine tests pass inside this crate |
| `reference_trace` | The in-crate runner reproduces `restricted_trace.json` from the Jax oracle byte-for-byte (CRLF normalized) |
| `layout` | The three `repr(C)` kernel structs are 104/36/244 bytes with the exact field offsets, and `kernel_params()` equals the engine's `Parameters` public methods for every scalar the kernel reads |
| `comparator` | No device needed: one injected wrong value in a fake GPU outcome reports exactly the injected tick, runner, field and both raw bit patterns, equal frames with a flag are not parity |
| `gpu_gate` | See below |
| `rng_gate` (gpu only) | See "RNG port" below |
| `geometry_gate` (gpu + geometry only) | See "Geometry port" below |
| `race_diff` (gpu only) | The whole comparison pipeline on `examples/v77-race.jsonl` line 1 with geometry 10504 and seed 7: the CPU engine runs the whole race (1463 ticks), the stub's ticks diverge, and the outcome carries the fallback flag |
| `solo_race_gate` (gpu + initialization + geometry + race) | See "Race loop port" below |
| `paired_race_gate` (gpu + initialization + geometry + race) | See "Paired mode" below |
| `zero_skill_gate` (gpu + initialization + geometry + race) | See "Zero-skill mixed-style fields" below |
| `skills_gate` (gpu + initialization + geometry + race) | See "Skills port" below |
| `app_preset_gate` (gpu + initialization + geometry + race) | Real app mob-preset jobs: zero-fallback compile gate and frame identity both green (see "App mob-preset payloads") |
| `facade_gate` (gpu + initialization + geometry + race) | See "Batch API" below |

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
handling need no change. `RaceOutcome` carries no tick timestamps, so a kernel
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
`1.0/i32::MAX as f64` (raw bits `0x3e00000000200000`). The kernel embeds it as the
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
and writes (`run` in `src/geometry.rs` uploads them, and f32 values in the
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
  (`race_solver_builder.rs:867`) is drawn into and discarded. Its seed pair is
  pinned only for stream parity. The six stage counters would catch any stream
  shift if skills joined later.
- Request-level handling stays host-side: the `gate_perm` entry permutation
  (`src/api.rs`) and the experimental `startup_pass` (`src/lib.rs:828`, a tick,
  not initialization). The v77 request uses neither.
- The gate pins only the reply fields listed above. Everything a tick writes
  belongs to the loop slice (handoff step 6, remaining).

## Race loop port (handoff step 6, no-skill races, multi-runner field)

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
speed cap, the HP rate table with the rush and downhill multipliers, the
rank column and oikomi/oonige field styles, the duel proximity timers with
their warmup and entry rules, the spot struggle state machine, the clean
conserve-power ledger and its legacy gauge, and the asitame release that rides
the last-spurt transition, plus the finish step with the `finishTimeRaw`
interpolation. The Rust reference side is the engine through its public API
(`src/dep_cpu.rs`).

Host side is `src/race_exec.rs` behind the `race` feature. `run` takes one
spec per race (course id, distance, geometry, seed, horse list, ground) and
returns frames plus finish data. `GpuRace` implements `race::RaceRunner`. The
tick loop runs the runners sequentially in roster order inside one work item,
so the shared race dice are consumed in the same order the engine's field
moves them. Paired mode gives every runner its own stream instead (see
"Paired mode" below). Domain gates run before any launch: the startup pass,
non-retail runtime and a roster outside the kernel's 1..24 range each return
the race flagged with its reason instead of launching wrong numbers, and
skill-bearing fields are gated by the skill compiler (see "Skills port"). A
field of up to `RACE_MAXR` (24) runners launches unflagged when it has no
skills or only skills the compiler accepts.

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
- `v77_multi_runner_races_are_bit_exact`: the 9-runner v77 line-1 roster (all
  Senkou, skills empty, ground Good) on courses 10504 (seeds 1..8), 10604
  (seeds 1..4) and 11004 (seeds 1..4). Every tick, every runner, every frame
  column compared through `to_bits()` against the CPU engine with all frames
  retained, equal tick counts, equal `finishTimeRaw` bits, and zero fallback
  on every race. This is the field gate: it exercises roster order, ranks,
  blocking, lane-target selection and its PassingSearch draws, the duel warmup
  and entry, the spot struggle, the conserve ledger and the asitame release,
  none of which a field of one can reach.

### Paired mode

A request with `pairedRandomStreams: true` replaces each runner's shared
runtime stream with a fresh per-runner stream: the engine's
`retail_random::substream(seed, runner, "RUN")` at `lib.rs:794` (FNV-1a of
seed, runner and the tag, then the shift-multiply avalanche, then
`RetailRandom::new`). The `lib.rs:773` clone the runtime takes is replaced per
runner and never consumed, and the Rule30 solver rng stays exactly as it is:
paired mode keys only the retail stream. Inside the kernel, paired mode sends
each runner's rushing lottery to its own RUSH substream in `im_init` (the same
switch `initialize_sections_and_rushing` applies, pinned by
`initialization_gate`'s 376 paired cases), and builds one fresh RUN bank per
runner (`r128_substream_hash_run` in `rng_core.cl`) in scratch (`SCR_BANKS`,
60 words per runner). The sequential tick loop reads each runner's own bank
for the whole race. Pace, rushing, lane, downhill and spurt draws all come
from it, matching the engine's per-runner `RetailRuntimeConfig` (its HP policy
shares the same stream). Unpaired mode keeps the shared-bank path untouched.

### What `paired_race_gate` pins

Verified 2026-11-18 on the gated B580 (`single_fp_config 0xbf`,
`double_fp_config 0x3f`), one kernel launch for all 16 races.

- `v77_paired_races_are_bit_exact`: the same 9-runner v77 line-1 roster and
  courses, requested with `pairedRandomStreams: true`, seeds 1..8 on 10504
  and 1..4 on 10604 and 11004. Every tick, every runner, every frame column
  compared through `to_bits()` against the CPU engine running the same paired
  request, equal tick counts, equal `finishTimeRaw` bits, zero fallback on
  every race. 1,370,466 compared values.
- `paired_requests_change_the_race`: the same request with and without the
  paired key differ in the CPU engine (course 10504 seed 1, runner 4 pos at
  tick 72, unpaired bits `0x40540dd360000000`, paired bits
  `0x40540ddda0000000`), so the pinned equality proves the kernel switched
  streams, not that both sides ignored the flag.

### Zero-skill mixed-style fields

`zero_skill_gate` runs app-shaped zero-skill rosters (mob presets, uma stat
blocks, mixed styles, roster sizes 2 to 18) through the kernel and the CPU
engine: equal tick counts, every frame column `to_bits()`-equal, and
`finishTimeRaw` bits equal, for seeds 1..3 on 10504 and 10604, zero fallback
on every race.

Oonige (strategy 5, Runaway) is ported and gated on this suite: a dedicated
4-runner probe, the mob9 layout with its gate-0 Nige swapped to an Oonige,
and the 18-runner field with its gate-10 Nige swapped, each on both courses.
The tick-side start-dash state machine, the runaway HP drain, the Ex phase
rows, the runaway pace columns, rush mode 4 and the field style rules came in
with the race loop itself. The missing piece was on the initialization side:
the engine hands the retail initialization the base style (`lib.rs:1533`), so
an Oonige enters the force-in-move pass as style 1, consumes one shared-stream
`real(0, forceInMoveTargetSpeedAddRandomRange)` draw, and carries the
RunningOut coefficient. The kernel packed the raw strategy into `RR_STYLE`,
saw style 5 there, skipped the draw, and every runner's start delay, section
modifiers and rushing lotteries then came from a shifted stream: whole-field
divergence at tick 1. `RR_STYLE` now packs the base style (`race_exec.rs`),
and `oonige_fields_are_not_refused` asserts the host and facade gates let
Oonige through. `gpu/OONIGE_CALLGRAPH.md` maps every place an Oonige touches
the engine and the kernel.

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
- The lane collect overwrote the force-in coefficient slot with the force-in
  flag instead of setting the separate flag column, so the block decision lost
  the coefficient. Both the coefficient and the flag need their own slot.
- The lane blocker scan was anchored on the owner's live position rather than
  the candidate's cached row, so a mid-pack blocker scan disagreed with the
  engine's `front_index`. Anchor on the cached row.
- The duel path is gated by `has_corner_events`, the host's
  `RI_NSTR && RI_NCOR`. Those two counts were packed nowhere, so duels never
  fired on the GPU and the CPU's final-straight duel target bump had no GPU
  match. Set them from the course's straights and corners.

## Skills port (handoff step 7)

Skills ride the same race kernel, and the kernel does not change with the
skill set. The host compiles every skill of a race into one flat numeric table
(`src/skills_compile.rs`, `uint[4480]` per race) and the kernel interprets it:
init pass in `src/kernels/skills.cl` (trigger samplers, wisdom lottery, region
intersection) inside `im_init` before the existing rushing and section draws,
runtime in `src/kernels/retail_core.cl` (timer step and cooldown start, then
the priority 1 and 0 fire passes, then the receiver folds). Encoding, the
condition-support matrix, and the unsupported-construct rule are in
`SKILLS_DESIGN.md`.

The per-tick order mirrors `step_all_retail`: `skills_advance` (timers, expiry,
cooldown start), `skills_scan` x2 (the fire gate: trigger and window check over
the record's final window list, cooldown readiness, then the program over the
six dynamic leaves), `sk_columns` (fold the receiver lists into the live
columns), and only then the blocking sample that feeds duel and field. The
modifiers land in the engine's `AdditiveReceiver` folds at the exact physics
application points (target chain, accel chain, current speed), so the tick
chain from "Race loop port" is unchanged.

An unsupported construct never reaches the kernel. The compiler refuses the
table and the host flags the whole race with the construct named in the reason.

### What `skills_gate` pins

Verified 2026-11-18 on the gated B580 (`single_fp_config 0xbf`,
`double_fp_config 0x3f`), one kernel launch per batch, every frame column
compared through `to_bits()` against the CPU engine with all frames retained,
equal tick counts, zero fallback.

- `six_skills_on_horse_zero_replay_bit_exact`: six real skills (200361,
  200341, 200333, 10071, 100401, 110591) on horse 0 of the 9-horse v77
  roster, courses 10504 and 10510 x seeds 1..4, one batch of 8 races. The set
  carries sampled trigger windows (`all_corner_random`, four rounds each),
  wisdom rolls, the retain-for-random rule, and a 300000 ms cooldown scaled
  by course distance. The same batch without the skills differs in all 8
  replays, so the comparison is not vacuous.
- `one_skill_per_race_over_the_matrix_bit_exact`: 6 skills x 2 courses x 2
  seeds, one skill per race, as one 24-race batch. Batching is a tested
  dimension: the stride bug pinned below stayed invisible in solo batches and
  only crossed race boundaries inside one launch. The engine's
  `skillDetailEvents` reply records a horse-0 activation in 12 of the 24
  races, the other three skills replay as never activated on these courses,
  and both outcomes are pinned bit-exact.
- `unsupported_construct_flags_the_whole_race`: one skill's `durationUsage`
  patched to 4 (the delayed running-instance path) flags the race before any
  launch, and the reason names `duration_usage 4`.
- `unsupported_random_term_refuses_by_name`: one skill's condition rewritten to
  `random_lot==1` (a sampled term with no ported sampler) refuses the race, and
  the reason names `random_lot`, so an unported random term never runs on an
  empty or wrong window.

### App mob-preset payloads

The app sends each mob's skill list as bare skill-id strings and carries the
skill definitions only as a separate merged `skills.json`, so the compiler must
accept the bare-string form, take the definitions from the request `skillData`,
and refuse any skill name it cannot lower rather than guess. `app_preset_gate`
replays the two real job payloads in `gpu/testdata/` (courses 10504 and 10604,
seeds 1..3) against the CPU engine.

- `mob_preset_jobs_compile_without_fallback` (green): all 12 preset races reach
  the kernel with zero fallback, and at least one race activates a skill, so the
  identity gate below is not vacuous.
- `mob_preset_jobs_replay_bit_exact` (green): every frame column and every
  `finishTimeRaw` of the 12 preset races matches the CPU engine through
  `to_bits()`. Two engine behaviours had to be ported for that, the solver
  constructor's activation pass for immediate records, which fixes an immediate
  effect giving up its modifier one tick early, and the one-tick speed offset a
  natural-deceleration current speed (effect type 22) leaves when it expires,
  which fixes a runner snapping to its lower target instead of decaying to it.
  Both are written up in `SKILLS_DESIGN.md`.

### Cost

`cargo run --features gpu,initialization,geometry,race --release --bin skills_bench -- --runners 4096`
prepares and uploads once, then times kernel-only launches of the same buffers
(warm 2 plus 5 timed, no frames read back). At the 4096-runner budget (456
races of the 9-horse v77 roster, courses 10504/10510, seeds 1..4) on the
B580: no skills 1.90 us per race tick, 6 skills on 1 runner 2.18 us (+14%),
6 skills on 2 runners 2.39 us (+25%). Six skills on all nine runners would
need 54 records, and even five runners (30 skills) already overflow the
table's 512-byte tag block, so the stress case stops at two runners.

### One sharp edge found

The init pass chained a record's region lists through `sk_chain`, which read
the list argument at stride 4 words while every caller fills dense f64 pairs
at stride 2. With one list entry the bug hides (entry 0 is at offset 0). With
four sampled windows it read the wrong pairs and then past the end of the
private array, where the B580 hands back stack memory an earlier work item
left behind. The symptoms moved with batch composition: window lists from a
neighboring race, exact-duplicate windows in solo batches, and a bogus refire
around tick 1433 that looked like a cooldown bug. The fix is the stride 2 read,
and the 24-race matrix test above pins the class of it.

## Batch API (handoff step 8)

`facade::GpuRaces` is the lane an app holds. `new` runs the exactness gate,
loads the engine data and builds the race kernel once, `run_batch(requests,
seeds)` answers a launch with one `BatchOutcome` per request, in input order,
and `run_repeated(request, seeds)` answers one request over many seeds after
preparing it once (see "Batch cost").

The contract is the app's `run_compact` path. `meta` is the engine reply
JSON without frames, restricted to the fields that path reads:
`playerHorseIndex`, then per runner in roster order `finishTimeRaw`,
`finishOrder`, `hpEmptyOnRace`, `rushingStartCount`, `rushingPenaltyCount`,
`lastSpurtStartDistance` and `startDelayTime`, then the `skillEvents`
`(horseIndex, skillId)` pairs. `packet` is the GSIMRP01 replay packet:
`src/replay_packet.rs` hands the frames to the vendored
`binary_replay::encode`, the same call the engine's `into_binary` makes, so
the framing matches the engine byte for byte and the meta bytes embedded at
offset 32 equal the returned `meta` string exactly (frame rows start at
`32 + meta_len`). Frame `k` carries time `loop_end(k)`, the engine's f32
time sum.

A race a gate rejects (startup, grid, non-retail runtime, a roster outside
1..24, a skill construct the compiler refuses, an event log past `EV_MAX`
512, or a request that fails to parse) comes back as `fallback` with meta
and packet None, and the batch's other races run unaffected.
`gate_reasons(request, device)` answers the same refusals as a pure
function, so a caller can pre-filter against known device facts before
opening one.

Constraints. One `GpuRaces` per thread: it owns the OpenCL queue and
program and is not Sync. Every launchable race in one `run_batch` shares one
runner count, the kernel's uniform grid constraint, and a mixed batch is an
Err for the caller to split. Full replies and CPU replays stay on the CPU
lanes: the kernel exports no `skillDetailEvents`, `competitionEvents`,
`positioningPhases` or per-tick frame times inside the meta, and the compact
contract asks for none of them.

### `run_repeated` and the cost of one call

`run_repeated(request, seeds)` runs one request over many seeds in one
launch after preparing it once. The seed enters the prepared kernel inputs
only through the `ri[RI_SEED]` word (`prepare` reads nothing else from it,
the stats cases, the skill table, the parameter rows and the geometry are
seed-independent, and the kernel reseeds every RNG stream from that word),
so the entries past the first are a template clone with `with_seed` rewriting
that word. Outputs are byte-identical to `run_batch` on the request repeated
per seed (pinned by the two gate tests below), fallbacks included. A
`run_batch` prepare cache keyed by request hash was the alternative and was
rejected: it would keep every request alive indefinitely and make
`run_batch`'s cost depend on call history.

Measured on the B580, 2026-11-18, v77 line 1 through `multi_request`
(course 10504, 9 runners, one course per batch; `--bin batch_profile`
prints the tables):

- Program builds are per `RaceContext::open`, not per call: race program
  about 6 ms, initialization program about 2 ms, the device gate with its
  `restricted_tick` program about 3 ms warm (first open in a process about
  700 ms, driver cache cold). The 1.5 MB race binary costs a few ms at its
  first dispatch, never per launch. The stats pre-pass used to run
  `initialization::run`, which opened a fresh gated device and rebuilt the
  initialization program every call (about 7-9 ms warm, 700 ms cold); it
  now launches through the context's prebuilt program and costs about
  0.3-4 ms.
- The geometry registry loads and flattens each course once per context
  (about 4-9 ms cold), then clones the flat tables per prepared race, and
  one launch dedups the courses into shared tables (about 1 ms at 96
  races of one course).
- The fixed cost of a warm call is dominated by the race kernel itself: at
  m = 1 enqueue takes 0 ms and the device run about 429 ms. It scales with
  the race's tick count (course 11401, 1000 m: 712 ticks, 206 ms), not with
  the race count (marginal about 3.1 ms per race to n = 96), and not with
  buffer churn (ten back-to-back launches on the same buffers cost the same
  429 ms each). Batching is therefore the lever, and every facade entry
  already launches once per batch.
- Per-seed marginal at n = 96: `run_batch` about 9.8 ms per seed (prepare
  about 1.9, one 350 KB `playerHorseIndex` parse per race about 1.7, upload
  about 1.1, launch share about 3.1, readback about 0.8, packet about 1,
  decode about 0.2). `run_repeated` removes the prepare and the parse: about
  6.3 ms per seed, wall 1372 ms to 1034 ms at n = 96.

### What `facade_gate` pins

Verified 2026-11-18 on the gated B580 (`single_fp_config 0xbf`,
`double_fp_config 0x3f`).

- `compact_meta_and_packet_match_the_engine_lane`: one batch of four paired
  v77 races (course 10504, seeds 1..4) and the six-skill request. Every
  compact field equals the engine's own `write_json` reply, numbers through
  `to_bits()`, and the `skillEvents` `(horseIndex, skillId)` multiset equals
  the engine's, string ids included. The packet carries the engine magic,
  equal frame and horse counts, and frame bytes equal to the engine
  `into_binary` packet's (timestamps included), and the meta string equals
  the packet's embedded meta bytes.
- `a_startup_request_falls_back_and_its_siblings_stay_whole`: a startup
  request between two runnable siblings answers `fallback` naming the
  domain, carries no meta or packet, and its siblings keep theirs.
- `gate_reasons_names_every_refusal`: every refusal string against
  synthetic device facts (fp64, single and double fp-config bits, and the
  no-device case), then grid, startup, runtime and roster refusals for the
  request. No device is opened.
- `run_repeated_matches_run_batch_over_64_seeds`: the v77 line-1 request
  (course 10504, 9 runners) with seeds 1..64 through `run_batch` on the
  request repeated and through `run_repeated`: every meta string, packet
  byte and fallback identical.
- `run_repeated_matches_a_mixed_two_course_batch`: courses 10504 and 10604
  (9 runners each, so launchable in one batch) plus a startup and an
  unparseable request, `run_batch` on the mixed batch against
  `run_repeated` per request: all four groups byte-identical, the flagged
  and unparseable groups carrying the fallback arms.

### skillEvents: the engine writer sites, and what the kernel logs

From `src/lib.rs`: the activate hook records the kakari start time and emits
nothing, the deactivate hook emits the entry with `skillId` the string
`"kakari"`. A no-skill race therefore carries one entry per completed rush
episode and none for a rush still active when the loop ends. A real skill's
activate hook emits an immediate entry with a numeric `skillId` when its
duration is already spent, else parks it pending, and the deactivate hook (or
the race-end finalizer for effects still pending) emits the entry. The kernel
logs one `(runner, id)` per skill fire (`race_event` in
`src/kernels/retail_core.cl`) and one per rush end (transition `-1` in
`race.cl`, `EV_KAKARI`), which reproduces the engine multiset exactly: a
fire lands one entry, and an episode outliving the race emits an engine
entry only through the finalizer, one the fire already covers.

## Not covered (step 6 blockers)

Engine behavior the race loop port does not reach, each needing its own port
and gate:

- Skills outside the ported subset: the stateful time machines, activation
  counters, roster reads beyond rank and rank rate, effects outside 27/31/21,
  and the other constructs named in the `SKILLS_DESIGN.md` matrix. Each flags
  its race with the construct named
- The experimental startup pass (flagged, gated)
- Grid mode (`experimentalGrid`): not ported, host-gated. The engine enables
  it whenever the key is present (21 to 120 horses), so `prepare` flags any
  request carrying the key
- Ground and terrain lookups beyond the slope and corner geometry events
- Full (non-compact) replies: the version keys, `finishTime`, `gutsOrder`,
  `wizOrder`, `runningStyle`, `positioningPhases`, the `actualLastSpurt*`
  pair and the other telemetry channels the compact contract does not read.
  Those races stay on the CPU lanes (see "Batch API").
