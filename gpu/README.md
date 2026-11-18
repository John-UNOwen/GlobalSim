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

## Not covered (step 6 blockers)

Engine calls the restricted chain never reaches, each needing its own port and
gate:

- RNG: ported and gated for step 6.1 (see "RNG port" below), but the whole-race
  loop still consumes it through paths not yet in the kernel
- Retail simulation start: case construction (entry, `ReplayConfig` to
  `Simulation` fields, initial `Runner` states)
- Geometry queries: the four runtime queries are ported and gated (see
  "Geometry port"); ground and terrain lookups are not
- Lane and field queries: lane occupancy, `blocking_speed_cap`
- Pacing engine: section/pace sources and target-speed changes mid-race
- Spurt plan: `maximum_spurt_speed`, the only `powf` in the physics module
- Skill tables and additive modifiers (`&[f32]` slices into `acceleration` and
  `target_speed`)
- `hp_state_rate` and the hp style tables
- Replay frame encoding (`ReplayFrames` rows and the frame codec)
- The full `Simulation` loop: paired streams, race pacing, skills, finish order
