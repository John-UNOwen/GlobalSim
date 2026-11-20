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
| `layout` | The three `repr(C)` kernel structs are 104/36/244 bytes with the exact field offsets, `kernel_params()` equals the engine's `Parameters` public methods for every scalar the kernel reads, and the scratch stride mirrors are pinned (`SK_STRIDE` 54,912, `SCR_U_WORDS` 479,796, `SRL_WORDS` 233,720, the `RI`/`RIL`/`ST` strides and the init strides) |
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
| `startup_gate` (gpu + initialization + geometry + race) | Startup pass: the app mob-preset jobs flagged `experimentalRetailStartup: true`, both courses, seeds 1..3: zero fallback, frame identity, and per-race start values against the in-process engine's `retail_initialization::initialize` (see "Startup pass") |
| `wave1_folds` | Compile-time gates for the request-static condition folds and the multi-alternative rules, both fold directions plus the still-refused names (see "Wave-1 fallback-histogram constructs") |
| `wave1_gate` (gpu + initialization + geometry + race) | The six Phase A chart variant jobs, course 10504 seeds 1..3: zero fallback and frame identity, plus a v77-roster race firing the any-skill, shadowed-alternative, wisdom-carrier and out-band constructs bit-exact (see "Wave-1 fallback-histogram constructs") |
| `wave2_folds` | Compile-time gates for the wave-2 dynamic leaves and the roster style-count folds: each live leaf stays live, the sampled and folded shapes, and the still-refused names |
| `wave2_gate` (gpu + initialization + geometry + race) | One real merged skill per ported construct plus a v77-roster race firing each through synthetic defs, seeds 1..3: zero fallback and frame identity, including the constructor-pass defer and the absolute init cooldown (see "Wave-2 condition fields") |
| `wave2b_folds` | Compile-time gates for the effect side: cross-target records stay live, self-only kinds refuse cross targets, kinds 4 and 5 refuse ungated self fires and record construction-time bumps, the 8-effect cap, the live `value_usage` readers and the live random-activation kinds 36 and 37 at self target |
| `wave2b_gate` (gpu + initialization + geometry + race) | Ten real merged skills, one per ported effect family, plus a twelve-skill synthetic battery firing every cross-target and bump path, v77 roster seeds 1..3: zero fallback and frame identity (see "Wave-2b effect side") |
| `wave3_folds` | Compile-time gates for the wave-3 leaves: the in-band cell forms, the activation-history folds including the provably-false self-gate, the roster reads in both fold directions, the opponent-count team gate and the still-refused names |
| `wave3_gate` (gpu + initialization + geometry + race) | Fourteen real merged rows, one per ported construct with the seats the condition needs, plus a twenty-eight-skill synthetic battery firing every new leaf, v77 roster seeds 1..3: zero fallback and frame identity (see "Wave-3 condition tail") |
| `wave4_folds` | Compile-time gates for the effect tail: cross debuffs compile on any roster, a cross-target start-delay fires inert, the still-refused effect families (all-stat 32, variantless 33/34) refuse by name with the reason; GutsUp cross compiles live since wave 8, every wire `value_usage` compiles since wave 9, and an `additionalActivate` on a kind whose delivery writes per-receiver state still refuses by name; lane goals, kakari, immunity, kinds 36/37 at self target and the shared-key alternative pair assert live records (see "Wave-6a lane, rush and cooldown tail") |
| `wave4_gate` (gpu + initialization + geometry + race) | The real cross-debuff 910581 on a teamed roster plus a four-skill battery forcing the same-team drop, the different-team delivery and the inert cross start-delay both ways, v77 roster seeds 1..3: zero fallback and frame identity (see "Wave-4 effect tail") |
| `wave6a_folds` | Compile-time gates for the wave-6a constructs: lane goal and immunity records compile live for self and cross targets, stray target combinations refuse, and the shared-key alternative pair lowers both live alternatives (see "Wave-6a lane, rush and cooldown tail") |
| `wave6a_gate` (gpu + initialization + geometry + race) | The real shared-cooldown pair 910151, the real SetTargetLane 201261 and the real ExtendKakari 200801 on fitted seats, plus a ten-skill synthetic battery firing kinds 11, 12, 13 (self and group), 35 (self and delivered), 38 and the shared-key pair both ways, v77 roster seeds 1..3: zero fallback and frame identity with the expected skill ids in the CPU activation events (see "Wave-6a lane, rush and cooldown tail") |
| `wave6b_folds` | Compile-time gates: the runtime `value_usage` readers (8, 9, 11, 14) and the self-target additionalActivate shapes compile live, and `value_usage` 15 refuses by name (see "Wave-6b activation-time values and turns") |
| `wave6b_gate` (gpu + initialization + geometry + race) | The real merged ids for value_usage 8/11/14 (202031, 202032, 910341, 100981), additionalActivate (910211, 910351) and kind 37 (910071) on fitted seats, plus a 667xxx battery for kind 36, value_usage 9, the aa 1 cap and the never-fires white candidate, v77 roster seeds 1..3: zero fallback and frame identity with the expected skill ids in the CPU activation events (see "Wave-6b activation-time values and turns") |
| `wave7_folds` | Compile-time gates for the mid-race stat fires: cross SpeedUp, GutsUp and WisdomUp compile live on targets 4, 9 and 10 (kind 4 sets the plan's `live_guts` flag), ungated self GutsUp and WisdomUp refuse by name, every phase-gated self shape lowers, and `SK_FLAG_BUMPED` marks exactly the static 4/5/29 bumps (see "Wave-7 mid-race stat fires") |
| `wave7_gate` (gpu + initialization + geometry + race) | The real kind-1 loop fire 201641 beside a constructor-pass SpeedUp probe, plus a 668xxx battery for self WisdomUp refire, the static-bump-to-loop-fire accumulator chain, cross SpeedUp and WisdomUp on all three fans, the 10 s refire cycle and the immune and same-team debuff drops, v77 roster seeds 1..3: zero fallback and frame identity with the expected skill ids in the CPU activation events (see "Wave-7 mid-race stat fires") |
| `wave8_powf_gate` (gpu) | Bit-exactness gate for the transcribed ucrtbase `pow` and `powf` (`powf_core.cl`): the engine's guts domains plus fuzz bands, `to_bits` equality against Rust `f64::powf` and `f32::powf` at 100%, default 38.7M pairs, env-widened to 419.4M with zero differences |
| `wave8_folds` | Compile-time gates for live guts: the shipped GutsUp ids compile as baked constructor bumps (two keep their `random_lot` condition refusals, and 210141 folds dead on its `motivation` gate since wave 9c), and the mid-race and cross synthetic shapes set the plan's `live_guts` flag while static-only shapes never do |
| `wave8_gate` (gpu + initialization + geometry + race) | The 669xxx GutsUp battery: self phase-gated refire and drop, static bump chained into a loop fire, all three cross fans, the immune and same-team debuff drops, the constructor-only static control, and the real static ids 200211, 202181 and 1200051, v77 roster seeds 1..3: zero fallback and frame identity with the expected skill ids in the CPU activation events (see "Wave-8 live guts") |
 | `wave9_folds` | Compile-time gates for the tail: a `duration_usage` 4 record lowers with its running instance and the aux kinds 11, 12, 35 and 38 refuse under it, the `additionalActivate` fan compiles live on kinds 21, 22, 27, 28 and 31 at targets 1, 4, 9 and 10 and refuses outside that, `value_usage` at any wire value compiles as the engine's pass-through, and the real `duration_usage` 4 ids 100531 and 900531 plan live (see "Wave-9 running instance, fan and pass-through values") |
 | `wave9_gate` (gpu + initialization + geometry + race) | The real 100531 and 900531 beside a 6691xxx `duration_usage` 4 battery (no delayed effect, an extended `additionalActivate` sibling, a kind-2 turn on a short cooldown, a cross-target fan), a 6692xxx battery for the `additionalActivate` fan outside self on kinds 21, 22, 27, 28 and 31, and 6693xxx for the `value_usage` pass-through, v77 roster seeds 1..3: zero fallback and frame identity with the expected skill ids in the CPU activation events, plus one ignored scene that reproduces the fan-churn drift (see "Wave-9 running instance, fan and pass-through values") |
 | `wave9c_folds` | Compile-time gates for the condition-name tail: `grade`, `time`, `motivation`, `base_power` and `post_number` fold both directions with the refused shapes, the corner sampler resolves its row live, refuses an untagged non-integer course index and folds a missing-row arm, the viewer leaf lowers live with its header row and refuses an out-of-range `playerHorseIndex`, the roll-gated constructor bump marks `INIT_BUMP` and `BUMPED` with no host bump only for the kinds the kernel can bake, and the advantage-skill machine refuses (see "Wave-9c condition-name tail") |
 | `wave9c_gate` (gpu + initialization + geometry + race) | The real merged ids 100471, 900471, 200251, 202231, 202331, 210141 and 200351 on fitted seats beside a 6694xx battery for the directions the roster cannot reach (the false-side post_number fold, the activating time, grade, motivation, base_power and viewer-row fires, the corner sampler with a missing row and with four arms, the StaminaUp kind static, mixed and roll-gated), v77 roster seeds 1..3: zero fallback and frame identity with the battery's mid-race rows activating (see "Wave-9c condition-name tail") |
| `grid_gate` (gpu + initialization + geometry + race) | Grid starts at 21, 24, 40 and 120 horses, two courses, seeds 1..2: zero fallback and frame-for-frame `to_bits` identity against the CPU engine, the 21-horse grid-vs-retail control pair differing at frame 0, and invalid grid shapes falling back with the engine's own message (see "Grid mode") |
| `wave10_gate` (gpu + initialization + geometry + race) | The full (non-compact) reply lane, seeds 1..3: the retail lane, the zero-skill control, the fan and `duration_usage` 4 battery, the startup pass, paired streams with a data signature, world geometry, retain-all frames and the competition-reorder roster each come out of `run_batch_full` and `full_reply::build` whole and byte-for-byte equal to the engine's own reply text, frames included (see "Wave-10 full replies") |
| `wave10_folds` (gpu + initialization + geometry + race) | Full-reply provenance folds: mutating one kernel channel moves exactly the arrays decoded from it, every channel carries live records in a gated scenario (pending and immediate activations, rush edges, fan and cross-target selections, the asitame release, kinds 4 and 5, the style-leader and non-leader phase records, snapshot ranks and the spurt columns), and the requests outside the full lane stay refused (see "Wave-10 full replies") |
| `facade_gate` (gpu + initialization + geometry + race) | See "Batch API" below |
| `parity_audit` (gpu + initialization + geometry + race) | Engine parity audit: the app comparison traffic on both lanes compact bit-exact and full byte-equal, all 751 merged skill ids through both compilers against pinned refusal-family lists, and the `gate_reasons` fallback set pinned exactly (see "Engine parity audit" under "Wave 13b") |

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
| `in_u` | u32[384] | mode at 0, scalars at 1..6 (seed, n, paired flag, paired seed, lot min, lot max), css at 7..10, per-runner triple (style, adjusted-wisdom bits, bonus bits) from 16, stride 3 |
| `in_l` | u64[144] | per mode: stats raws 0..4, r30 first-slope start, value and adjusted wisdom 0..2, init distance at 0, per-runner base wisdom 8..127, parameters 128..137, hp stamina..`hp_drain[4]` 0..14 |
| `out_u` | u32[3840] | RetailRandom bank 0..59, paired RUSH bank 60..119, fallback flag at 120, six stage counters 121..126, r30 scalars 128..159, runner rows at `128 + 30 * r` (force, chance, delay, rush chance, lot section, rush present, 24 sections), continuation `next_int` at 3728..3735, hp guts and wisdom factors at 170..171, race-rng bank snapshot at 3744..3803 (see "Race loop port") |
| `out_l` | u64[144] | continuation `next_double` bits at 120..127, init `kakari_start` feed 0..119, hp ground modifier, base speed and max hp at 0..2 |

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
  (`src/api.rs`). The experimental `startup_pass` is ported in the race loop,
  not in this slice (see "Startup pass" below). The v77 request uses neither.
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
"Paired mode" below). Domain gates run before any launch: an invalid
`experimentalGrid` shape flags with the engine's message, a roster beyond the
kernel layout's 120 cap flags as outside the kernel's range, non-retail
runtime flags, and the grid-plus-geometry combination flags with the engine's
refusal (api.rs:68-71). Each flagged race returns its reason instead of
launching wrong
numbers, and skill-bearing fields are gated by the skill compiler (see
"Skills port"). A field of up to `RACE_MAXR` (120) runners launches
unflagged when it has no skills or only skills the compiler accepts. The
experimental startup pass is
ported: `RI_STARTUP` runs one dt=0 tick ahead of the frame-0 capture (see
"Startup pass" below).

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

## Startup pass (`experimentalRetailStartup`)

Engine authority: `run_race` takes the `startup_pass` argument (`src/lib.rs:577`,
parsed as a boolean next to the retail-runtime requirement at `src/api.rs:61-67`),
and consumes it in exactly one branch (`src/lib.rs:831-833`): one extra
`field.step_all(0., all)` call between `RaceField::new` and the frame-0 capture.
At dt=0 `advance_start_delay` (`src/retail_physics.rs:109-121`) returns without
moving anyone, so the pass publishes only the pre-movement field state: duel
and lane collect, `prepare_retail_tick`, the two `process_skill_priority` scans
with live field columns (the loop-pass shape, `src/race_solver.rs:1946`), the
skill-effect flush and the front-blocker sample.

Kernel side: the loop body moved into `static int race_tick(...)`
(`gpu/src/kernels/race.cl:381`), parameterized by `dt`, `snap` and `tick`. The
frame loop calls it with `dt = 0.0666f` and the debug window (`race.cl:1106`).
when `RI_STARTUP` is set the kernel calls the same function once with
`dt = 0.0f`, `snap = 0` and `tick = -1`, after the constructor scan and before
the frame-0 capture (`race.cl:1075`). The frame-0 blocker column reads the live
sample (`race.cl:1090`), which is the one column the pass can put on the wire.
The host parses the flag in `prepare` (`race_exec.rs:601`, a non-boolean flags
the race like the engine errors it) and uploads it as `ri[RI_STARTUP]`
(`race_exec.rs:658`). `facade::gate_reasons` no longer refuses the flag.

What the gates say (`gpu/tests/startup_gate.rs`): the app's case (b) shape, the
mob-preset jobs of courses 10504 and 10604 flagged
`"experimentalRetailStartup": true`, seeds 1..3, twelve races, zero fallback,
all six replay columns identical through `to_bits()` against the CPU engine,
`finishTimeRaw` bit-exact. The stage gate pins the kernel's exported start
delays bit-exact against the in-process engine's own
`retail_initialization::initialize` (vendored at `gpu/src/lib.rs:43-44`) and
against the reply's `startDelayTime`: the mob rosters draw 220 to 234 trigger
draws before the delay stage, which adds exactly one draw per runner, and the
flag moves nothing in that chain.

The engine fact worth knowing: the flag is output-inert on every shape probed,
the twelve flagged mob presets and sixteen v77 and solo cases, engine replies
byte-identical with and without it. The extra dt=0 pass re-evaluates the state
tick 1 evaluates, with dt=0 timers, and consumes nothing the tick-1 pass does
not consume anyway, so the stream stays aligned. The port reproduces the
mechanism, not the outcome: the identity gate covers the flagged engine
races, the plain gate covers the unflagged ones, and `RI_STARTUP` selects the
extra call, so both outcomes are pinned wherever the engine ever makes them
differ.



Skills ride the same race kernel, and the kernel does not change with the
skill set. The host compiles every skill of a race into one flat numeric table
(`src/skills_compile.rs`, `uint[54912]` per race) and the kernel interprets it:
init pass in `src/kernels/skills.cl` (trigger samplers, wisdom lottery, region
intersection) inside `im_init` before the existing rushing and section draws,
runtime in `src/kernels/retail_core.cl` (timer step and cooldown start, then
the priority 1 and 0 fire passes, then the receiver folds). Encoding, the
condition-support matrix, and the unsupported-construct rule are in
`SKILLS_DESIGN.md`.

The per-tick order mirrors `step_all_retail`: `skills_advance` (timers, expiry,
cooldown start), `skills_scan` x2 (per-record any-skill observe at the visit
top, then the fire gate: trigger and window check over the record's final
window list, precondition latch, cooldown readiness, then the per-arm
programs), `sk_columns` (fold the receiver lists into the live columns), and
only then the blocking sample that feeds duel and field. The
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
  patched to 4 beside a `PreferInsideLane` effect (kind 12, whose per-fire aux
  entry is what the overtake extension walks) flags the race before any launch,
  and the reason names `duration_usage 4`. The same alternative without that
  effect compiles, which is the shape the real 100531 and 900531 have.
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

### Wave-1 fallback-histogram constructs

The measured fallback histogram of the app's chart runs named six refusing
constructs on course 10504 (Phase A: 288 fallbacks, 6 reasons x 48 runs) and
920011's multi-alternative shape again in the wider sweep (Phase B: 390). All
six are ported, so both phases' counted fallbacks now compile:

- request-static value_filter fields, folded at compile time like
  `distance_type`: `ground_type`, `track_id`, `rotation` (course constants),
  `ground_condition`, `weather`, `season` (request enums), `is_dirtgrade`
  (request flag, else the engine's dirt-grade track-id list), `popularity`
  and `running_style` (horse). `lane_type` stays refused: it reads live lane
  state, not the request.
- `order_rate_out20/40/50/70_continue`: four order-band cells per runner in
  race scratch, updated each tick from the previous-tick rank once the clock
  passes 5 with the engine's permanent false latch. The rate in the name
  picks the cell and rides in the comparison argument. Only `==1` passes.
  The in-band cells were first refused on the belief that the engine never
  inserts them; `track_order_bands` says otherwise and wave 3 ports them, see
  "Wave-3 condition tail".
- `is_activate_any_skill`: the v75 retail observe model, a per-runner
  cumulative activation counter read through a per-record changed flag
  evaluated at every scan visit (see `SKILLS_DESIGN.md`).
- multi-alternative skills: one record per alternative, folded-empty
  alternatives kept as draw-only dead carriers, one wisdom roll per runner
  and skill id inherited by the followers, shadowed alternatives (same
  condition and precondition signature, no sampled term) dropped with only
  their start-delay factor folded. A second live alternative that is not
  shadowed stays refused by skill id.

Gates: `gpu/testdata/wave1_variant_jobs.json` holds the six exact Phase A
variant requests (9-horse chart jobs, course 10504, player skill swap).
`wave1_gate` runs them over seeds 1..3 (18 races) with zero fallback plus
frame-for-frame `to_bits()` identity against the CPU engine, and runs a
v77-roster race where synthetic defs built from 200361's real alternative
fire the any-skill, shadowed-pair, dead-wisdom-carrier and out-band
constructs (the chart jobs' late-race gates never fire the real golds, 0/200
seeds measured, so the exercise race proves the paths are not vacuous).
`wave1_folds` compiles single synthetic skills and reads the record flags
back: every static fold in both directions (live record for true, DEAD-flag
record for false), the dynamic leaves kept live, the refused names still
named (`lane_type`, `order_rate_out60/continue>=2`, `order_rate_in50`,
`is_dirtgrade>=`, `is_activate_any_skill==2`, precondition anyskill), and the
multi-alternative collapse counts.

### Wave-2 condition fields

The fallback histogram's next tier named the dynamic condition fields real
player-swapped chart traffic still refused (Phase C, course 10504):
`overtake_target_time` (88 seeds), `distance_diff_top` (64), the
`distance_rate_after_random` sampled term (48), `activate_count_heal` (32),
`change_order_up_end_after` (32), then `lane_type`, `compete_fight_count`,
`is_behind_in`, `is_move_lane`, `blocked_front_continuetime` and the
`running_style_count_<style>_otherself` folds (24 each). All are ported.

- Live leaves: `lane_type` (`get_Lane` bands on the live lane), `is_move_lane`,
  `is_behind_in` (the lane-gap sign, last place reports `None` and rejects every
  comparison), `distance_diff_top` (the f32 gap to the leader, own position when
  leading), `overtake_target_time`, `change_order_up_end_after`,
  `compete_fight_count` (`>` only), `activate_count_heal` (`>=` only),
  `blocked_front_continuetime`. `distance_rate_after_random` is a ninth sampled
  term through the init windows. The `running_style_count_*_otherself` names fold
  at compile time from the roster style counts (`Statics.other_style`).
- Scratch: the live clocks and counters ride the runner scratch, which grew to 74
  `SI_` u32 slots and 105 `SD_` f64 slots (the new `SD_OTIME`, `SD_BFRONT`,
  `SD_POSPREV` and the 24-slot `SD_OVC` overtake pair table). `tests/layout.rs`
  pins the word totals the bump lands on: the u32 side holds at `SCR_U_WORDS`
  51932 (the alignment mask absorbs the added slots) and the f64 side moves to
  `SRL_WORDS` 29096. The wave-2b effect tables moved the pins again, see below.
- Two constructor-pass behaviors surfaced once multi-fire skills could be gated,
  both pre-existing gaps the earlier single-effect gates never reached. A record
  whose condition reads a field or retail-loop leaf cannot be judged at time zero
  (the engine's closure reports false with no field), so the compiler marks it
  `SK_FLAG_INIT_DEFER` and the pre-loop scan drops its fire, first firing on the
  loop's first tick. A record that does fire in the constructor pass has no retail
  timer map yet, so its cooldown starts at once as an absolute ready time instead
  of the Waiting-to-instance path a loop fire takes. Both are in `SKILLS_DESIGN.md`
  and reproduced by `wave2_gate`.
- `value_usage` 3 to 14 refused by name in this wave: those usages rescale the
  modifier from stat, win and fan tiers, the activation rolls, the late-corner
  count and the green-tag count, state the numeric table could not carry. This
  was the reason a couple of the histogram's carriers (for example the fans-tier
  skill 210072) stayed refused even though their condition was ported. The whole
  range landed with wave 6b.

The compile-only scratch sweep over the 108 merged skills that carry these
constructs (plus the wave-1 static recheck) compiled 62 at the time;
the rest refused on an unported effect kind (37), a target-another-runner effect,
or a rescaled `value_usage`, each named in its reason. Gates: `wave2_gate` puts
one real merged skill per construct on horse 0 of the v77 roster (seeds 1..3) with
zero fallback and frame identity, and a v77 race where synthetic defs built from
200361's alternative fire every construct (the `otherself` fold has no compiling
real skill, all its users carry an unported target-another effect, so the firing
gate covers the fold there). `wave2_folds` compiles each construct alone and reads
the record flags and roster counts back.

### Wave-2b effect side

The fallback histogram's effect tier: records whose effects left the runner
that owns them. The compiler now accepts effect kinds 1, 4, 5, 6, 8, 9, 10,
14, 21, 22, 27, 28, 29 and 31 with targets 1 (self), 4 (visible), 9 (ahead,
nearest first) and 10 (behind, self first), and refuses by name everything the
kernel cannot run.

- Cross delivery: a queued kind (`TargetSpeed`, `Accel`, `CurrentSpeed`,
  `CurrentSpeedWithNaturalDeceleration`, `LaneMoveSpeedUp`, `Vision`,
  `Recovery`) aimed at 9, 10 or 4 resolves receivers from the rank columns of
  the previous field update and pushes the caster's token into each receiver's
  order. Recovery lands `f32(receiver max hp) * f32(modifier)` in the
  receiver's pending-HP bank. A delivery never registers a receiver-side
  instance: the engine's `apply_external_effect` activates with
  `follow_up = true` and that path returns before the cooldown-map insert
  (`race_solver.rs:1325-1328`, `:2644-2653`), verified against a same-skill-id
  self-plus-ahead delivery that would otherwise couple the two records.
- Construction-time bumps: `GutsUp(4)`, `WisdomUp(5)` and
  `ModifyKakariChance(29)` on a static-fire self record apply host-side to
  the init inputs (the `add_stat_modifier` f32 chain and the kakari-chance
  fold, `PRD` rows 5 to 7, the WisdomUp accumulator in `PRD_WIS_ACC`). The
  record carries `SK_FLAG_BUMPED` so its init-pass fire skips the write the
  host folded. Kind 4 anywhere else refuses ("only ports as a
  construction-time bump"), a self WisdomUp without an all-arms phase gate
  refuses, and kind 4 aimed at another runner refuses.
- `Recovery(9)` self fires heal the count, positive-only and self-side only,
  and bank HP that lands at the step head (a gain in the last two phases
  defers the last-spurt plan, as in the engine). `Vision(8)` folds per tick
  with the engine's `fmax(20 + sum, 0)` shape. `RunAway(6)` and
  `SetStartDelay(14)` fire inert: the RunAway runtime arm is null and the
  retail physics never reads the reassigned `start_delay`. What RunAway
  really does is at build time: a Nige that learned a type 6 effect races as
  Oonige (`race_solver_builder.rs:858-865`), and the prepass applies the same
  upgrade before initialization or conditions see the strategy (202051 on a
  Nige seat pinned this).
- Repeats: several effects of one kind in one record are separate slots with
  separate duration timers, capped at 8 effects per record (the token cap),
  40 live tokens per runner and kind, overflow raising the `FB_SKILL_SEQ`
  fallback. The kind filter in `sk_tok_value` is load-bearing: the current
  order hosts kinds 21 and 22 together, and a fold that counted the wrong
  kind double-booked the natural values.
- Init-defer: engine condition closures that read the field report false in
  the constructor pass, including the order ones, whose rank input the kernel
  rank column does populate at init. `order` and `near_count` joined the
  deferred feature set after a battery bisect caught a same-tick delivery the
  early kernel fire had consumed the cooldown slot for.
- Scratch grew to `SCR_U_WORDS` 58332 and `SRL_WORDS` 30376 (the five token
  orders, the pending-HP bank and the per-effect-slot timers), pinned in
  `tests/layout.rs`.

`wave2b_folds` compiles each shape alone: cross records of every queued kind
live, self-only kinds refuse the same target, 4 and 5 refuse mid-race fires,
static fires record the bump triples, the 8-effect cap fires, and 36, 37 and 38
refuse by name. `wave2b_gate` replays ten real merged
skills on horse 0 of the v77 roster, seeds 1..3, zero fallback, frame
identity, and each skill's id present in the CPU activation events: 201502
(vision), 201522 (wisdom bump plus vision), 110581 (target-speed repeat with
current-speed ahead and behind), 910581 (current-speed pair), 201442
(recovery on the visible field), 202051 (runaway upgrade), 202141
(start-delay plus wisdom bump), 202511 (kakari chance), 202181 (guts and
wisdom bumps), 110301 (recovery self plus ahead, spread over three seats
because its mid-pack condition never holds for the all-Senkou field's horse
0). The battery (synthetic 663001 to 663012, one per path, forced to fire on
order and cooldown-free conditions) covers what the real skills cannot
(accel-behind, lane-speed-behind, natural-speed-behind, vision-debuff-visible).
A compile-only sweep of the 32 ids the wave-2a histogram refused on effect
constructs now compiles 29 (solo horse, best over the seven strategy seats);
the three remain refused `value_usage` 7 or 11, both tiers since ported (wave 6b).

### Wave-3 condition tail

The last condition tier: stateful time machines, the activation history, the
live in-band order cells, the two `distance_diff_*` siblings and the roster
reads. The kernel stayed vocabulary-generic: eight new `SI_` slots, two new
`SD_` slots, one new per-record bit, nineteen new feature codes (38 to 56), no
per-skill branch anywhere.

- Activation history: the phase counters (`SI_ACT0/1/2`, bumped by the fire's
  phase), `SI_ACTLH` (fires at or past half the course), `SI_ACTALL` (already
  there for the observe flag) and a per-record used bit that makes
  `is_used_skill_id` true once the runner has activated that id. `used_skills`
  takes only self-fired ids, because `apply_external_effect` passes
  `follow_up = true` and that path returns before the insert
  (`race_solver.rs:2644-2669`). Two shapes fold instead of running: an id no
  roster horse carries can never enter the set, and
  `is_activate_other_skill_detail==1` reads the record's own id, so the record's
  first fire is the only writer of the bit the same gate blocks: it is false
  forever and the battery asserts the CPU never fires it either. The
  `activate_count_*` comparators are registered narrowly in the engine (lte and
  gte for `activate_count_all`, gte only for the other two) and the compiler
  refuses what the registry does not register.
- `is_activate_heal_skill` lowers to the engine's boolean filter: argument 1
  becomes `activate_count_heal >= 1`, any other argument its false branch
  `== 0` (`activation_conditions.rs:638-641`).
- In-band order cells: the wave-1 note that these never enter the retail band
  map was wrong, and the code says so. `RaceField::update` feeds
  `track_order_bands` in both directions (`race_field.rs:239-246`), so
  `order_rate_in20/40/50/80_continue` are live and ported as four cells beside
  the out-band ones, mirroring the same set/timer/NaN machine on the
  pre-publish rank, the first five clock seconds skipped, and only the `==1`
  comparison reaching the kernel.
- No-order pair: `overtake_target_no_order_up_time` and `..._down_time`
  accumulate beside `SD_OTIME` in the movement loop on the same
  f32-into-f64-into-f64 step, armed by the runner's own lane target count, and
  the rank pass zeroes them on the up event and on any rank drop
  (`race_field.rs:258-276`, `race_field_retail.rs:282-298`). Their condition
  values are the floor of the f64 accumulator, so the timers can be one tick
  ahead of the same-tick pass that would reset them; the engine reads them the
  same way.
- `change_order_up_middle` and `change_order_up_finalcorner_after` count the
  same rank-gain pass into the tick-start phase and the tick-start
  final-corner flag, matching `record_overtakes` reading `retail.final_corner`
  before the movement (`race_solver.rs:1755-1774`).
- `blocked_all_continuetime` truncates `min` of the two existing block clocks,
  and `blocked_front` is the one leaf that must not read the stored sample: the
  engine's getter rebuilds `front_blocker` from live solver rows at evaluation
  time (`activation_conditions.rs:1290-1292`), one tick fresher than the field
  pass's sample. The stored-sample version first fired at all once its carrier
  reached a mid-pack seat, and it diverged on one of the three seeds there; the
  live scan closes it.
- `is_surrounded`, `near_infront_count` and `visiblehorse` are field reads: the
  three-direction membership test on f32-rounded gaps, the near corridor's
  forward half on f64 gaps, and the lane cache's visible count.
- Temptations: `temptation_count_behind/infront` count kakari rows behind and
  ahead in the published order. The opponent variants are identity there only
  while every roster team is zero, which is the engine's `is_opponent`
  (`race_field.rs:913-947`); a team id in the roster refuses them by name, as it
  does the debuff deliveries from wave-2b. The four style temptations lower to
  one leaf with the base group in the argument, counting that group's kakari
  rows including self, exactly the engine's loop.
- `distance_diff_rate` reproduces the ranked endpoints with self substitution,
  the f32 spread percentage floored, and an approximately-zero spread (under the
  subnormal `8 * f32::from_bits(1)`) returning exactly 0;
  `distance_diff_top_float` is the leader gap through `floor(x*10)/10` in f32.
- Roster reads now cover `running_style_count_same`,
  `running_style_count_same_rate` (the 0x35c7d30 `+0.5` f32 truncation of the
  runner-count threshold, and false on an empty field),
  `running_style_equal_popularity_one` (the unique `popularity == 1` horse, its
  strategy group matching the runner's) and `same_skill_horse_count`. A roster
  lookup that finds nothing folds false for every comparator, `!=` included, as
  the engine's `roster_condition` does (`activation_conditions.rs:247-256`).
  The fold reads the same two skill-entry shapes as the wire pass
  (`normalize_skill_entry`), which matters: an entry shape mismatch made the
  count wrong and 201641 replayed with a real divergence on the first try.
- Still refused, and the reason: `behind_time`, `near_time` and `infront_time`
  have no engine registry entry at all (the nearest names are the ported
  `*_near_lane_time` timers), `is_other_character_activate_advantage_skill`
  caches another runner's previous-tick activation, and
  `remain_distance_viewer_id` reads a viewer's remaining distance, which has no
  concept in the field.
- Scratch grew to `SCR_U_WORDS` 58524 and `SRL_WORDS` 30424 (eight `SI_` slots
  times 24, two `SD_` slots times 24), repinned in `tests/layout.rs` (the grid
  capacity port later resized every region to 120 runners, now 450,300 and
  232,400).

`wave3_folds` compiles each shape alone: the four in-band rates live with `==1`,
30 and every other comparator refused, the narrow counters refuse their
unregistered comparators, the used-id fold stays live for a roster id and folds
dead for an absent one, the self-gate folds dead, the roster reads fold both
ways with the 83/84 percent threshold edge and the no-favorite case, the
opponent pair refuses under a team id, and the refused names refuse by name.
`wave3_gate` replays fourteen merged rows on the v77 roster, seeds 1..3, zero
fallback and frame identity: 900561 (blocked_front, carried by all nine horses
because its rank window never holds for one seat), 200622, 910231, 900641,
210131 and 201621 (the two counters, riding a self-firing helper to reach their
thresholds), 910371 (final-corner passes, spread for the same reason), 201641
with two shape probes, 200271, 200291 (horse 0 at popularity 1), 910031 (identity
only, its precondition window is narrow) and 200401 (is_surrounded). The battery
(synthetic 664001 to 664028, one per path) fires every new leaf, holds the used
and heal pair in wire order so the reader follows its provider, proves the
self-gate dead on the CPU, and puts the blocker and surrounded cells on a
mid-pack seat where they can actually hold. Four merged ids still refuse on the
effect side (910151 two alternatives, 200281 effect type 3, 200771 and 200781
effect target 20 and 19, 200801 effect type 13), so their conditions ride the
battery and fold gates.

### Wave-4 effect tail

The effect-side tail of the Phase C histogram: the team-filtered debuff delivery
became a real port, one effect kind gained a path, and the rest refuse by name at
the exact engine boundary the kernel cannot model. Per family, with the engine
authority:

- Team-filtered debuff (was refused unless every team was zero): now a real port.
  `race_field.rs:458-461` drops a debuff receiver who shares the owner's non-zero
  team, applied after the `target_value` cap so no candidate backfills. Each
  runner's fixed team id reaches the kernel in the request word `RR_TEAM`
  (`lib.rs:335` is the CPU source) and is mirrored to `SI_TEAM` at kernel start;
  `skills_fire` skips the same-team receivers off the capped list for a negative
  modifier. For every cross-deliverable queued kind (8, 9, 21, 22, 27, 28, 31)
  the drop condition is exactly "modifier negative", because `is_skill_demerit`
  (`race_field.rs:55-63`) agrees with the sign there and `value_usage` is pinned
  to 1 on delivery. Debuff immunity, the filter's other arm, refused with kind
  38 in this wave and landed with wave 6a.
- `MultiplyStartDelay(10)` at another runner: the retail receiver arm does nothing
  (initialization already applied the start gate), so the record now fires inert.
  SpeedUp, GutsUp and WisdomUp (1, 4, 5) at another runner stayed refused: they
  write the receiver's stats through a full `refresh_stat_effects`. SpeedUp and
  WisdomUp landed with wave 7, GutsUp with wave 8.
- Lane goals 11, 12 (a lane-preference presence flag) and 35 (a last-set
  `lane_target`), read by the passing-lane search (`race_solver.rs:1310-1358`): the
  kernel already threaded a `lane_preference` and `fixed_goal` argument through the
  target search but drove them at zero, and no ported leaf consumed a live lane
  goal, so a real-skill bit-exact gate was not available and the wiring stayed out.
  Refused by name in this wave; ported with wave 6a.
- 13 ExtendKakari (a replace into the rush forced-end and kakari timer, no-op
  unless already rushing), 32 AllStatusUp (a mid-race all-stat bump with no kernel
  consumer), 33 and 34 (no `SkillType` variant at all), 36 and 37 random skill
  activation (a re-activation of another eligible skill via an activation-time
  draw, `race_solver.rs:2677-2737`), 38 debuff immunity: all refused by name in
  this wave; 13 and 38 ported with wave 6a, 36 and 37 with wave 6b.
- `additionalActivate`: refused with the precise reason in this wave. The extra
  activation is never applied on the activation tick; it is enqueued into
  `running_skills` and applied later on the running skill's own turn in the
  ascending-id pass (`race_solver.rs:1841-1941`), with turn-taking counters,
  per-kind caps and a deferred cooldown. Ported with wave 6b.
- `value_usage` 3 to 14: refused by name in this wave. Usages 3 to 7, 10 and 12
  to 13 rescale the modifier from roster team stats, wins, fans and the
  displayed-stat tier (`race_solver_builder.rs:908-939`) and fold at build;
  usages 8 and 9 draw `RetailRandom` at activation (`race_solver.rs:2485-2488`)
  and usages 11 and 14 read runtime counters. All of it landed with wave 6b.
- Two live alternatives (910151): refused in this wave. Every alternative shares the
  skill cooldown key (`race_solver_builder.rs:589`); the first to fire parks it and
  blocks the rest, a shared per-skill state the kernel lacked while it cooled per
  record, so it would let a second alternative fire where the engine does not.
  Ported with wave 6a.

Gates: `wave4_folds` compiles each shape alone: cross debuffs of every queued kind
compile whether the runner has a team or not, a cross-target start-delay lowers a
live inert record, and each refused family names its reason (all-stat bump, no
SkillType variant, targeting another runner, `value_usage` outside 1 to 14); the
lane goal, kakari, immunity, random-activation and alternative-pair shapes
assert live records, gated in `wave6a_folds` and `wave6b_folds`. `wave4_gate` replays the real
merged cross-debuff 910581 on a roster where
the owner shares a team with some runners and not others, seeds 1..3, zero fallback
and frame identity with 910581 present in the CPU activation events. The battery
(synthetic 665001 to 665004) forces every new path both ways: an ahead speed
debuff and a behind speed debuff that keeps self and drops same-team runners, a
positive control delivered to all candidates, and the inert cross start-delay, each
fired on the teamed back seat. Scratch grew to `SCR_U_WORDS` 58588 (`SI_TEAM` plus
the 64-byte alignment of the cached block); `SRL_WORDS` holds at 30424, repinned in
`tests/layout.rs` (the grid capacity port later resized every region to 120
runners, then lane and rush aux cells moved it further, now 479,796 and 233,120).

### Wave-6a lane, rush and cooldown tail

Four wave-4 refusals closed, with no app traffic: the merged skill table is the
traffic. Each construct is gated on a real id from the merged table, and a
synthetic battery covers what the table cannot.

- Lane goals 11, 12 and 35. Two presence countdown cells per runner
  (`SD_GOAL_OUT`, `SD_GOAL_IN`) and a four-slot ordered goal vector
  (`SD_GOALV`/`SD_GOALT`) step and compact in `skills_advance`. While a
  preference cell is present the lane search carries the outside (bit 1) or
  inside (bit 2) flag and both present reads 3, no preference
  (`lane_preference_flags()`, `race_solver.rs:1310-1314`); the same test
  moves the skill lane cost to 1.5 when the preference points against the
  gap. The goal push takes the first free slot and the lane search reads the
  last live entry, the engine's `.rev().find` at `race_solver_retail.rs:766-771`,
  with the passing-lane extra and the non-pass clamp of `retail_lane.rs:400-421`
  consuming both arguments the search already threaded.
- ExtendKakari 13. On a rushing receiver `set_forced_end` replaces the forced
  remaining with the effect value (`race_solver.rs:2584-2590`) and the forced
  stretch skips the natural clocks and the per-tick end-percent lottery
  (`retail_states.rs:154-158, 178-181`). Target `21` resolves the rushing
  runners of the strategy group carried in `target_value`, uncapped and
  without self, Nige and Oonige in the same group (`race_field.rs:439-445`).
- Debuff immunity 38. A presence cell `SD_IMMUNE` gates the target filter's
  immunity arm (an immune receiver is dropped ahead of the same-team test)
  and the owner's own debuff fires skip self while present
  (`race_solver.rs:2526`, `race_field.rs:458-461`).
- Shared cooldown key. The two-live-alternative refusal is gone: every
  alternative of a skill id rides one cooldown key
  (`race_solver_builder.rs:589`), so a fire fans its parked state and both
  timers over the ready records of the (runner, skill id) group. READY and
  COOLDOWN-with-expired-timer both read ready, the same pair the engine's key
  map reports (`race_solver.rs:2371-2435`); WAITING records keep their own
  running instance, and DEAD or DISABLED ones are skipped. The wave-1 rules
  stand unchanged: the wisdom roll stays once per (runner, skill id), the
  priority draws share one roll, and a shadowed alternative still folds into
  the first live record's start-delay multiplier.

Three engine-read bugs the gates caught, each a real divergence:

- The `phase` condition reads the runtime phase, not the clamped solver
  field. Retail keeps 3 while `RaceSolver::phase` clamps at 2
  (`condition_parser.rs:170`, `race_solver_retail.rs:1311-1312`). The leaf
  moved to `FEAT_RPHASE` 57 reading `SI_RPHASE`; `FEAT_PHASE` keeps the
  clamped read for the last-spurt and phase-half constructs.
- `running_style_temptation_count_*` compared the count against the strategy
  group instead of the threshold: the program slot's one f64 carried the
  group. The group now lives in the feature id (`FEAT_RTEMP` 58 to 61 and
  `FEAT_RTEMP_OPP` 62 to 65, the opponent arms dropping self like
  `race_field.rs:945`) and the slot carries the threshold.
- `SK_FLAG_RETAIN` was set for every alternative of a retaining rarity while
  the engine keeps only `detail_index == 0` (`race_solver.rs:2243`), so a
  second alternative could survive the scan removal that the engine does not.

Gates: `wave6a_folds` compiles the shapes: lane goals and immunity compile
live for self and cross targets, target `21` is live only with kind `13` (a
kind `27` at target `21` refuses), and the merged 910151, 201261 and 200801
shapes lower with the expected live, shadowed and dead records. `wave6a_gate`
replays on the v77 roster, seeds 1..3, zero fallback and frame identity:
910151 on three runners fires through the shared key and the second
alternative takes the fire exactly when the key reopens, 201261 on three nige
seats fires the fixed goal with the lane column tracking it, 200801 on nine
senko seats fires the extend onto rushing allies, and the battery
666001 to 666010 fires all ten ids on all three seeds: kinds 11 and 12 both
ways, 35 self and delivered, 13 self and group, the immunity drop pair (an
immune ally survives the same-team drop that takes the plain ally, and the
immune host's own debuff lands outside the armed window and skips inside it),
and the shared-key pair firing both alternatives across the runtime-phase
boundary. Kind 38 has no real id in the merged table, so its gate is
battery-only.

Scratch: `SD_SLOTS` grew from `2 * RACE_MAXR + 59` to `2 * RACE_MAXR + 70`
(the eleven aux cells per runner), `SCR_U_WORDS` holds at 450,300 and
`SRL_WORDS` moved 231,080 to 232,400, repinned in `tests/layout.rs`.

### Wave-6b activation-time values and turns

The last skill-side refusals closed, still with the merged table as the traffic.
Three families landed: the activation-time `value_usage` tiers, the
`additionalActivate` turn machinery, and the random-skill reservations (kinds 36
and 37).

`value_usage`, per tier, with the fold or read point:

| usage | authority | how it lands | gated on |
| --- | --- | --- | --- |
| 1, 2 | level table | compile fold, unchanged | wave 2b |
| 3 to 7 | ranked team stat, bounds 1200 1800 2600 3600, factor 0.8 to 1.2, sentinel 99999 gives 1.0 (`race_solver_builder.rs:908-939`) | compile fold `value_fold` (`skills_compile.rs:1157-1213`) | `wave6b_folds` |
| 8, 9 | `rng_uniform(100)` at each application, `<=60` gives 0, `<=90` gives 0.02, else 0.04 (`race_solver.rs:2485-2488`) | runtime draw in `sk_vu_factor` (`retail_core.cl:3669`) on the same bank the fire's other draws use | 202031 and 202032 real, 667010 battery (no real 9 id in the table) |
| 10, 12 | career wins 6 14 18 25, fans 20000 50000 100000 160000 | compile fold, same table | `wave6b_folds` |
| 11 | late-corner overtakes, tiers 2 3 4 (`race_solver.rs:2500-2503`) | runtime read of `SI_LCO`, bumped in `field_update` when an overtake lands inside a corner in runtime phase 2 or later | 910341 real |
| 13 | highest raw stat, bounds 600 800 1000 1100 | compile fold | `wave6b_folds` |
| 14 | activated green-tag id count, tiers 3 5 6 giving 0 1 2 3 (`race_solver.rs:296-301`) | runtime read of `SI_GREEN`, a per-runner set mirror bumped once per id at fire | 100981 real, trio 667012 to 667014 carries the tags |

Every read happens at the same point in `skills_fire`'s effect loop as the
engine's, before that tick's movement draws, and the roll consumption is per
application: an additionalActivate follow-up re-draws exactly like the engine's
second `apply_activation`. The real-id gates are the proof.

`additionalActivate` mirrors the engine's deferral exactly
(`race_solver.rs:1841-1941`, 2443-2461, 2642-2648): a slot with a non-zero count
counts into the instance duration but never applies on the fire tick; the fire
pushes the record onto the runner's running list (8 slots), and each activation
elsewhere files one pending turn per running skill (`additional_skill_event(2)`,
kind-2 or kind-3 slot required, own id excluded), as does each overtake
(`additional_skill_event(1)`, kind-1 slots, three per slot, durations collected
before delivery). The priority-0 scan walks the pending records in the engine's
order and, at each record id boundary, delivers each running skill's queued
turns before moving on, with one final sweep per runner at the pass end: up to
`min(count, cap - already)` applications, cap 3 for kind-2 slots and 2 for
kind-3. `aa_retire` drops a running entry only past
its timer and past the id floor, and defers any entry with a kind-2 or kind-3
slot while a later entry is unretired, the engine's `retiring` flag. The
follow-up applications register their value and duration in a 128-entry pool
whose tokens join the same per-kind orders, so `sk_tok_value` reports them
without reading the wire. Gates: real 910211 (aa 3, gated by
`activate_count_middle >= 2`, lifted by inert carriers 667120/667121) and
910351 (aa 2, last-spurt window); battery 667001 (two aa-3 follow-ups, one
turn), 667002 (four aa-2 follow-ups, the cap strands one and the entry never
retires) and 667003 (aa 1, overtake-driven). No self-target aa id with
`durationUsage` 1 exists for kind 1, so the kind-1 delivery arm stays
battery-covered, and 100531 keeps refusing for its refused `durationUsage` 4.

Random activation, kinds 36 and 37. The effect stores its request count
(`race_solver.rs:2677-2684`, never during the constructor pass); at the end of
every priority-0 pass `aa_select` rebuilds the candidate list (primary
alternative, general skill, every effect `base_duration >= 0` and kind above
5 and not 32, rarity gold or evolution and also white for a 36 request, key
ready, no live instance, id not already running), draws one index per requested
pick and removes it, and marks the winners. A marked record fires on the next
visit bypassing the arms, the windows and the extra condition, which also
overrides a failed wisdom lottery: a retained primary that lost the roll stays
pending exactly for this path (`race_solver_builder.rs:1002-1005`), so the scan
keeps DISABLED retained primaries as candidates and the record returns to
DISABLED, not READY, after such a fire (`SKR_WISFAIL`, `skills.cl:198`).
Gates: real 910071 (kind 37, gold request at the half-distance point, gold
generals 200331/200361 as the pool); battery 667020 (kind 36) with the
never-fires whites 667021/667022 as the pool, so their appearance in the
activation events is the reserved path's own proof.

Three real bugs the gates caught:

- The fold readers (`sk_has_speed`, the target column, the physics chains) read
  the wire modifier of a queued record token, so a usage whose factor is 0
  (real 100981 at green count 0) still moved the target speed while the engine's
  active-skill entry held 0. Every fired application now stages its applied
  value per token (`sk_tok_stage`, `retail_core.cl:3084`) and the readers take
  it, matching the pool tokens' behavior.
- The host scratch mirror pinned the scalar block at 83 slots per runner while
  the kernel grew to 89, so every region after it sat 720 words short of the
  kernel's view and late-runner lists wrote past the buffer. `race_exec.rs:146`
  now mirrors `SI_SLOTS`.
- The any-request pool required both the gold and the white rarity bit, an
  impossible mask, so no kind-36 pick was ever reserved. The filter now asks
  for one of the pool's rarity bits (`retail_core.cl:4452-4464`).

Scratch: the record state grew 8 to 20 words (aa counters, the reservation and
wisdom-fail flags), the scalar block 83 to 89 slots (`SI_LCO`, `SI_PENDN`,
`SI_GREEN`, `SI_AANU`, `SI_AARET`, `SI_RESRV`), plus the 128-entry token pool,
the per-runner running lists and the staged value table. `SCR_U_WORDS` moved
450,300 to 479,796 and `SRL_WORDS` 232,400 to 233,120, repinned in
`tests/layout.rs`.

Sharp edges: the additionalActivate turn order assumes the scan visits records
in wire order, which equals the engine's `first_index` order, true while the
builder pushes alternatives in request order; a reserved fire of a DISABLED
record promotes it into the live state machine, and `SKR_WISFAIL` keeps it from
re-arming through the normal gate; a stalled entry that never retires holds one
of the 8 running slots for the rest of the race, and more than 8 such records
for one runner flag the race (same for the 128-entry pool and 16 candidates per
pass); the other-advantage classifier stays unmodeled as before (no merged
consumer of the aa-1 window is reachable with real ids).

### Wave-7 mid-race stat fires

The last skill-family refusals, cross SpeedUp and WisdomUp and the self
WisdomUp mid-race fire. GutsUp landed later, self and cross, in wave 8.

- A self fire writes the accumulator and the clamped field only: the
  engine's self path returns without a refresh (`race_solver.rs:2644-2674`).
  A mid-race WisdomUp self fire needs every arm gated false at phase zero,
  so the constructor pass cannot fire it and move the field ahead of the
  kernel's baked init draws and section rolls. The kernel's new `SD_HPW`
  column carries the live wisdom field, read by the pace and downhill draws
  each tick (`race_solver_retail.rs:899` and `:1118`), and the spurt plan
  keeps its `SF_HPW` capture until a refresh, as in the engine.
- `PRD` row 8 (`PRD_WIS_ACC`) carries the unclamped wisdom accumulator when
  a static WisdomUp bump folded into the init inputs, NaN when none did, so
  a loop fire chains onto the construction bump instead of the clamped
  field. `PD_RUN_STRIDE` grew 8 to 9 (`RIL_STRIDE` 2072 to 2192, repinned
  in `tests/layout.rs`), the two new SD columns (`SD_HPW`, `SD_HPW_A`) fit
  the existing SD slot budget, and `SCR_U_WORDS`/`SRL_WORDS` are unchanged.
- The cross delivery writes each receiver's stat (SpeedUp also repatches the
  cached spurt maximum from the bumped speed, the value the engine's plan
  closure recomputes at the next spurt decision) and then runs the
  delivery's refresh, `sk_stat_refresh`: the target base rewrites to the
  legacy f64 formula (`race_solver.rs:43-60`) from the clamped speed field,
  HP takes the init-plus-spend f32 re-round, and the spurt-plan wisdom
  capture recaptures from the live field. The refresh's `min_speed`, legacy
  `last_spurt_speed` and `base_accel` rewrites are skipped as retail-dead:
  the step head recomputes the retail minimum and the phase-2 spurt before
  any consumer reads them (`race_solver_retail.rs:961` and `:1007-1034`),
  the sole pre-phase-2 spurt-speed reader is the unmodeled staminasyoubu
  hook, and the accel chain recomputes from power.
- GutsUp stayed refused at any mid-race fire in this wave: the spurt maximum
  term, the duel entry and the spot entry read `powf(bounded_stat(guts) *
  spurt[2], spurt[3])` from the runtime guts at every plan recompute
  (`retail_physics.rs:362`, `race_solver.rs:2928-2939` and `:3194-3207`),
  and the kernel held that term as the one prepare-baked `PRD_POWF` word
  per runner. Wave 8 replaced the baked word with bit-exact in-kernel pow
  ports that recompute at those same reads.
- Gates: `wave7_folds` compiles each shape alone. `wave7_gate` replays the
  v77 roster seeds 1..3 with zero fallback: self wisdom on a 15 s refire
  and a phase-2 drop (668001/668002), static bump then loop fire on one
  runner (668030/668031), all three cross fans on both kinds (668010 to
  668012 and 668020 to 668022, the self-inclusive behind fan refreshes the
  owner), a 10 s refire cycle (668050), a negative cross wisdom dropped at
  selection by the immune runner and the same-team runner while the rest of
  the field takes write and refresh (668041/668042), and the real kind-1
  loop fire 201641 beside the constructor-pass SpeedUp probe 668060, whose
  fire the CPU event stream cannot see, so frame identity pins the
  constructor ordering. No merged id fires kinds 1, 4 or 5 mid-race or
  cross, so the battery carries those paths.

### Wave-8 live guts

The final guts refusals: a mid-race self or cross `GutsUp(4)` fire now moves
the live accumulator and field, and every guts-derived `powf` word recomputes
from the field at the points the engine reads it.

- The prerequisite was bit-exactness. The engine's terms are Windows UCRT
  `pow` and `powf` calls, so the kernels needed the same bit results, not a
  nearby value. `powf_core.cl` carries transcriptions of the live ucrtbase
  FMA variants: the f64 path (`gs_pow`, the double-double log and 257- and
  64-entry table pipeline) and the float-tuned f32 path (`gs_powf`, whose
  result differs from the f32 rounding of `pow` on 8,621 of 99.8M engine
  pairs, so a dedicated port was required). `wave8_powf_gate` demands
  `to_bits` equality against Rust `f64::powf` and `f32::powf` over the
  engine's guts domains (all f32 values in [1, 2000] at stride, clamp bands,
  log-uniform randoms and the battery's duration, spot and duel shapes):
  38,675,241 pairs at default settings, 419,395,547 widened (env), zero
  mismatches on every domain. `cl` `pow`/`powr` miss 2.02M pairs per domain
  and were replaced, not kept. The kernel side needs `fma()` single-rounding
  and FP_CONTRACT off, both already in force.
- The compiler gates the switch. A kind-4 arm that can fire outside the
  constructor pass (mid-race self, gated, or any live cross) sets the plan's
  `live_guts` flag; prepare bakes it into the race word `PD_LIVE_GUTS`. A
  static-only kind-4 catalog (all shipped merged ids are static bumps) keeps
  the flag zero, and the race reads the prepare-baked words exactly as before,
  so the guts-static control is the flag-off path itself. The self phase gate
  rule from wave 7 extends to kind 4: an ungated guts arm could fire in the
  engine's constructor pass ahead of the baked init draws.
- The kernel's live reads mirror the engine's read sites. `SD_HPG`/`SD_HPG_A`
  carry the field and the unclamped f32 accumulator, seeded from the
  construction bump through `PRD_GUTS_ACC` (NaN seed means none) like the
  wisdom pair. A self or cross bump writes both, then repatches the two
  step-head caches: `SF_BMIN_VAL` (minimum is constant in guts while the
  field holds, so repatching equals the engine's per-tick recompute) and
  `SF_SPURT_MAX` through `sk_powf_word`. Duel entry snapshots both modifiers
  into `SD_DUEL_T`/`SD_DUEL_A` (the engine captures at activate), spot entry
  snapshots the modifier into `SD_SPOT_MOD` and the duration into the timer
  (proficiency rides `PRD_SPOT_PROF`), and a later guts change never touches
  an active token. Without the flag every one of those reads the old `PRD`
  word instead.
- `PD_RUN_STRIDE` grew 9 to 11 (rows `PRD_GUTS_ACC`, `PRD_SPOT_PROF`,
  `RIL_STRIDE` 2192 to 2432) and `SD_SLOTS` 76 to 81 (five new
  `RACE_MAXR` columns: `SD_HPG`, `SD_HPG_A`, `SD_SPOT_MOD`, `SD_DUEL_T`,
  `SD_DUEL_A`); `SCR_U_WORDS` is unchanged at 479,796 and `SRL_WORDS` moves
  233,120 to 233,720, repinned in `tests/layout.rs`.
- Gates: `wave8_powf_gate` above. `wave8_folds` audits the shipped catalog
  (13 of the 16 GutsUp ids compile as baked bumps; 202441 and 202442 keep
  their pre-existing `random_lot` condition refusals, and 210141, whose
  `motivation>=4` gate the wave-9c fold closes, folds dead on the default
  mood instead of refusing) and the flag matrix. `wave8_gate` replays
  the v77 roster seeds 1..3 with zero fallback: 669001/669002 self refire
  and phase-2 drop, 669030/669031 static bump chained into a loop fire,
  669010/669011/669012 all three cross fans, 669041/669042 the immune and
  same-team debuff drops on a negative guts fan, 669009 the constructor-only
  static control, and the real static 200211, 202181 and 1200051. The
  untouched wave-7 batteries are the other half of the guts-static pin:
  same binaries, same kernel, same identity.
- Perf: an ignored `wave8_perf_m96_live_vs_baked_walls` probe times one
  96-race batch with a live GutsUp carrier in every race against the baked
  batch: live 1720 and 1713 ms, baked 1702 and 1704 ms (two runs, B580
  warm), so the live path costs at most about 1%. It costs nothing per tick
  (the pow ports run only at bump, duel and spot events); the flag adds one
  branch at each guts-word read.

Sharp edges: the first cross SpeedUp flips the receiver's target base from
the retail form to the legacy form for the rest of the race, exactly as the
engine's refresh does (`race_solver.rs:1332`); the refresh count must equal
the delivery count because the HP re-round re-runs per (effect, receiver),
an idempotent round trip the kernel mirrors one for one; a self WisdomUp
leaves the spurt-plan capture stale until the next cross delivery refreshes
it, engine behavior the gates reproduce; the duel and spot `PRD` words stay
exact while guts holds, which was the wave-7 state, since wave 8 the entry
snapshots move to `SD` columns when a live guts fire is in the race.

### Wave-9 running instance, fan and pass-through values

Three more compiler refusals closed, and one shape moved from "refused" to
"compiles, and the app no longer sends it to the CPU".

- `duration_usage` 4 keeps a running instance. The fire writes `SKD_INSTANCE`
  (the `random_active_details` countdown, `race_solver.rs:1100-1101`) and
  `SKD_RUN_T` (the `RunningSkill.timer`, `race_solver.rs:2655`). Each of at most
  three overtakes subtracts `course.distance / 1000` from `SKD_RUN_T`, from every
  token timer the owner holds for that record across the five orders and the
  `additionalActivate` pool, and from the copy a shared-key sibling parked
  (`race_solver.rs:1784-1789, 1824-1839`). The retire sweep, the kind-1 event, the
  kind-2 and kind-3 turns and the kind-2 queue gate read `SKD_RUN_T`, because the
  engine reads `RunningSkill.timer` there (`race_solver.rs:1804-1822,
  1841-1876`). `aa_flip` and `aa_select` keep reading `SKD_INSTANCE`, which the
  extension never reaches, matching `is_expired`
  (`race_solver.rs:2404-2444`). The engine's entry filter is
  `effect.skill_id == runtime.skill_id && effect.perspective == runtime.perspective`,
  so a copy delivered to another runner is never extended, and the count cap is
  three per instance. `100531` (gold) and `900531` (its white twin) are the merged
  table's only `duration_usage` 4 ids and now compile, so the app acceptance path
  that forced those two to the CPU engine no longer needs to. Kinds 11, 12, 35
  and 38 still refuse under `duration_usage` 4: the extension walks the aux entry
  those kinds register per fire, and the kernel keeps one presence cell per
  runner for them, so there is nothing to extend.
- `additionalActivate` outside its caster. A follow-up re-runs the receiver pick
  at its own moment, so `aa_apply` re-runs `sk_targets`, the immunity skip and the
  same-team drop per application and files one pool entry per receiver
  (`race_solver.rs:2479-2537`). Kinds 21, 22, 27, 28 and 31 now compile at targets
  1, 4, 9 and 10. Those kinds at other targets, and every other kind, keep
  refusing: a follow-up registers one token holding one value, and the rest write
  per-receiver state. No merged id carries `additionalActivate` outside its own
  runner, so the fan shapes are battery-only.
- `value_usage` outside 1 to 14 is a pass-through, mirroring the engine rather
  than inventing a rule. Neither the builder's tier fold (`race_solver_builder.rs:908-939`)
  nor the runtime draw (`race_solver.rs:2485-2503`) has an arm or an error for
  those values, so the raw modifier is what lands. The range gate is gone from
  `skills_compile.rs`, and a wire value of 0, 15, 2.1 billion or negative now
  compiles and applies `modifier * 1`.

Scratch is unchanged: `SKD_STRIDE` 148 and `SKR_STRIDE` 20 are the wave-8 values,
the running-instance timer and its extension count fit in the slack of the
per-record doubles block, and `tests/layout.rs` still pins `SCR_U_WORDS` 479,796
and `SRL_WORDS` 233,720.

Gates: `wave9_folds` asserts the du4 lowering (effect word 3 is 4, the running
instance and its extension slots), the four aux kinds that refuse under du4, the
aa fan matrix live and refused by name, the pass-through at eleven wire
`value_usage` values, and that 100531 and 900531 plan live. `wave9_gate` replays
the v77 roster seeds 1..3 with zero fallback: 100531 and 900531 on four seats,
669101 (du4, no delayed effect), 669102 (an extended `additionalActivate`
sibling), 669103 (a kind-2 turn on an 8 s cooldown) and 669148 (a cross-target
fan under du4), each beside a 669141 carrier that lifts the field; 669201 to
669205 for the fan outside self on kinds 27, 31, 28, 22 and 21 across the ahead,
behind and visible windows and the same-team drop; 669301 to 669305 for
`value_usage` 15, 0, 16, -2 and 2147483647.

`wave9_gate` also holds `fan_churn_tracks_the_engine`: six hosts refanning one
cross-target `additionalActivate` 1 effect on short cooldowns, over a pool that
holds one entry per receiver per delivery. The scene pins two pool semantics
the single-fire shapes never exercise. Retail delivery is inline
(`race_field.rs:472-477`) inside the host's own prepare pass, so a receiver
whose roster index is above the host's has not yet stepped its timers that
tick: the kernel gives such a copy its first f32 timer step at delivery
(`race_field_retail.rs:221-225`, `race_solver_retail.rs:825-827`), and the copy
expires one tick before an identical copy delivered to a lower-indexed
receiver. The three-delivery cap counts per running instance, so the fire path
zeroes the delivery counters when it creates the instance, mirroring the fresh
`delayed` counters each activation builds (`race_solver.rs:2474, 2525`). A
ten-seed frame sweep of the same field runs in `tests/wave9_fan_trace.rs`.

Sharp edges: the demerit classification of a turn's fan reads the wire modifier
(`race_solver.rs:2482`) while the direct-fire path classifies from the drawn
value, and `aa_apply` follows that split; a cross-target fan allocates one pool
entry per receiver against the fixed 128-entry arena, so a wide fan on a wide
field is what `FB_SKILL_SEQ` protects; the kernel holds one instance slot per
record, so an engine `running_skills` with two entries for one (id, detail) is
not representable (no merged id reaches it, since one fire registers one entry);
the extension of a shared-key sibling's parked instance copy (the sibling loop in
`aa_extend`) has no gate: it needs a `duration_usage` 4 skill with two live
alternatives, and none of the 751 merged ids has one;
the batched extension relies on `extend_first` running the whole overtake batch
before the retire sweep, which is what the engine does
(`race_solver.rs:1776-1798`).

### Wave-9c condition-name tail

The last skill-side fallbacks in the merged table were condition names the
compiler refused by name. All are ported except one, which refuses on
equivalence.

- `grade` and `time` fold the request enums. An absent key takes the spec
  default, `g1` and midday (`lib.rs:284-285`); a present unparsable string
  fails the engine's own parse, so the compiler refuses it by name.
- `motivation` folds `clamp(mood, -2, 2) + 3` (`lib.rs:505`), and
  `base_power` the ranked wire stat: overcap and team rank bonus, truncated
  `as i32` (`race_solver_builder.rs:872-881`). The sibling reads
  `base_speed`, `base_stamina`, `base_guts` and `base_wiz` stay refused, no
  consumer yet.
- `post_number` mirrors the client gate block from the wire `gateIndex`
  (absent means the roster index, `lib.rs:311`) over the roster size, on the
  three comparators the engine registers, `== <= >=`
  (`activation_conditions.rs:1002-1039`).
- `corner_random` is sampled term id 9: the argument resolves a course corner
  row (the index-tagged row, else the legacy `rows.len() + arg - 5` slot) and
  its `start..end` is the one window; a non-integer tagged index refuses
  (the engine's init error) and an unresolved row folds the arm dead. Each
  arm resolves and draws its own window, so the four-arm shape 200351 carries
  (`==1@==2@==3@==4`) compiles, and 200351 fires in the gate.
- `remain_distance_viewer_id` is a live leaf (feature 66): course distance
  minus `floor` of the viewer row's position, the viewer from the request
  `playerHorseIndex` (default 0) through the header word. An out-of-range or
  non-integer present value fails the engine's spec parse (`lib.rs:292-300`),
  so it refuses, and the leaf defers the record out of the constructor pass.
- `is_other_character_activate_advantage_skill` stays refused on equivalence:
  the engine reads a field-wide previous-tick `(runner, kind)` advantage pair
  set with same-team exclusion, swapped at every field update
  (`race_field.rs:373-398`), and no per-runner kernel flag mirrors it.
- Roll-gated constructor bump. The merged table also proved a bump path that
  only a kernel-side bake can carry: a static-true constructor fire behind
  the wisdom lottery (`210141`: motivation gate, cooldown 0, permanent, self
  stamina, guts and wisdom bumps). The host cannot know the roll's
  outcome, so it pushes no bump and flags the record
  `SK_FLAG_INIT_BUMP | SK_FLAG_BUMPED`. The kernel replays the fire once
  enabled: `sk_init_wis_bump` inside `im_init`, after `sk_init_wisdom`, moves
  the enabled records' `WisdomUp` modifiers onto the section and rushing
  wisdom input (the engine's constructor fires before
  `initialize_sections_and_rushing` and hands it the bumped `horse.wisdom` at
  `lib.rs:760`), and `sk_init_stat_bump` in the race kernel replays the
  kinds' `2`, `4` and `5` effects onto the stat output columns before the HP
  chain, so every later seed starts from the bumped field. A failed roll
  leaves the record disabled and nothing moves, matching the engine not
  installing the skill. The same shape without the lottery keeps the
  pre-existing host-bump path.

Gates: `wave9c_folds` compiles each direction alone, the folds true and
false, `post_number` on its three comparators and the refused fourth, the
corner row resolution live, refused and arm-dead, the viewer leaf live with
its header word and the out-of-range refusal, the gated bake matrix
(`INIT_BUMP` set with no host bump only when the roll gates kinds 2, 4 or 5),
and the advantage-skill refusal. `wave9c_gate` replays the v77 roster seeds
1..3 with zero fallback and frame identity on the real ids 100471, 900471,
200251, 202231, 202331, 210141 and 200351, and on a 6694xx battery for the
directions the real ids cannot reach on this roster (the false-side
post_number fold, the true sides of time, grade, motivation and base_power
as activating mid-race fires, the viewer-row distance, the corner sampler
with a missing row and with four arms, and the StaminaUp kind static, mixed
and roll-gated). The battery's mid-race rows must and do activate. The
earlier wave suites reran green on the same binaries.

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
`run_batch_full(requests, seeds)` answers with the engine's whole non-compact
reply text per race instead of meta and packet (see "Wave-10 full replies"),
and `run_repeated(request, seeds)` answers one request over many seeds after
preparing it once (see "Batch cost").

`register_geometry(course_id, json_text)` makes the app's registered course
geometry the kernel's authority for the course: `prepare` flattens the
injected copy and only falls back to the `data/course_geometry` files when
nothing was registered, which is what standalone tests and tools use. The
app's CPU lane keeps its own `register_course_geometry` flow with the same
text, so kernel and CPU lanes answer from one geometry instead of two trees
of files that can drift apart. Injecting replaces any earlier copy (file or
injected) from the next prepare. The app-side wiring is the `register_geometry`
export in the app's `globalsim-native` addon.

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
time sum. A classic-lane packet goes through `build_classic` instead, whose
frame `k` carries `classic_loop_end(k + 1)`, the engine's f64 sum of `1 / 15`
with no pre-loop frame (`src/lib.rs:581`, `:939`).

A race a gate rejects (an invalid `experimentalGrid` shape, which names the
engine's own message, grid combined with the geometry flag, non-retail
runtime, a roster outside 1..120, a skill construct the compiler refuses, an
event log past `EV_MAX`
512, or a request that fails to parse) comes back as `fallback` with meta
and packet None, and the batch's other races run unaffected. A classic-lane
request the classic gates reject comes back the same way, naming the classic
refusal. A valid grid
race runs on the GPU and its meta carries `experimentalGridVersion: 1`, like
the engine's reply (`src/lib.rs:1135-1138`).
`gate_reasons(request, device)` answers the same refusals as a pure
function, so a caller can pre-filter against known device facts before
opening one.

Constraints. One `GpuRaces` per thread: it owns the OpenCL queue and
program and is not Sync. Every launchable race in one `run_batch` shares one
runner count, the kernel's uniform grid constraint, and a mixed batch is an
Err for the caller to split. CPU replays stay on the CPU lanes. The compact
meta carries no `skillDetailEvents`, `competitionEvents` or
`positioningPhases` and asks for none of them; the full lane answers those
from kernel channels and refuses what it does not carry, listed in "Wave-10
full replies".

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

Measured on the B580 after the grid capacity port (scratch was 3.65 MB per
race slot at every roster then, 5.91 MB at the wave-13b cap), v77 line 1
through `multi_request`
(course 10504, 9 runners, one course per batch; `--bin batch_profile`
prints the tables):

- Program builds are per `RaceContext::open`, not per call: context open
  measures 18.7 ms total (gate with its `restricted_tick` program 5.4, race
  program 8.4, init program 3.4, engine data 1.5), first open in a process
  longer on a cold driver cache. The 1.3 MB race binary costs a few ms at its
  first dispatch, never per launch. The stats pre-pass used to run
  `initialization::run`, which opened a fresh gated device and rebuilt the
  initialization program every call (about 7-9 ms warm, 700 ms cold); it
  now launches through the context's prebuilt program and costs about
  0.3-4 ms.
- The geometry registry loads and flattens each course once per context
  (about 4-9 ms cold, an injected copy re-flattens on the next prepare after
  a re-register), then clones the flat tables per prepared race, and one
  launch dedups the courses into shared tables (about 1 ms at 96 races of
  one course).
- The fixed cost of a warm call is dominated by the race kernel itself: at
  m = 1 enqueue takes 0 ms and the device run about 434 ms. It scales with
  the race's tick count (course 11401, 1000 m: 712 ticks, 208 ms), not with
  the race count (marginal about 3.8 ms per race to n = 96), and not with
  buffer churn (ten back-to-back launches on the same buffers cost about
  432 ms each). Batching is therefore the lever, and every facade entry
  already launches once per batch.
- Per-seed marginal at n = 96 (64 to 96 step): `run_batch` about 11.3 ms per
  seed (prepare about 3.7 including the per-request parse, upload about 2.4
  for the 3.65 MB scratch plus 2.5 MB frames, readback about 0.8, meta and
  packet about 4.7, launch already saturated at this roster), wall 2002 ms.
  `run_repeated` removes the prepare and the parse: about 5.8 ms per seed
  marginal, wall 1353 ms. Before the capacity port the marginals were about
  9.8 and 6.3 ms and the walls 1372 and 1034 ms: the upload grew with the
  scratch and the allocator works harder around the bigger buffers, while the
  kernel itself costs what it cost.

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
- `a_startup_request_runs_on_gpu_and_its_siblings_stay_whole`: a startup
  request between two runnable siblings now runs on the GPU (meta and packet)
  with no fallback, and its siblings keep theirs.
- `gate_reasons_names_every_refusal`: every refusal string against
  synthetic device facts (fp64, single and double fp-config bits, and the
  no-device case), then the grid shape (invalid, and a valid 21-horse grid
  that no longer refuses), runtime and roster refusals for the
  request. No device is opened.
- `run_repeated_matches_run_batch_over_64_seeds`: the v77 line-1 request
  (course 10504, 9 runners) with seeds 1..64 through `run_batch` on the
  request repeated and through `run_repeated`: every meta string, packet
  byte and fallback identical.
- `run_repeated_matches_a_mixed_two_course_batch`: courses 10504 and 10604
  (9 runners each, so launchable in one batch) plus a startup-flagged and an
  unparseable request, `run_batch` on the mixed batch against
  `run_repeated` per request: all four groups byte-identical, the startup
  group now running on the GPU and only the unparseable group carrying the
  fallback arm.
- `injected_geometry_outranks_the_file_copy`: the v77 line-1 race against a
  geometry file whose nearest kind-0 corner event distance is shifted by
  +30 m. The kernel's answer to the injected copy differs from the
  file-loaded baseline (meta and packet), and re-registering the original
  file text restores the baseline byte for byte: injection is the
  authority, it replaces, and the file only serves unregistered courses.
- `injected_app_vendor_geometry_keeps_the_race_bit_exact`: the app's vendor
  copy of 10504 (sibling main app checkout, override with
  `GLOBALSIM_APP_VENDOR`) must equal the engine copy, and with that vendor
  text injected the v77 line-1 race stays byte-for-byte frame-exact against
  the CPU engine.

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

## Wave-10 full replies

The refusal "full (non-compact) replies" is closed. `facade::run_batch_full`
and `full_reply::build` answer each race with the engine's whole reply text,
byte for byte, frames included. `src/full_reply.rs` rebuilds the reply object
(`src/lib.rs:1055-1135`) from kernel channels alone and emits it through the
vendored json module, the same one the engine's `write_json` uses, so key
order and number formatting are byte-equal by construction. The gate
(`tests/wave10_gate.rs`) compares whole texts; the fold (`tests/wave10_folds.rs`)
mutates one channel at a time and moves exactly the arrays decoded from it.

The kernel writes seven record streams, at the points the engine records the
same events, into the per-race u32 scratch (`retail_core.cl:459-477`, caps in
words: ACT 24,576, TGT 49,152, FAN 16,384, CONS 768, COMP 768, PHR 8,192, KAK
1,024), and the reply columns into the ST status block, read back only when
the race was prepared full. The host decodes them into `race_exec::ReplyRaw`
as flat word streams, one header word per channel.

| reply field | engine authority | kernel source |
| --- | --- | --- |
| `skillEvents` | `src/lib.rs:635-676` activate and deactivate hooks | ACT ten-word activation records plus KAK rush-edge pairs, merged and sorted by time, ordinary events first at equal time, then ordinal, kakari after every ordinal (`lib.rs:1030-1033`) |
| `skillDetailEvents`, `skillTargetEvents` | `race_solver.rs:2446-2451`, grouped by horse then push order | TGT records, per-effect length-prefixed selections, 0 encoding null |
| `additionalTargetEvents` | `race_solver.rs:1871-1875, 1933-1937` fan delivery | FAN deliveries with the `runtime_subset` scope |
| `competitionEvents` | `race_field.rs:181-201` spot reorder, duel pushes | COMP `[tick, kind, runner]`, kind 4 re-sorted per drain batch by the OONIGE-first, leader-first, roster-rank key, the startup marker word joining the tick 0 batch |
| `conservePowerEvents` | `src/lib.rs:637-644` asitame hook | CONS records written inside `asitame_release` (`retail_core.cl:3155`), gated by the same 1200 power rule as the engine's injected pseudo-skill (`race_solver_builder.rs:786-814`) |
| `positioningPhases` | `src/lib.rs:866-887` per-tick leader update, `stadium_score.rs:20-46` | PHR `[runner, phase, tick+1, lead]` per phase change of the unclamped `phase_boundaries` count (`retail_physics.rs:9-12`), published only where the lead word marks the runner as its style group's rank argmin that tick |
| `gutsOrder`, `wizOrder` | `src/lib.rs:811-827` snapshot orders with tolerance ties | ST snapshot columns, re-ranked by `parameter_orders` |
| `actualLastSpurtStartTime`, `actualLastSpurtStartDistance` | captured at the last-spurt transition | ST pair captured at the same transition (`race.cl:265-273`) |
| `playerLastSpurtTime` | `spfirst` frame or the finish fallback | ST first-spurt-frame column |
| `runningStyle`, `rushingStartCount`, `rushingPenaltyCount`, `hpEmptyOnRace`, `startDelayTime`, `lastSpurtStartDistance`, `finishTime`, `finishTimeRaw`, `finishOrder`, frames, version keys | unchanged | the compact exports, replay capture and static keys |

The full lane refuses, by name at `full_reply::build` and by gate before the
run: `singleMode`, `traceConservePower` and `experimentalLaterStaminaMechanics`
(wave 11 moved the `singleMode` and `experimentalLaterStaminaMechanics`
refusals onto both lanes and ported the compact `traceConservePower` ignore,
see "Wave-11 equivalence closures"). Grid races answer full too: wave 13b
rebuilt the staggered roster ranks from the grid starts, see "Wave 13b"
below.

## Wave-11 equivalence closures

`tests/equivalence_gate.rs` pairs every GPU refusal with what the CPU engine
does on the same request, so each refusal names the same construct the engine
sees, and an engine error text appears verbatim in the gate.

Effect kinds 33 and 34 have no `SkillType` variant: the engine's retail
builder errors (`race_solver_builder.rs:513`, run error `skill 669331 detail
0: unsupported effect type 33`), and the kernel refuses with a reason naming
that missing variant (`skills_compile.rs:1018-1021`), on both lanes.

Request shapes the engine itself errors on now get its text byte-equal from
`prepare()` and `facade::gate_reasons`: non-boolean
`experimentalRetailStartup`, `experimentalLaterStaminaMechanics` and
`traceConservePower` (`lib.rs:343-349`), and every invalid `experimentalGrid`
key. A 121-horse roster without a grid key is not an engine error: the engine
runs it, the kernel refuses on its own capacity (`roster size outside the
kernel's range`).

The four wave-10 named refusals, each classified against the engine:

- `singleMode` is an engine mode: the adjusted stats gain +400
  (`race_solver_builder.rs:304-320`, `build_adjusted_stats_for_mode`) and the
  reply gains `singleModeVersion` (`lib.rs:1127`), so the flag changes the
  answer. A non-boolean value is ignored at parse (`lib.rs:342`), and that
  ignore is ported: the full lane answers a `singleMode: 1.0` request
  byte-equal to the engine. The classic lane ports the `true` form: the +400
  folds inside `initialization::stats_reference` and the reply tags
  `singleModeVersion`, frames bit-exact (wave 12c). The retail lane refuses
  it ("singleMode is outside the GPU lane"); the compact lane had answered
  with the flag-off result, which the gate shows is not the engine's answer.
- `traceConservePower` is an engine mode that only adds reply diagnostics:
  `conservePowerDiagnosticVersion` and `conservePowerDiagnostics`
  (`lib.rs:1115-1123`), and the rest of the engine reply is unchanged, so the
  physics are identical. The engine also requires the retail runtime with it
  on (`api.rs:58-59`, `lib.rs:1504-1506`) and errors on a non-boolean value
  (`lib.rs:349`), both quoted in the gate: the classic lane refuses the flag
  with the engine's own error text. The compact lane accepts the flag as the
  ignore port (those keys never reach the compact output; meta and packet
  equal the flag-off output byte for byte), and the full lane refuses it by
  name.
- `experimentalLaterStaminaMechanics` is an engine mode: it enables the
  stamina keep, compete-before-spurt and secure-lead per-tick systems
  (`race_solver.rs:2767,2948,2998`, `update_competition_mechanics:2746`) and
  the reply echoes the flag (`lib.rs:1071`). Wave 11 refused the `true` form
  by name on both lanes; wave 13a ports it to the classic lane up to 2100 m
  and keeps the retail refusal, see "Wave 13a" in the wave history above.
- Grid mode is engine-supported end to end: a 21-horse grid request runs and
  the engine answers full, with `experimentalGridVersion` (`lib.rs:1137`).
  Wave 11 ran grid compact bit-exactly and refused only the full reply,
  naming the staggered snapshot ranks. Wave 13b ports both closes: the full
  reply rebuilds the grid-aware ranks, and the classic lane runs grid starts
  itself. See "Wave 13b" below.

`gpu/SKILLS_DESIGN.md` gained "Non-retail runtime physics: wave-11 port-scope
survey": a system-by-system diff of the engine's `step_all` classic path
against the retail path the kernel ports, with authorities and cost. The
verdict is a second solver, not a flag swap, so the non-retail refusal stays
and the port needs a dedicated wave.

Two engine rules surfaced while chasing byte equality and both were ports of
the engine, not the engine's behavior changing:

- Style leader scan. `StyleLeaders::update` (`stadium_score.rs:20-30`) picks
  the smallest-index runner with the strictly smallest cached rank across the
  whole style group. The kernel's leader word had only compared against
  lower-index peers, so the lowest-index runner of a style led vacuously and
  published phase changes the engine suppresses. The scan is now the full
  argmin (`race.cl` leader block).
- Duration compile in f32. The engine folds the ms base at compile time as
  `(base as f32) * 0.0001_f32` (`race_solver_builder.rs:416-418`). f32 of
  0.0001 rounds down, so 12000 ms becomes 1.1999999284744263, and 900531's
  pending duration chain lands two ulps below the f64 divide a `* 0.0001`
  compile produced (`skills_compile.rs:962-965` mirrors the f32 product).

Scratch deltas, pinned by `tests/layout.rs`: `SCR_U_WORDS` 500,021 to 602,999
(reply channels plus the record-state and scalar-slot shifts the phase mirror
needs, 1.91 to 2.29 MB per race slot), `ST_STRIDE` 1,088 to 2,168 (snapshot,
spurt pair and first-frame columns), `SK_STRIDE` 54,912 to 55,872.

The pending-slot model matches the engine's: one slot per (horse, skill id)
holding the larger remaining duration, so two live entries of one id collapse
their durations the same way the engine's scan does. Startup repeats sort
after ordinary events at equal time, carrying the `repeat` flag while the
engine marks them by ordinal (`lib.rs:1030-1033`), and kakari events carry no
ordinal, ordering after every ordinal event.

## Classic lane (wave 12a)

A request carrying no retail flag runs the engine's classic solver:
`step_forward` in f64, `dt = 1.0 / 15.0` (`src/lib.rs:66`,
`race_solver.rs:1176-1223`). The retail kernel does not implement it.
`classic_exec.rs` prepares the race, `kernels/classic_core.cl` and
`kernels/race_classic.cl` run it, `classic_reply.rs` assembles the full reply,
and `facade::run_batch` splits a batch by lane and launches each side once.

Construction runs host side in Rust, in the engine's own float types: the stats
chain with the mood and rank adjustments, the per-runner seeding chain, the
course sections, and the duel, spot and position-keep constants. The host builds
it with the vendored engine code (`initialization::stats_reference`,
`Rule30CARng`) and mirrors the pieces that stay private, so the kernel carries
only the tick loop and the draws the engine makes inside it: the kakari
`uniform(100)`, the extra lane-move `random()`, the position-keep wisdom
`random()`, the hill `random()`, and the HP stream's spurt accept loop. With the
construction on the host the classic kernel contains no `powf` or `log10` call
at all.

Every runner keeps its own `Rule30CARng`, seeded `(seed + i * 7919,
race_seed_hi(seed))` like the engine (`src/lib.rs:505-509`,
`race_solver_builder.rs:866-867`). The host replays the engine's draw order
before the loop: the builder int32 pair, the hp seed, the `gorosi` and
`pace_effect` int32s (`race_solver.rs:770-771`, which `initialization.cl`'s
`im_r30` does not draw), the gate roll, the random lot, the start delay, the
per-slope hill seeds (plus the extra draw when the first slope starts at 0),
the kakari style draw, the skip check, the kakari duration draws, and the 24
section modifiers. The retail lane never consumes
this stream, so the retail substream schedule is untouched.

Gates: `tests/wave12a_classic_gate.rs` compares every frame column with
`to_bits` and `finishTimeRaw` against `dep_cpu::run_cpu_race`, over solo
styles, the v77 field, and 8 and 18 horse app rosters on courses 10504 and
10508 with seeds 1 to 3. `tests/wave12a_full_gate.rs` compares the whole reply
text against `PreparedRace::run().write_json` on the same scenes.
`facade_gate::a_classic_request_answers_from_the_gpu_lane` checks the facade
path, meta and packet bytes and full reply, beside a retail race in the same
batch. Scratch sizes are pinned in `tests/layout.rs` under `CLS_*`.

## Not covered

The remaining fallbacks, each named at the race, at the reply lane or at
compile time, with what the engine does at the same point and why the GPU
lane does not answer. `tests/parity_audit.rs` enumerates the app-facing set
and proves every one of them is reachable, nothing else is (`gate_reasons_enumerates_exactly_the_documented_fallbacks`). The skill-side lists live in `SKILLS_DESIGN.md`.

| fallback text | engine side | GPU side |
| --- | --- | --- |
| `no GPU passes the exactness gate` | n/a | `facade::gate_reasons` with no device that passed `gpu::fp_gate_rejections` |
| `missing cl_khr_fp64, missing CL_FP_DENORM in single_fp_config, missing CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT, missing CL_FP_DENORM in double_fp_config` | n/a | `gpu::fp_gate_rejections`, joined |
| `experimentalGrid requires 21 through 120 horses, columns=20 and rowSpacing=3.0.` | engine parse error, `lib.rs:262-272` | quoted verbatim by `prepare` and the gate, retail and classic |
| `World geometry requires the experimental runtime and standard starting grid` | engine error, `api.rs:68-71` | quoted verbatim |
| `traceConservePower requires experimentalRetailRuntime` | engine error, `lib.rs:1504-1506` | quoted verbatim on the classic lane |
| `experimentalLaterStaminaMechanics must be a boolean.`, `traceConservePower must be a boolean`, `experimentalRetailStartup must be a boolean` | engine parse errors, `lib.rs:343-349` | quoted verbatim |
| `non-retail runtime` | the engine answers on its classic solver | the retail kernel is a retail-runtime port, it refuses rather than approximate, `race_exec::prepare` |
| `retail runtime belongs to the retail lane` | the retail config switches the solver (`lib.rs:1514-1523`) | the classic kernel is a classic-runtime port, `classic_exec::prepare` |
| `staminasyoubu, injected above 2100 m, stays outside the classic kernel` | the engine injects the synthetic skill, `race_solver_builder.rs:819` | classic lane, `classic_exec::later_stamina_refusal`, distance-gated |
| `experimentalLaterStaminaMechanics draws from a live solver stream the retail kernel resolves host-side` | the three systems draw live (`race_solver.rs:2751-2756`) | retail lane, `race_exec::prepare` |
| `traceConservePower is outside the full GPU lane` | the mode only adds reply keys (`lib.rs:1115-1123`) | `full_reply::build`, the compact lane answers |
| `roster size outside the kernel's range` | the engine answers any roster it is given | capacity refusal past `RACE_MAXR` (144 retail) or `CLS_MAXR` (120 classic) |
| `more slopes than the classic kernel carries` | the engine's course table carries any count | classic lane, past `CLS_MAX_SLOPES` (16). `gate_reasons` alone cannot name it: the check needs the course table, it runs at prepare |
| `the initialization reference flagged` | the engine answers | the host's `initialization::stats_reference` reports a fallback word for the roster, and the classic lane refuses instead of guessing adjusted stats |
| skill-compiler refusals | the engine's retail and classic builders reject or accept | `skills_compile`, each reason names the construct. The retail set is 17 families and the classic set 54, enumerated and pinned in `parity_audit` (see "Engine parity audit") |
| kernel `FB_*` fallbacks | n/a, kernel-internal | `race_exec::kernel_fallback`, capacity or geometry words the host missed |

Wave 13b removed three rows from this table: "grid mode needs the engine's
staggered roster ranks" (full replies now ported), and "experimentalGrid
belongs to the retail lane" plus "experimentalGrid is outside the classic
lane" (classic grid starts ported).

Wave 12b landed the classic skill runtime end to end. The compiler profile
(`skills_compile::plan_classic`) compiles the classic effect-value chain
(`modifier / 10000 * multiplier`) and duration (`base_duration / 10000`), and
refuses live-state condition leaves by name, gated in `tests/classic_folds.rs`.
In the kernel the skill pass runs the engine order (expire armed effects, the
priority walk, cooldown and window bookkeeping, the classic duel and spot
arms), activations resolve self and cross effects through the end-of-tick
batched field queue, and the reply lanes emit `skillEvents`,
`skillDetailEvents`, `skillTargetEvents`, `additionalTargetEvents[]` and
`conservePowerEvents` in engine push order. The asitame conserve-power release
(the release scan, its release clock and the `conservePowerEvents` channel) runs
on the GPU too. Full replies for skill and asitame rosters, real ids included,
are byte-equal to the engine (`tests/wave12b_gate.rs`,
`tests/wave12b_full_gate.rs`).

Wave 12c closed the sampled condition leaves on classic. The compiler resolves
each sampled leaf through the engine's own filter table (`corner_random`,
`straight_random`, `all_corner_random`, the phase, slope and corner variants,
18 names), folds one sampler job per live record carrying the reconciled
policy, the filter chain the windows draw from and the arm slots the windows
repoint, and `classic_exec` replays the engine's builder sampler host-side in
skilldata order (wire skill order, alternatives reversed inside one skill),
patching the merged window chains before the kernel starts, with the engine's
per-key reuse map so duplicate entries share one draw. Sampled leaves in a
precondition stay static filters with no sampler visit, as in the engine.
Classic condition filter errors and the one rejected policy pair are engine
build errors, so like `lib.rs:728-739` the kernel drops just that skill and
answers the rest, keeping the surviving skills' draw stream untouched; the
engine's own error texts travel unwrapped where a refusal reason is still
raised. Duplicate skill ids on one runner answer from the kernel (the shared
sampler cache and cooldown key mirror the kernel slot merge), the classic arm
cap of one sampled term is retail-only, and the live-state runtime reads
(`hp_per`, `order_rate`, `accumulatetime` and siblings) stay refused by name
until the kernel carries per-arm programs, except `blocked_side_continuetime`,
which wave 13a ported (next paragraph). Gated in `tests/classic_folds.rs`
and `tests/wave12c_classic_gate.rs`: every sampled name solo and eighteen on
one horse over seeds 1..3, frames bit-exact, full replies byte-equal, each
skill seen firing, duplicate reuse, and drop parity for eight engine-error
shapes.

Wave 13a also closed two classic skill runtime leaves. The compiler's
`blocked_side_continuetime` leaf (`FEAT_BSIDE`) is reader-ported: the classic
kernel already accumulates the blocked-side clock in its blocking update
(`DF_BSIDE`), `cc_sk_feat_value` reads it for the arm program, and a
prerequisite latch per record (`CLS_SKD_PRE`) opens when the compiled
prerequisite first holds, gating the fire like the engine's
stateful-prerequisite records. The live host 900351 (condition
`is_finalcorner==1&corner==0&order<=5`, prerequisite `is_finalcorner==1&
blocked_side_continuetime>=2`) now answers on the GPU lane. `duration_usage` 3 is likewise compiled and folded in
the kernel: `cc_sk_du3_coeff` mirrors the engine's HP-tier table
(`race_solver.rs:283-290`) reading the ledger value the engine freezes before
the effect loop (`race_solver.rs:2466`), and the caster-frozen duration rides
the delivery queue (`UDB_SKQDUR`) exactly like the engine's
`duration_override` (`race_solver.rs:2509`); `duration_usage` 1 is unchanged,
du 2 stays refused (it scales on the live leader gap) and du 4 stays refused
(it extends running instances on an overtake, which the classic lane never
enters).
The live host 120131 (du 3, condition `distance_rate>=50&order_rate<=40`)
answers: at the 2000 m course its engine event duration 15.0 is the base 5.0
times the 1.5 HP tier (hp in [2000, 2400) at activation) times the distance
factor 2.0, and rarity wire 3..=5 maps to Unique (`race_solver_builder.rs:
574,606`) so the evolution term is exactly 1.0. Both hosts ride the real-host
gate `wave12c_classic_gate::real_live_leaf_ids_answer_bit_exact`: fired at
least once on six seeds, frames and full replies bit-exact on all six.

Wave 13a ported `experimentalLaterStaminaMechanics` to the classic lane. The
flag turns on three per-tick systems (`update_competition_mechanics`
`race_solver.rs:2746-2758`): stamina keep (`:2767-2781`, HP-ratio gate, one
solver-stream draw once open), compete before spurt (`:2948-2992`, section
window `[section*10, section*15)`, cooldown, a far-gap or same-lane-nearby
gate, a `roll_competition_wisdom(15.0, dt)` draw, then a speed modifier into
the target-speed accumulator plus an HP cost), and secure lead (`:2998-3043`,
same shape with the `distance_to_behind` gate and a flat `0.20 * (dt/2)`
draw). The kernel mirrors them at the engine's position inside
`cc_step_field_conditions`, draws from the runner's solver stream
(`UR_SOLVER_HI/LO`), computes the wisdom rolls with the prepended `engine_log10`
and `gs_pow` ports, and ticks the four new clocks in the same `cc_sweep`
pass as the spot timer (engine registrations `race_solver.rs:979-982`). A
Recovery effect clears the keep latch on whoever applies it
(`race_solver.rs:2632`). The latches pack three bits into the one free runner
word (`UR_KEEP` slot 71); the six doubles (modifier, countdown, reuse wait,
per system) sit at `DF_CMPM` to `DF_SECCL`, 108 to 113, so the runner double
stride moved 112 to 120. The classic reply now echoes the real flag
(`lib.rs:1071`) instead of a fixed false.

The port is scoped by distance, not dropped: above 2100 m the engine also
injects the synthetic staminasyoubu skill (`race_solver_builder.rs:817-848`),
a construction hook the classic kernel does not model, so those requests
carry the named host fallback "staminasyoubu, injected above 2100 m, stays
outside the classic kernel" from both `classic_exec::prepare` and
`facade::gate_reasons` (`classic_exec::later_stamina_refusal`).

The retail lane keeps its refusal, now with the architectural reason: the
three systems draw live from the shared solver stream whose positions follow
the physics, while the retail runtime kernel carries no live stream consumer
at all (randomness enters as host-resolved reservations, `SI_RESRV`
`retail_core.cl:307`, and precomputed sub-streams), so the draws cannot be
placed there. The reason text is "experimentalLaterStaminaMechanics draws
from a live solver stream the retail kernel resolves host-side".

For activation evidence the kernel counts, per race, keep sets, compete
activations, secure activations and keep's gate stops (scratch words 24096 to
24099, mirrored into runner 0's status slots 15 to 18, status runner stride
17 to 20, surfaced as `ClassicGpu::later_stats`; never part of a reply).
`tests/wave13a_later_stamina.rs` runs seven seeds: the v77 field (1..3) and a
nine-horse high-stamina variant without skills (1, 2, 3 and 6) where the keep
HP gate closes early so the two section-window systems run. Frames are
bit-exact against the CPU engine and full replies byte-equal on all seven;
flag-off controls keep zero counters; the battery totals are keep 62, compete
4, secure 1 and gate stops 38,844, and every flagged scene moves the physics
(first divergence 34.7 to 84.3 s). `tests/wave12c_modes.rs` moved the
named-fallback row to the 2400 m course 10108, and `facade_gate` accepts the
flag at 2000 m and quotes the fallback at 2400 m.

## Wave 13b: grid full replies, classic grid starts, the 144-horse cap

Grid races answer full. The staggered roster ranks the competition reorder
sorts by (`race_field.rs:118-125`) are rebuilt from the grid starts, and the
reply carries the engine's `experimentalGridVersion: 1` (`lib.rs:1135-1138`).
`equivalence_gate::grid_full_replies_are_byte_equal_to_the_engine` pins the
whole reply byte-equal to the engine on 21-horse and 120-horse grids, seeds
1 to 3, and proves non-vacuity by counting the kind-4 competition events.

Grid starts run on the classic lane. The classic geometry is the engine's
pure-f64 grid (`lib.rs:274-279` without the retail f32 narrowing, which is
retail-only at `lib.rs:610`): column `rank % 20`, lane
`(column + (column >= 9 ? 0.6 : 0)) / 18`, position `-3 * (rank / 20)`, all
computed host-side in `classic_exec::prepare` and seeded through the new
`DF_POS` runner column (`F_POS0` slot 62). The grid tag rides the classic
compact meta and the full reply. `wave12c_modes::classic_grid_starts_answer_bit_exact`
gates frames bit-exact and replies byte-equal on 21-horse classic grids,
seeds 1 to 3, with a non-vacuity probe on the second row's lane and position,
and `facade_gate` runs the classic grid through the mixed-lane batch and full
lane. The three former grid refusals ("grid mode needs the engine's staggered
roster ranks", "experimentalGrid belongs to the retail lane",
"experimentalGrid is outside the classic lane") are gone.

The non-grid roster cap rose 120 to 144. The engine answers 121 through 145
(`lib.rs` has no roster cap; the grid shape check still stops at 120), so the
kernel carries what it measured: `RACE_MAXR` 144 with the init/IO strides
raised to match (`IIU` 448, `IIL` 168, `OOU` 4608, `OOL` 168), and
`equivalence_gate::rosters_beyond_the_grid_cap_run_and_only_the_kernel_cap_refuses`
pins that the engine and the GPU lane both accept 121, 128 and 144 while
roster 145 falls back by name from the run gate and `gate_reasons` alike.

Engine parity audit. `tests/parity_audit.rs` sweeps the whole app surface
instead of one scene per capability: the app's own comparison traffic (the
`testdata/app_preset_jobs_10504/10604` mob payloads with their uma skill sets
on the retail lane, the v77, mob8 and mob18 rosters on the classic lane, both
courses, seeds 1 to 3) answers with zero fallbacks, compact frames bit-exact
and full replies byte-equal in one mixed-lane batch; every one of the 751
merged-table skill ids compiles or refuses through both compilers, retail
with 17 documented refusal families and classic with 54, pinned family lists
in the test; and `gate_reasons_enumerates_exactly_the_documented_fallbacks`
proves the app-facing fallback set equals the "Not covered" table exactly,
nothing reachable is undocumented and nothing documented is unreachable.
`classic_prepare_only_fallbacks_stay_reachable` keeps the course-table-only
refusal ("more slopes than the classic kernel carries") provable through the
injected course text.

Sharp edge, pinned: non-grid fields wider than the grid's stagger bite a
pre-existing crowd physics gap. Dense non-grid rosters answer bit-exact
through 24 horses (their starting lanes stay under the lane rail) and start
diverging at 28 to 128 in the first crowded ticks, one `0.4/18` resolve-overlaps
push around the lane rail, reproduced value-for-value at HEAD 2f8f4b8 before
this wave's changes. `equivalence_gate::non_grid_crowds_above_twenty_four_diverge_from_the_engine`
pins the first divergent tick, runner and both values for 28, 40 and 128 so
a future port fix turns the pin into a failure signal. Grid mode is
unaffected at every size it admits because the staggered starts keep every
lane under the rail.


## Grid mode

`experimentalGrid` puts the field in 20-lane rows behind the line: rank 119
starts at -15 m. The kernel needs no grid flag. The host writes each runner's
start lane and start position into the `pd` runner block (slot `PI_RUN_LANE`,
the new `PI_RUN_POS`, stride 6 to 7) and the kernel seeds `SD_POS` from the
position slot instead of 0 (`race.cl:1017`). Retail races keep 0.0 there, so
the retail path is byte-for-byte what it was.

### Start authority (host, mirrors the engine)

`prepare` (`race_exec.rs`) validates the key the way `experimental_grid_enabled`
does (`src/lib.rs:262-272`): an object of exactly two keys, `columns: 20.0`,
`rowSpacing: 3.0`, roster 21 to 120, otherwise the race flags with the
engine's own message, "experimentalGrid requires 21 through 120 horses,
columns=20 and rowSpacing=3.0." Grid with `experimentalRetailGeometry` flags
with the engine's refusal, "World geometry requires the experimental runtime
and standard starting grid" (`api.rs:68-71`), and a valid grid request runs
without geometry, so positions advance arithmetically exactly like the
engine's geometry-free path (`race_solver_retail.rs:1322`).

Starts follow the engine: dense rank over the gateIndex-sorted field
(`src/lib.rs:603-606`, ties keep roster order), `column = rank % 20`,
`lane = (column + (column >= 9 ? 0.6 : 0)) / 18`, `pos = -3 * (rank / 20)`
(`src/lib.rs:274-279`), both values narrowed through f32 once on the host
(the engine does the same at `src/lib.rs:610`) before upload into `pd`. The
kernel then reads the exact value, `convert_float_rte` for the lane and
directly as f64 for the position. Grid replies add `experimentalGridVersion: 1`
to the compact meta, like the engine's reply (`src/lib.rs:1135-1138`).

### Capacity (shipped)

Measured from the compiled const chains and pinned by `tests/layout.rs` (the
kernel `#define` chains were checked line by line against them):

| constant | before | after | note |
| --- | --- | --- | --- |
| `RACE_MAXR` / `MAXR` | 24 | 120 | the engine's grid cap, `src/lib.rs:267` |
| `SD_SLOTS` | 107 | 299 | `SD_DUEL`/`SD_OVC` are n² pair maps at `slot + other * RACE_MAXR + r`, so `SD_OVC` moves to slot `40 + R` and `SD_SPOT_TIMER` to `40 + 2R`, then 17 per-runner scalars chain at `k * RACE_MAXR` |
| `LIST_STRIDE` | 48 | 240 | lane target and visible lists hold n-1 entries each, so 2n, and the plan's 128 would have run the visible list past the block |
| `CACHE_STRIDE` | none | 120 | the cached list block is indexed `4 * CACHE_STRIDE * r` and holds n entries, decoupled from `LIST_STRIDE` |
| `SK_RMAX` | 160 | 960 | 8 record slots per runner at the 120 cap. `sk_targets` caps live targets at the roster, so the raise adds no behavior at small rosters |
| `SK_STRIDE` | 34,112 | 54,912 | records at 26 words per slot drive the whole offset chain (`SKILLS_DESIGN.md` table updated) |
| `SCR_U_WORDS` | 58,588 | 450,300 | 1.80 MB u32 scratch per race slot, fixed at the cap for every roster |
| `SRL_WORDS` | 30,424 | 232,400 | 1.86 MB f64 scratch per race slot |
| `PI_RUN_STRIDE` | 6 | 7 | the `PI_RUN_POS` slot, and `PL_FRONT_LANE` moves 384 to 1,080 with `PD_CF_*` and `PD_RUN0` chaining behind it |
| `RI_STRIDE` | 352 | 1,592 | 13 debug rows per runner at the cap, `RI_DEBUG_TICK + 16` |
| `RIL_STRIDE` | 640 | 2,072 | pd block plus race scalars plus per-runner precompute |
| `ST_STRIDE` | 232 | 1,088 | finish bits and five exports per runner at the cap |
| `IIU` / `IIL` | 96 / 48 | 384 / 144 | the `IM_STATS` per-runner triples (16 + 3r) and `IM_INIT` base-wisdom rows (8 + r, params at 128 + 9) collide at 120 |
| `OOU` / `OOL` | 992 / 64 | 3,840 / 144 | runner rows (128 + 30r), `OS_CONT_I` 848 to 3,728, `OS_BANK_FOR_RACE` 3,744, `OL_CONT` 32 to 120 |

The plan's estimate "120, `SK_RMAX` 960: 367,772 / 198,264" undershot
because the per-runner stats and hp scratch columns ride the init strides
(384 / 144), not the old 96 / 48. Scratch is sized once, so a 24-horse race
now uploads 3.65 MB of scratch against 0.48 MB before (7.6x), frames aside.
Slope packing gained a real bound too: `pack_course_scalars` now stops at
`RACE_MAX_SLOPES` (16, the kernel's `RACE_MAX_SLOPES`) instead of whatever
fitted the buffer, keeping the pc block clear of the moved PL block.

Wave 13b raises the race roster cap from the grid's 120 to 144, measured
from the engine answering 121 to 145 non-grid rosters. The values above are
the grid-port history; the shipped values now are: `RACE_MAXR` 144 (grid
validation stays 21 to 120, the engine's own shape check), `RI_STRIDE`
1,904, `RIL_STRIDE` 2,864, `ST_STRIDE` 2,600, `IIU` / `IIL` 448 / 168,
`OOU` / `OOL` 4,608 / 168, `SCR_U_WORDS` 692,271 and `SRL_WORDS` 392,480,
together 5.91 MB of scratch per race slot, plus 3.14 MB of f64 in the double
block. `SK_RMAX` stays 960, the record cap the grid was sized for: a
non-grid roster past 120 that needs more than 960 skill records is refused
by name at compile. All pinned in `tests/layout.rs`.

### Launch budget

`run_prepared_mode` splits a batch by frame bytes: a `FRAME_BYTES_BUDGET` of
`512 << 20` divided by one race's frames (`RACE_MAX_TICKS * n * 6 * 8`),
minimum 1 race per launch. A race at the 144 cap writes about 42.5 MB of
frames, so 12 races launch per chunk at the cap; at 120 the same formula
gives 15. Every existing suite (up to 96 races at 9 runners, 2.65 MB frames)
stays a single launch. Buffers exist per chunk and
are released before the next one, which keeps the driver's resident set
bounded when scratch is 5.91 MB per race slot.

### Triage of the literal 24s

The plan counted 61 literal-24 sites across the kernel and its host mirrors.
Every stride one is now `RACE_MAXR`-derived. What remains is genuine and was
checked site by site:

| site | why it stays |
| --- | --- |
| `SEC 24` (`initialization.cl:14`), the `/ 24.0` section physics (`retail_core.cl:614,974,1340,2831`, `race.cl:127,128,158,797,1054`), the `s < 24` section draws (`race.cl:1037`), `SD_SECMOD + 24 * RACE_MAXR` (the 25th element, `race.cl:1041`), the 24 section randoms in `initialization.cl` | the engine's section physics divide the course into 24 sections, independent of roster |
| `SI_DS_ACTIVE SI_SLOT(24)` (`retail_core.cl:237`), `SF_L_FRONT_LANE SF_SLOT(24)` (`retail_core.cl:338`), `rd[24]`/`rd[25]` (`retail_core.cl:3445`), `FEAT_BASH_INFRONT 24` (`skills.cl:106`) | slot numbers and a feature code, not a count |
| `restricted_tick.cl` (3 sites) | the legacy restricted spike program, not part of the race program |
| `[u64; 24]` sections and `distance_f32 / 24.0` in `initialization.rs`, `FEAT_BASH_INFRONT` and record word 24 in `skills_compile.rs` | host mirrors of the same genuine constants |

### Gates

`grid_gate.rs` (gpu + initialization + geometry + race):

- `grid_races_replay_bit_exact`: 21, 24, 40 and 120 horses, courses 10504
  and 10604, seeds 1 and 2, synthetic rosters (the v77 base horse copied with
  `gateIndex` 0..n-1), zero fallback, frame-for-frame `to_bits` identity
  against `dep_cpu`, equal tick counts and `finishTimeRaw`, and the grid flag
  set on every race. 120 horses needs no ceiling: it passes as shipped.
- `grid_and_retail_starts_differ_at_21_horses`: the same 21-horse roster with
  and without the key, both bit-exact against the CPU engine, frame 0
  different, flags true and false.
- `invalid_grid_shapes_quote_the_engine_reason`: a one-key object, `columns`
  19, and roster 121 all flag on the GPU with the engine's message and the
  engine rejects the same request with it.
- `grid_with_geometry_quotes_the_engine_reason`: grid plus the geometry flag
  falls back naming the engine's refusal, GPU and CPU alike.

`facade_gate` expects the engine message for an invalid grid and accepts a
valid 21-horse grid in a mixed-lane batch (classic grid included), and
`layout` pins the strides. Wave 13b adds:
`equivalence_gate::grid_full_replies_are_byte_equal_to_the_engine` (full
replies byte-equal at 21 and 120 horses, seeds 1 to 3, kind-4 event count
as non-vacuity) and `wave12c_modes::classic_grid_starts_answer_bit_exact`
(classic 21-horse grids bit-exact frames and byte-equal replies, seeds 1 to
3, non-vacuity on the second row's lane and position).

### Cost at 120 runners

`cargo run --features race --release --bin grid_perf` (B580, course 10504,
grid roster 120, `run_repeated` path, one timed run per size on a warm
context, ms; re-measured on wave 13b's build, whose per-slot scratch is 5.91
MB at the 144 cap rather than the 5.27 MB of the 120-era tables):

| m | upload | launch | readback | decode | total | ms per seed |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 18.6 | 39,932.1 | 11.0 | 3.1 | 39,981.8 | 39,983.4 |
| 2 | 33.5 | 51,946.7 | 20.7 | 7.4 | 52,016.2 | 26,009.8 |
| 4 | 64.1 | 69,179.1 | 40.6 | 14.2 | 69,307.0 | 17,328.4 |
| 8 | 133.0 | 96,059.4 | 129.8 | 36.1 | 96,376.9 | 12,049.5 |
| 16 | 298.1 | 162,292.7 | 167.1 | 66.9 | 162,856.2 | 10,180.4 |
| 32 | 538.3 | 291,927.7 | 339.7 | 136.3 | 292,994.4 | 9,157.9 |
| 128 | 2,170.8 | 1,071,959.6 | 1,327.9 | 1,263.3 | 1,076,922.5 | 8,415.4 |

One race fills one core, so a single 120-horse grid race costs about 40 s.
The marginal cost per added seed falls from 12.0 s at m 2 to about 8.1 s at
m 16 and stays there through m 128 (the per-tick field work is n², and each
extra launch inside the 512 MiB budget adds its own sync). m 16, m 32 and
m 128 cross the budget (15 races per launch at n 120, so m 128 spans nine
launches): the binary checks that a chunked batch decodes bit-identical to
single-race launches, on the first slot of launch 1 and the last of launch 2.

### Sharp edges carried

- The lane the kernel reads is the host's f64-widened f32 (`pd` slot 5) and
  `SD_POS` seeds from `pd` slot 6. One narrowing on the host, then
  `convert_float_rte` on the lane, keeps the engine's `as f32 as f64` round
  trip exact.
- `SD_DUEL` and `SD_OVC` address as `slot + other * RACE_MAXR + r`: n²
  addressing, and the pair maps are the SD region's biggest consumer at the
  cap.
- Scratch is one fixed 5.91 MB per race slot whatever the roster (692,271 u32
  plus 392,480 f64 words, pinned in `tests/layout.rs`), and launch
  chunks re-allocate it each time. The OpenCL driver's buffer release lags
  process exit, which showed up once as a locked `.exe` right after a run.
- Grid frame-0 positions are negative (rows behind the line). The finish
  detector reads `SD_POS` deltas, so it is sign-safe, but an assertion of
  non-negative positions would be wrong.
- The `column >= 9` lane offset applies to the wrapped column, so rows behind
  the first repeat the first row's inner-half shift, and the grid lane is a
  different function of rank than `retail_starting_lane`.
- `kernel_fallback` matches `(FB_N, _)` on the always-1 `ST_FALLBACK` word,
  so a kernel `FB_GEOMETRY` fallback reports as "roster size the kernel
  rejects" (the code sits in the arg word). Pre-existing, worth untangling
  next time the mapping is touched.
