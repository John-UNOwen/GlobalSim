# GlobalSim GPU spike

One OpenCL kernel of the restricted flat no-skills tick chain (eight ticks per
race), proven bit-identical to the Rust reference on an Intel Arc B580, plus
timing against a 20-thread Rust baseline. See `docs/GPU_HANDOFF.md` in the
engine repo for the plan this spike implements.

The crate is standalone (empty `[workspace]` table) and vendors the engine's
`src/retail_physics.rs`, `src/json.rs` and `src/replay_frames.rs` in place
through `#[path]` includes in `src/lib.rs`. The engine files are never edited
to make the GPU pass. License and provenance: `NOTICE`.

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
needed), which covers the reference tests.

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

## Not covered (step 6 blockers)

Engine calls the restricted chain never reaches, each needing its own port and
gate:

- RNG: the Rule30 `SeededRng` stream feeding spurt noise and decisions
- Retail simulation start: case construction (entry, `ReplayConfig` to
  `Simulation` fields, initial `Runner` states)
- Geometry queries: arc versus straight, ground and terrain lookups
- Lane and field queries: lane occupancy, `blocking_speed_cap`
- Pacing engine: section/pace sources and target-speed changes mid-race
- Spurt plan: `maximum_spurt_speed`, the only `powf` in the physics module
- Skill tables and additive modifiers (`&[f32]` slices into `acceleration` and
  `target_speed`)
- `hp_state_rate` and the hp style tables
- Replay frame encoding (`ReplayFrames` rows and the frame codec)
- The full `Simulation` loop: paired streams, race pacing, skills, finish order
