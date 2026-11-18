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

Four suites:

| Suite | Proves |
| --- | --- |
| `lib` (24 tests) | Vendored engine tests pass inside this crate |
| `reference_trace` | The in-crate runner reproduces `restricted_trace.json` from the Jax oracle byte-for-byte (CRLF normalized) |
| `layout` | The three `repr(C)` kernel structs are 104/36/244 bytes with the exact field offsets, and `kernel_params()` equals the engine's `Parameters` public methods for every scalar the kernel reads |
| `gpu_gate` | See below |

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
