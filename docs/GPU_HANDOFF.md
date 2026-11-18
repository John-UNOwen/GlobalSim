# GlobalSim GPU backend: handoff

Written 2026-11-18. This replaces the earlier GlobalSim-Jax companion handoff (`docs/COMPANION_HANDOFF.md` in the `John-UNOwen/GlobalSim-Jax` clone), which planned a JAX port in a separate repo. Copy this file into the engine fork's branch (for example as `docs/GPU_HANDOFF.md`) as its first commit.

## Goal

Run races faster by splitting each batch of seeds between the CPU and a GPU, with **bit-identical** results: for one request and one seed, every replay frame (all six values for every runner, every tick) and every result field must equal the shipped engine's output byte for byte. Matching the finish time alone is not enough, because Compare depends on the whole race. Because results are identical, users without a suitable GPU get the same numbers from the CPU, only more slowly. Nobody is excluded.

## Decisions (2026-11-18)

- **No JAX.** It offers no benefit in speed or accuracy here. Its compiler (XLA) rewrites arithmetic, which works against bit-exactness. Running a batch as one array program makes every race compute both sides of every branch, and compile time grows with the number of skills. In practice it targets NVIDIA only (Intel support is preliminary and Linux-only). The existing JAX port (`Sevoii/GlobalSim-Jax`, forked to `John-UNOwen/GlobalSim-Jax`) is kept as reference material only (see "Reusing the JAX port").
- **OpenCL for the GPU backend.** One kernel source runs on NVIDIA, AMD and Intel, on Windows directly (no WSL). One race per GPU thread, written to read almost line for line against the Rust code. It gives the same control over floating-point behaviour as CUDA. A CUDA build of the same C kernel through a thin macro layer is an option only if measurements show a clear gain on NVIDIA.
- **The work lives in our engine fork**, `John-UNOwen/GlobalSim`, next to the engine it must match.
- **Layered plan:** native CPU engine in the sim server first, then the scheduler, then a small GPU spike. The GPU port grows only if the spike shows a real gain.

## Where the work lives

| What | Where |
|---|---|
| Repo | `John-UNOwen/GlobalSim` (remotes: `origin` = our fork, `upstream` = `daftuyda/GlobalSim`) |
| Branch | `gpu-backend`, created off `paired-draws` (the engine the app ships, `00d79fa`) |
| GPU code | a new crate in `gpu/`: OpenCL C kernels plus a Rust host (for example the `opencl3` crate), depending on the engine as `race_engine_rs = { path = ".." }` |
| App-side work (native CPU path, scheduler) | `John-UNOwen/Yet-Another-Umalator-Fork`, on its own branch |

- The engine is a single crate (`race_engine_rs`, library name `globalsim`), not a workspace. If cargo complains that `gpu/` sits inside another package, add an empty `[workspace]` table to `gpu/Cargo.toml` so it is its own root. Do not turn the engine's root `Cargo.toml` into a workspace: the app's `globalsim-wasm` crate depends on it by path.
- **Work in a separate checkout,** not in the app's `vendor/GlobalSim` submodule. That folder is checked out on `paired-draws`, and committing or switching branches there changes the app's submodule pointer. Either clone the fork again next to the app repo, or add a worktree:

```powershell
git -C "$env:USERPROFILE\Documents\GitHub\Yet-Another-Umalator-Fork\vendor\GlobalSim" worktree add "$env:USERPROFILE\Documents\GitHub\GlobalSim-gpu" -b gpu-backend paired-draws
```

```powershell
git -C "$env:USERPROFILE\Documents\GitHub\GlobalSim-gpu" push -u origin gpu-backend
```

- The app keeps its pin on `paired-draws` until the backend is ready, so the site is unaffected meanwhile.
- Never push to `upstream`.

## The engine we must match

| | Commit | What it is |
|---|---|---|
| Upstream v77 | `27a4d13` | the engine release |
| **Our pin** | `paired-draws` at `00d79fa` | v77 + 3 commits adding `pairedRandomStreams` |
| Upstream after v77 | `58da7bc` and later on `upstream/main` | v77 + AGPL notices, docs, Python (pyo3) and wasm bindings; engine behaviour unchanged |

- With `pairedRandomStreams` off, our pin replies byte-identical to v77 (re-checked 2026-11-16, `docs/globalsim-upgrade.md` in the app repo). With it on:
  - the in-race random stream is keyed by (seed, runner);
  - each skill's initialization windows and wisdom lottery are keyed by (seed, runner, skill);
  - the rushing lottery is keyed by (seed, runner).

  Compare sends the flag, so the GPU backend must implement this keying.
- Licence: the engine is AGPL-3.0-only (v77's `Cargo.toml`), and so is the JAX port. Anything copied from either keeps that licence and its notices.
- Size: about 21,300 lines of Rust in `src/`. The largest files are `race_solver.rs` (4,376), `activation_conditions.rs` (1,790), `lib.rs` (1,781), `retail_lane.rs` (1,555) and `race_solver_retail.rs` (1,338).

## Exactness rules for the OpenCL kernels

OpenCL's defaults are not exact. Every kernel and build must follow these rules:

- `#pragma OPENCL FP_CONTRACT OFF` at the top of every kernel file, so `a * b + c` is never merged into one fused operation.
- Never pass `-cl-mad-enable`, `-cl-fast-relaxed-math`, `-cl-unsafe-math-optimizations`, `-cl-finite-math-only` or `-cl-denorms-are-zero`.
- Pass `-cl-fp32-correctly-rounded-divide-sqrt`. Without it, OpenCL allows f32 division up to 2.5 ULP and f32 `sqrt` up to 3 ULP of error. Check that the device reports `CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT`; if it doesn't, it can't run the backend. f64 division and `sqrt` are always correctly rounded.
- Check that the device supports f32 subnormals (`CL_FP_DENORM` in its single-precision config) and f64 (`cl_khr_fp64`). Refuse to run on a device without them.
- **Never call OpenCL's built-in `pow`, `log`, `log10`, `exp`** and similar. Their accuracy varies by vendor. Port the routines the reference engine actually uses (next section) as plain arithmetic and bit operations.
- Use explicit conversions with round-to-nearest-even (`convert_float_rte` and friends) wherever Rust casts `f64 as f32`. Mirror the Rust code's order of operations and its f32/f64 widths exactly. Never reassociate a sum.
- Keep the random number generator in 32-bit integer arithmetic, as in `src/retail_random.rs` (a .NET `System.Random` port).

### Transcendental calls in the engine

`sqrt` is correctly rounded everywhere and is safe. Everything else must follow the algorithm the reference build uses:

| Call | Where | Type |
|---|---|---|
| `powf` | `src/retail_physics.rs:362` (spurt guts term) | f32 |
| `log10` | `src/retail_initialization.rs:526`, `:544` (wisdom) | f64 then cast |
| `log10`, `ln` | `src/retail_position_keep.rs:11`, `:25` (custom log10f reduction adapted from SunPro, see `licenses/pacing-log.txt`) | f64 then cast |
| `ln`, `exp`, `sqrt` | `src/activation_sample_policy.rs:192-209` (log-normal and exponential sampling for skill activation) | f64 |
| `powf`, `log10`, `powi` | `src/race_solver.rs` (13, 4 and 1 sites) | check each |
| `powi` | `src/hp_policy.rs` | check |

The wasm build and native Linux agree byte for byte (the app's CI runs `globalsim-wasm/test/phase0.mjs`), consistent with both using the same musl-derived maths. **The reference is the wasm build, or native Linux.** A native Windows (MSVC) build links Microsoft's C runtime maths and is not known to match until checked. The Rust `libm` crate is the musl port, so it is a convenient source for the exact algorithms and for reference values.

## Machines

- **This PC:** i7-12700K (12 cores, 20 threads), Intel Arc B580 and A770, no NVIDIA GPU. Python 3.12 to 3.14 through `py`. Rust 1.100 with the `wasm32-unknown-unknown` and `x86_64-pc-windows-msvc` targets, in `%USERPROFILE%\.cargo\bin` (not on every shell's PATH). The B580 runs f64 at a reported 1/16 of its f32 rate. The A770 reportedly has no native f64, so it is probably unusable.
- **A LAN PC with an NVIDIA GPU**, model not recorded yet. Consumer NVIDIA cards run f64 far slower than f32 (I believe 1/64 on RTX 30-series; measure it).
- The OpenCL runtime comes with the GPU drivers on Windows for both vendors. Building the `gpu/` crate may need an OpenCL import library (`OpenCL.lib`, from an OpenCL SDK such as Khronos's), unless the Rust bindings are set to load OpenCL at runtime. Check on the first build.

The engine leans on f64 throughout. **f64 throughput is the main threat to any GPU speedup**, which is why the spike comes before the port.

## Plan

Each step ends with its tests passing, so the work can stop at any step without leaving a broken state.

### 1. Native CPU engine in the sim server (app repo)

The local sim server (`umalator-global/v2/engines/globalsim/sim-server.ts`, started by `sim-server.bat`) already runs races outside the browser, one Node child process per session. **Its children still run the wasm engine** (`sim-server.ts:70-86`).

- First run `node globalsim-wasm/test/phase0.mjs --seeds 25` on Windows against a native build (`cargo build --release` in the engine, giving `target/release/race_engine_rs_v2`). If native Windows does not match wasm byte for byte, the native path is blocked until the engine uses the musl-derived maths on every platform (an engine change on `paired-draws`, through the app's CI).
- Then measure: the same batch on the wasm children against a native engine. Only build the next part if the gain is worth it.
- The engine surface the children need is small: `register_geometry(courseId, json)` and a `PreparedRace` with `run(seed)`, `run_compact(...)` and `free()` (`workerCore.ts:66-98`, `globalsim-wasm/src/lib.rs`). A native Node addon (for example with napi-rs) built from the same Rust wrapper code is a drop-in replacement in `loadWasm`.

### 2. Seed-range scheduler (app repo)

In the sim server: divide each results-only batch (skill chart, optimizer) into seed ranges sized to each backend's measured speed, run them, re-run every flagged race on the CPU, and merge in seed order. Compare's traces and single replays stay on the CPU. Build it with only CPU backends first, then add the GPU slot. Acceptance: the merged output equals an all-CPU run byte for byte.

### 3. GPU spike (engine fork, `gpu/`)

Small and throwaway-friendly. It answers one question: is a GPU clearly faster than this CPU for this kind of code?

1. Port the JAX port's **restricted flat no-skills loop** to an OpenCL kernel, from the Rust oracle `tools/restricted_oracle.rs` in the GlobalSim-Jax repo (not from the Python).
2. Exactness: the kernel's output must equal that repo's `tests/fixtures/restricted_trace.json` byte for byte.
3. Speed: run the same restricted loop as native Rust on all 20 threads, and as the OpenCL kernel at batch sizes 1, 32, 256, 1024, 8192 and 65536, on the B580 here and on the NVIDIA PC. Time build, transfer and steady state separately. Both sides run identical logic, so the comparison is fair.
4. Repeat with a kernel dominated by f64 arithmetic, to see the f64 cost on each card.

**Go/no-go:** decide on the numbers. As a starting bar, continue only if steady-state races per second on a GPU are clearly above the 20-thread CPU, say twice or more. Otherwise stop and keep the CPU path.

### 4. Whole-race comparison tool (engine fork)

The engine is a Rust library in the same repo, so the tool is a Rust test in `gpu/`: run the engine and the GPU kernel on the same request and seed, compare every replay frame, and report the first tick, runner and field that differ, with both bit patterns. Until the kernel runs whole races, "first tick that differs" is the progress measure. For the frame layout see `docs/BINARY-REPLAY.md` (`retainAllFrames: true` gives every frame).

### 5. Fallback flag

Every race carries a flag. Any operation that hits a domain the kernel can't guarantee sets it instead of failing. Flagged races are re-run on the CPU by the scheduler. Count and report them. Acceptance: zero unflagged mismatches over a large seed sweep.

### 6. Port the race, in this order

1. RNG with `pairedRandomStreams` keying; the exact maths routines (test every input where the domain is small, such as guts and wisdom).
2. Initialization, geometry, physics.
3. Field: ranks, ties, blocking, lane movement, lane-target selection (PassingSearch and its RNG draws).
4. Pacing, pacemaker and position keep; rushing, competition, spurt; post-goal.
5. Done when complete races without skills match on several courses and field sizes.

### 7. Skills

Before porting all skills, test the design: compile skill conditions into a fixed table of numeric instructions interpreted by the kernel, so the kernel never changes with the skill set. Prototype with a handful of skills and measure cost per tick. Then port the rest.

## Reusing the JAX port

`John-UNOwen/GlobalSim-Jax` (clone at `%USERPROFILE%\Documents\GitHub\GlobalSim-Jax`) stays as reference. Useful parts:

- **Rust oracles** (`tools/*.rs`) and their **fixtures** (`tests/fixtures/*.json`, about 9 MB): exact expected values for the RNG, physics, geometry, lane movement, field ticks and target updates. Build instructions are in `tools/*_ORACLE.md`. Two build scripts hard-code the author's path (`tools/build_target_collect_oracle.py:9`, `tools/build_target_update_oracle.py:6`); point them at our engine checkout.
- **Do not run the committed `tools/*.exe` files.** They are prebuilt binaries from another machine. Rebuild from `.rs`.
- **Lessons on exactness** from that port, which apply to any backend:
  - XLA rewrote `x / y` as `x * (1 / y)` in the HP drain (blocked there with optimization barriers). OpenCL's equivalent risk is the relaxed-math options, which the rules above forbid.
  - XLA on CPU treated f32 subnormals as zero in comparisons, so the port refused that input domain (`globalsim_jax/lane_queries.py:21`). A race can't refuse an input; handle it exactly or raise the fallback flag.
  - The port used XLA's own `pow` for the spurt guts term (`globalsim_jax/physics.py:59`), which is not the engine's algorithm.
- **Their milestone order** (`docs/PORT_PLAN.md`) and their habit of proving each piece against an oracle before moving on.

Anything copied into the engine fork keeps AGPL-3.0-only and credits the original author in a `NOTICE` inside `gpu/`.

## Rules

- The Rust engine is the authority. Never change it to make the GPU code pass. Engine changes go on `paired-draws` through the app's CI.
- Never write GPU output into a fixture as an expected value.
- No claims of speedups or whole-race parity without the measurements behind them.
- Never push to `upstream`.

## Open questions

- Which NVIDIA GPU the LAN PC has, and its f64 rate.
- Whether native Windows Rust matches wasm byte for byte (step 1).
- Whether a native engine in the sim server is faster than wasm by enough to justify the addon.
- Whether the B580's OpenCL driver reports correctly rounded f32 divide/sqrt, f32 subnormals and `cl_khr_fp64`.
- What `GlobalSim-Jax` should become: archived, or left as is with a pointer to this branch.

## Where things are

| What | Where |
|---|---|
| Engine source (our pin) | `src/` in the engine fork; also `vendor/GlobalSim/src/` in the app repo |
| Binary replay format | `docs/BINARY-REPLAY.md` in the engine fork |
| Paired streams | `docs/globalsim-upgrade.md` in the app repo |
| wasm-vs-native parity check | `globalsim-wasm/test/phase0.mjs` in the app repo |
| Engine wrapper the sim server uses | `globalsim-wasm/src/lib.rs` in the app repo |
| Sim server | `umalator-global/v2/engines/globalsim/sim-server.ts`, `workerCore.ts` in the app repo |
| JAX port (reference) | `John-UNOwen/GlobalSim-Jax`: `tools/`, `tests/fixtures/`, `docs/PORT_PLAN.md` |

