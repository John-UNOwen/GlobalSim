//! OpenCL host for the restricted spike: exactness device gate, program build,
//! one __constant parameter buffer per job, timed upload/run/read phases.

use crate::reference::{KernelCase, KernelParams, KernelRow, ROWS};
use ocl::{Buffer, Device, DeviceType, Platform, Program, Queue, flags};
use std::cell::Cell;
use std::time::Instant;

pub const KERNEL_SOURCE: &str = include_str!("kernels/restricted_tick.cl");
// The two RNG consumers share one ported helper source, prepended here because
// OpenCL source strings resolve #include implementation-definedly.
pub const RNG_KERNEL_SOURCE: &str =
    concat!(include_str!("kernels/rng_core.cl"), "\n", include_str!("kernels/rng.cl"));
pub const INITIALIZATION_KERNEL_SOURCE: &str = concat!(
    include_str!("kernels/rng_core.cl"),
    "\n",
    include_str!("kernels/skills.cl"),
    "\n",
    include_str!("kernels/initialization.cl")
);
pub const GEOMETRY_KERNEL_SOURCE: &str = include_str!("kernels/geometry.cl");
// The race kernel reuses the RNG, initialization and geometry ports, so its
// source is those files concatenated ahead of race.cl. The powf port comes
// right after the rng core so every consumer below sees the helpers.
pub const RACE_KERNEL_SOURCE: &str = concat!(
    include_str!("kernels/rng_core.cl"),
    "\n",
    include_str!("kernels/powf_core.cl"),
    "\n",
    include_str!("kernels/skills.cl"),
    "\n",
    include_str!("kernels/initialization.cl"),
    "\n",
    include_str!("kernels/geometry.cl"),
    "\n",
    include_str!("kernels/retail_core.cl"),
    "\n",
    include_str!("kernels/race.cl")
);
pub const BUILD_OPTIONS: &str = "-cl-fp32-correctly-rounded-divide-sqrt";

/// The runner capacity every runner-indexed kernel region sizes to, on both
/// lanes. The game never fields more than 18, so 18 is the hard limit and a
/// race slot stays small enough to launch thousands at once. The engine
/// answers a larger roster on the CPU: past this cap the lane refuses by
/// size (`race_exec::RACE_MAXR`, `classic_exec::CLS_MAXR`).
pub const ROSTER_CAP: usize = 18;

/// `BUILD_OPTIONS` plus the capacity defines, so the kernels' `RACE_MAXR` and
/// `CLS_MAXR` always equal the host's strides.
pub fn build_options() -> String {
    format!("{BUILD_OPTIONS} -D RACE_MAXR={ROSTER_CAP} -D CLS_MAXR={ROSTER_CAP}")
}

/// The race program variant that compiles the diagnostics journal in (the
/// debug-scratch path). The default race program runs without it.
pub fn diag_build_options() -> String {
    format!("{} -D SK_DIAG", build_options())
}

/// The launch shape a race program is specialized for: the regions a launch
/// can touch. `Full` carries the reply channels, `Diag` the journal window.
/// Compact row masks change frame memory, not the program, so they need no
/// variant. A shape with neither flag is the sampling (and compact) hot path.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RaceShape {
    /// No reply channels, no journal: the sampling and compact hot path.
    Sampling,
    /// Reply channels on: a batch that can carry full requests.
    Full,
    /// Journal on, channels off: the debug-scratch path.
    Diag,
    /// Journal and channels on.
    FullDiag,
}

impl RaceShape {
    /// The shape of a launch whose chunk carries full requests, debug-scratch
    /// races, both or neither.
    pub const fn from_flags(full: bool, diag: bool) -> RaceShape {
        match (full, diag) {
            (false, false) => RaceShape::Sampling,
            (true, false) => RaceShape::Full,
            (false, true) => RaceShape::Diag,
            (true, true) => RaceShape::FullDiag,
        }
    }

    /// True when the program keeps the full-reply channels.
    pub const fn reply(self) -> bool {
        matches!(self, RaceShape::Full | RaceShape::FullDiag)
    }

    /// True when the program keeps the diagnostics journal.
    pub const fn journal(self) -> bool {
        matches!(self, RaceShape::Diag | RaceShape::FullDiag)
    }
}

/// The race program at one record bucket. The scratch regions size to
/// `rcap`, and the top bucket keeps the default build options so the
/// hot-path program is unchanged.
pub fn bucket_build_options(rcap: usize) -> String {
    let base = build_options();
    if rcap >= crate::skills_compile::SK_RMAX {
        base
    } else {
        format!("{base} -D SK_RCAP={rcap}")
    }
}

/// The diagnostics program at one record bucket.
pub fn diag_bucket_build_options(rcap: usize) -> String {
    let base = diag_build_options();
    if rcap >= crate::skills_compile::SK_RMAX {
        base
    } else {
        format!("{base} -D SK_RCAP={rcap}")
    }
}

/// The race program for one full define set: the effect-cap bucket, the
/// tag-cap bucket, the record bucket the scratch sizes to, and the launch
/// shape's region defines. The first tag bucket keeps its warm program.
pub fn race_bucket_build_options(
    emax: usize, tmax: usize, rcap: usize, shape: RaceShape,
) -> String {
    let base = if shape.journal() {
        diag_bucket_build_options(rcap)
    } else {
        bucket_build_options(rcap)
    };
    let base = if shape.reply() {
        base
    } else {
        format!("{base} -D GSIM_NO_REPLY")
    };
    let base = format!("{base} -D SK_EMAX={emax}");
    if tmax == crate::skills_compile::SK_TMAX {
        base
    } else {
        format!("{base} -D SK_TMAX={tmax}")
    }
}

/// One line of cache or build chatter, printed only when
/// `GSIM_GPU_PROGRAM_LOG` is set.
fn program_log(line: &str) {
    if std::env::var_os("GSIM_GPU_PROGRAM_LOG").is_some() {
        eprintln!("{line}");
    }
}

/// FNV-1a over the bytes, for cache file keys and the request cache keys in
/// `race_exec` (entries keep the full text and verify it, so a collision
/// costs a miss, never a wrong entry). Eight bytes a round, a wider mix.
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut chunks = bytes.chunks_exact(8);
    for c in &mut chunks {
        let word = u64::from_le_bytes(c.try_into().unwrap());
        h = (h ^ word).wrapping_mul(PRIME);
    }
    for b in chunks.remainder() {
        h = (h ^ *b as u64).wrapping_mul(PRIME);
    }
    h
}

/// The cache file for one option set, keyed by kernel source and options.
pub fn program_cache_path(dir: &std::path::Path, options: &str) -> std::path::PathBuf {
    dir.join(format!(
        "race-{:016x}-{:016x}.bin",
        fnv1a64(RACE_KERNEL_SOURCE.as_bytes()),
        fnv1a64(options.as_bytes()),
    ))
}

/// Writes one build's source and options into `GSIM_DUMP_KERNEL`, the input
/// for the offline `ocloc` compile: the driver reports no register or spill
/// figures, so the offline compiler is where the ISA comes from.
fn maybe_dump_kernel(stem: &str, options: &str, source: &str) {
    let Ok(dir) = std::env::var("GSIM_DUMP_KERNEL") else { return };
    let dir = dir.trim();
    if dir.is_empty() {
        return;
    }
    let key = fnv1a64(options.as_bytes());
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(format!("{dir}/{stem}-{key:016x}.cl"), source);
    let _ = std::fs::write(format!("{dir}/{stem}-{key:016x}.opts"), options);
}

const PROGRAM_CACHE_MAGIC: &[u8; 8] = b"GSIMPC02";

/// A second pass over the kernel source with a different starting basis and a
/// byte-wise fold, stored in the file header. The file name already keys one
/// 64-bit hash, this one means a colliding name alone cannot make a foreign
/// image load.
fn source_check_digest(bytes: &[u8]) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h: u64 = 0x4222_2325_cbf2_9ce4;
    for b in bytes {
        h = (h ^ *b as u64).wrapping_mul(PRIME);
    }
    h
}

/// The prefix every cache file for this device and option set carries: magic,
/// the three identity texts, the kernel source length and its check digest.
fn program_cache_prologue(device: &DeviceReport, options: &str) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(PROGRAM_CACHE_MAGIC);
    for text in [&device.name, &device.driver_version, options] {
        out.extend_from_slice(&(text.len() as u32).to_le_bytes());
        out.extend_from_slice(text.as_bytes());
    }
    out.extend_from_slice(&(RACE_KERNEL_SOURCE.len() as u64).to_le_bytes());
    out.extend_from_slice(&source_check_digest(RACE_KERNEL_SOURCE.as_bytes()).to_le_bytes());
    out
}

/// Write the program's device binaries plus everything that had to match to
/// compile them. The file lands via a temp rename, so a killed process
/// cannot leave a half image.
fn write_program_binary(
    path: &std::path::Path, device: &DeviceReport, options: &str, program: &Program,
) -> Result<(), String> {
    use ocl::core::{ProgramInfo, ProgramInfoResult};
    let ProgramInfoResult::Binaries(bins) =
        program.info(ProgramInfo::Binaries).map_err(|e| e.to_string())?
    else {
        return Err("driver gave no program binaries".into());
    };
    let mut out = program_cache_prologue(device, options);
    out.extend_from_slice(&(bins.len() as u32).to_le_bytes());
    for bin in &bins {
        out.extend_from_slice(&(bin.len() as u64).to_le_bytes());
        out.extend_from_slice(bin);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let temp = path.with_extension(format!("tmp.{}", std::process::id()));
    std::fs::write(&temp, &out).map_err(|e| e.to_string())?;
    std::fs::rename(&temp, path).map_err(|e| e.to_string())
}

/// One little-endian field of a cache file, advancing `p`. Truncation comes
/// back as an error, so the caller rebuilds from source instead of panicking.
fn cache_u32(bytes: &[u8], p: &mut usize) -> Result<u32, String> {
    if bytes.len() < *p + 4 {
        return Err("program cache file truncated".into());
    }
    let v = u32::from_le_bytes(bytes[*p..*p + 4].try_into().unwrap());
    *p += 4;
    Ok(v)
}

fn cache_u64(bytes: &[u8], p: &mut usize) -> Result<u64, String> {
    if bytes.len() < *p + 8 {
        return Err("program cache file truncated".into());
    }
    let v = u64::from_le_bytes(bytes[*p..*p + 8].try_into().unwrap());
    *p += 8;
    Ok(v)
}

fn cache_text(bytes: &[u8], p: &mut usize) -> Result<String, String> {
    let n = cache_u32(bytes, p)? as usize;
    if bytes.len() < *p + n {
        return Err("program cache file truncated".into());
    }
    let s = String::from_utf8_lossy(&bytes[*p..*p + n]).into_owned();
    *p += n;
    Ok(s)
}

/// Read a cached image back, refusing anything that does not carry this
/// device, driver, kernel source and option set.
fn parse_program_binary(
    bytes: &[u8], device: &DeviceReport, options: &str,
) -> Result<Vec<Vec<u8>>, String> {
    if bytes.len() < 8 || &bytes[..8] != PROGRAM_CACHE_MAGIC {
        return Err("not a program cache file".into());
    }
    let mut p = 8usize;
    if cache_text(bytes, &mut p)? != device.name
        || cache_text(bytes, &mut p)? != device.driver_version
        || cache_text(bytes, &mut p)? != options
    {
        return Err("device, driver or options differ from the saved image".into());
    }
    let src_len = cache_u64(bytes, &mut p)? as usize;
    if src_len != RACE_KERNEL_SOURCE.len() {
        return Err("kernel source length differs from the saved image".into());
    }
    if cache_u64(bytes, &mut p)? != source_check_digest(RACE_KERNEL_SOURCE.as_bytes()) {
        return Err("kernel source digest differs from the saved image".into());
    }
    let n_bins = cache_u32(bytes, &mut p)? as usize;
    let mut bins = Vec::with_capacity(n_bins.min(8));
    for _ in 0..n_bins {
        let n = cache_u64(bytes, &mut p)? as usize;
        if bytes.len() < p + n {
            return Err("program cache file truncated".into());
        }
        bins.push(bytes[p..p + n].to_vec());
        p += n;
    }
    Ok(bins)
}
// The classic lane carries the bit-exact powf table for the mid-race stat
// refresh (race_solver.rs:1330-1340 recomputes the guts and power derived
// terms when a delivered stat effect lands). Construction stays host side.
pub const CLASSIC_KERNEL_SOURCE: &str = concat!(
    include_str!("kernels/rng_core.cl"),
    "\n",
    include_str!("kernels/powf_core.cl"),
    "\n",
    include_str!("kernels/skills.cl"),
    "\n",
    include_str!("kernels/initialization.cl"),
    "\n",
    include_str!("kernels/classic_core.cl"),
    "\n",
    include_str!("kernels/race_classic.cl")
);

// cl.h values, as raw ids through Device::info_raw.
const CL_DEVICE_NAME: u32 = 0x102B;
const CL_DEVICE_EXTENSIONS: u32 = 0x1030;
const CL_DEVICE_MAX_COMPUTE_UNITS: u32 = 0x1002;
const CL_DEVICE_MAX_WORK_GROUP_SIZE: u32 = 0x1004;
const CL_DEVICE_SINGLE_FP_CONFIG: u32 = 0x101B;
const CL_DEVICE_DOUBLE_FP_CONFIG: u32 = 0x1032;
const CL_DEVICE_DRIVER_VERSION: u32 = 0x102D;
const CL_FP_DENORM: u64 = 0x1;
const CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT: u64 = 0x80;

fn read_u64(device: &Device, param: u32) -> Result<u64, String> {
    // cl_uint-valued params (compute units) come 4 bytes wide, cl_ulong-valued
    // ones (fp config flags) come 8.
    let raw = device.info_raw(param).map_err(|e| e.to_string())?;
    match raw.len() {
        8 => Ok(u64::from_le_bytes(raw[..8].try_into().unwrap())),
        4 => Ok(u32::from_le_bytes(raw[..4].try_into().unwrap()) as u64),
        _ => Err(format!("param {param:#x} neither 32 nor 64 bits")),
    }
}

fn read_string(device: &Device, param: u32) -> Result<String, String> {
    let raw = device.info_raw(param).map_err(|e| e.to_string())?;
    Ok(String::from_utf8_lossy(&raw).into_owned())
}

#[derive(Clone)]
pub struct DeviceReport {
    pub platform: String,
    pub name: String,
    pub driver_version: String,
    pub compute_units: u64,
    pub max_work_group_size: u64,
    pub single_fp_config: u64,
    pub double_fp_config: u64,
}

/// The exactness gate as one pure function, so a test can feed it device
/// metadata without opening a device. Empty means the device is accepted.
pub fn fp_gate_rejections(
    extensions: &str,
    single_fp_config: u64,
    double_fp_config: u64,
) -> Vec<String> {
    [
        (!extensions.split_whitespace().any(|word| word == "cl_khr_fp64"))
            .then(|| "missing cl_khr_fp64".to_string()),
        (single_fp_config & CL_FP_DENORM == 0)
            .then(|| "missing CL_FP_DENORM in single_fp_config".to_string()),
        (single_fp_config & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT == 0)
            .then(|| "missing CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT".to_string()),
        (double_fp_config & CL_FP_DENORM == 0)
            .then(|| "missing CL_FP_DENORM in double_fp_config".to_string()),
    ]
    .into_iter()
    .flatten()
    .collect()
}

struct Candidate(DeviceReport, Device, Platform);

fn candidates() -> Result<(Vec<Candidate>, Vec<(String, String)>), String> {
    let mut accepted = Vec::new();
    let mut rejected = Vec::new();
    for platform in Platform::list() {
        let platform_name = platform.name().unwrap_or_else(|_| "unknown".into());
        let devices = Device::list(&platform, Some(DeviceType::GPU)).map_err(|e| e.to_string())?;
        for device in devices {
            let name = read_string(&device, CL_DEVICE_NAME).unwrap_or_else(|_| "unknown".into());
            let report = DeviceReport {
                platform: platform_name.clone(),
                name: name.clone(),
                driver_version: read_string(&device, CL_DEVICE_DRIVER_VERSION).unwrap_or_else(|_| "unknown".into()),
                compute_units: read_u64(&device, CL_DEVICE_MAX_COMPUTE_UNITS).unwrap_or(0),
                max_work_group_size: read_u64(&device, CL_DEVICE_MAX_WORK_GROUP_SIZE).unwrap_or(0),
                single_fp_config: read_u64(&device, CL_DEVICE_SINGLE_FP_CONFIG).unwrap_or(0),
                double_fp_config: read_u64(&device, CL_DEVICE_DOUBLE_FP_CONFIG).unwrap_or(0),
            };
            // The handoff's exactness gate: GPU class, cl_khr_fp64 with f64
            // subnormals, f32 subnormals, and correctly-rounded f32 divide/sqrt.
            let reasons = fp_gate_rejections(
                &read_string(&device, CL_DEVICE_EXTENSIONS).unwrap_or_default(),
                report.single_fp_config,
                report.double_fp_config,
            );
            if reasons.is_empty() {
                accepted.push(Candidate(report, device, platform));
            } else {
                rejected.push((name, reasons.join(", ")));
            }
        }
    }
    Ok((accepted, rejected))
}

pub struct Spike {
    queue: Queue,
    context: ocl::Context,
    core_device: Device,
    program: Program,
    pub device: DeviceReport,
    pub build_ms: f64,
    build_failures: Cell<u32>,
}

impl Spike {
    /// Picks the first GPU that passes the exactness gate and builds the
    /// program once with the required options.
    pub fn open() -> Result<Spike, String> {
        let (accepted, rejected) = candidates()?;
        let Some(Candidate(report, device, platform)) = accepted.into_iter().next() else {
            let mut message = String::from("no GPU passes the exactness gate:");
            for (name, reasons) in rejected {
                message.push_str(&format!("\n  {name}: {reasons}"));
            }
            return Err(message);
        };
        let core_device = device.clone();
        let context = ocl::Context::builder()
            .platform(platform)
            .devices(device.clone())
            .build()
            .map_err(|e| e.to_string())?;
        let queue = Queue::new(&context, device.clone(), None).map_err(|e| e.to_string())?;
        let started = Instant::now();
        let program = Program::builder()
            .devices(device)
            .cmplr_opt(BUILD_OPTIONS)
            .source(KERNEL_SOURCE)
            .build(&context)
            .map_err(|e| e.to_string())?;
        let build_ms = started.elapsed().as_secs_f64() * 1000.;
        Ok(Spike {
            queue,
            context,
            core_device,
            program,
            device: report,
            build_ms,
            build_failures: Cell::new(0),
        })
    }

    /// Builds an extra kernel program (for example rng.cl) on the already
    /// gated device, with the same exactness build options.
    pub fn build_program(&self, source: &str) -> Result<Program, String> {
        let options = build_options();
        maybe_dump_kernel("program", &options, source);
        Program::builder()
            .cmplr_opt(options)
            .source(source)
            .build(&self.context)
            .map_err(|e| e.to_string())
    }

    /// The race program for one (effect-cap bucket, tag-cap bucket, record
    /// bucket, launch shape) define set, so the kernel's table offsets,
    /// scratch regions and shape regions match what the compiler packed and
    /// sized for them.
    pub fn build_race_program(
        &self, emax: usize, tmax: usize, rcap: usize, shape: RaceShape,
    ) -> Result<Program, String> {
        self.build_race_program_with(race_bucket_build_options(emax, tmax, rcap, shape))
    }

    fn build_race_program_with(&self, options: String) -> Result<Program, String> {
        maybe_dump_kernel("race", &options, RACE_KERNEL_SOURCE);
        Program::builder()
            .cmplr_opt(options)
            .source(RACE_KERNEL_SOURCE)
            .build(&self.context)
            .map_err(|e| e.to_string())
    }

    /// The race program for one define set, served from the program binary
    /// disk cache when `GSIM_GPU_PROGRAM_CACHE` names a directory. A stored
    /// image reloads only when the device name, driver version, kernel source
    /// and build options match what it was compiled from. A missing, stale or
    /// rejected file falls back to the source build and rewrites the cache.
    pub fn build_race_program_cached(
        &self, emax: usize, tmax: usize, rcap: usize, shape: RaceShape,
    ) -> Result<Program, String> {
        let options = race_bucket_build_options(emax, tmax, rcap, shape);
        let cache_dir = match std::env::var_os("GSIM_GPU_PROGRAM_CACHE") {
            Some(dir) if !dir.is_empty() => std::path::PathBuf::from(dir),
            _ => return self.build_race_program_with(options),
        };
        let path = program_cache_path(&cache_dir, &options);
        match self.load_program_binary(&path, &options) {
            Ok(program) => {
                program_log(&format!("program cache hit: {}", path.display()));
                return Ok(program);
            }
            Err(e) => program_log(&format!("program cache miss, {e}: {}", path.display())),
        }
        let program = self.build_race_program_with(options.clone())?;
        if let Err(e) = write_program_binary(&path, &self.device, &options, &program) {
            program_log(&format!("program cache write failed, {e}"));
        }
        Ok(program)
    }

    fn load_program_binary(&self, path: &std::path::Path, options: &str) -> Result<Program, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let mut bins = parse_program_binary(&bytes, &self.device, options)?;
        let refs: Vec<&[u8]> = bins.iter_mut().map(|b| b.as_slice()).collect();
        let device = self.core_device.clone();
        let no_opts = std::ffi::CString::new("").map_err(|e| e.to_string())?;
        Program::with_binary(&self.context, &[device], &refs, &no_opts).map_err(|e| e.to_string())
    }

    pub fn kernel_build_failures(&self) -> u32 {
        self.build_failures.get()
    }

    pub fn queue(&self) -> Queue {
        self.queue.clone()
    }

    /// Buffers for `count` cases plus the shared parameter struct. Parameter
    /// upload is not timed: it happens once per job, outside the measured loop.
    pub fn prepare(&self, params: &KernelParams, count: usize) -> Result<Job, String> {
        let params_buf = self.buffer(&struct_bytes(params), flags::MEM_READ_ONLY)?;
        let cases_buf = self.empty_buffer(count * std::mem::size_of::<KernelCase>(), flags::MEM_READ_ONLY)?;
        let rows_buf = self.empty_buffer(count * ROWS * std::mem::size_of::<KernelRow>(), flags::MEM_WRITE_ONLY)?;
        let kernel = ocl::Kernel::builder()
            .program(&self.program)
            .name("restricted_tick")
            .queue(self.queue.clone())
            .global_work_size(count)
            .arg(&params_buf)
            .arg(&cases_buf)
            .arg(&rows_buf)
            .build()
            .map_err(|e| {
                self.build_failures.set(self.build_failures.get() + 1);
                e.to_string()
            })?;
        Ok(Job {
            queue: self.queue.clone(),
            _params_buf: params_buf,
            cases_buf,
            rows_buf,
            kernel,
        })
    }

    /// One warmup launch then `repeats` timed launches of an fma probe kernel:
    /// `count` work items, four independent chains of `iterations` fma calls.
    /// Returns the average milliseconds per timed launch.
    pub fn fma_probe(&self, f64_variant: bool, count: usize, iterations: i32, repeats: usize) -> Result<f64, String> {
        let pairs = (0..count).flat_map(|i| [0.999999 + i as f64, 1.000001 - i as f64]);
        if f64_variant {
            self.probe("fma_probe_f64", &pairs.collect::<Vec<f64>>(), count, iterations, repeats)
        } else {
            self.probe("fma_probe_f32", &pairs.map(|v| v as f32).collect::<Vec<f32>>(), count, iterations, repeats)
        }
    }

    fn probe<T: ocl::OclPrm>(
        &self, name: &str, source: &[T], count: usize, iterations: i32, repeats: usize,
    ) -> Result<f64, String> {
        // Typed buffers: ocl type-checks built-in pointer arguments against
        // the kernel signature, unlike the opaque struct pointers in prepare.
        let source_buf = Buffer::<T>::builder()
            .queue(self.queue.clone())
            .flags(flags::MEM_READ_ONLY)
            .len(source.len())
            .build()
            .map_err(|e| e.to_string())?;
        source_buf.cmd().write(source).enq().map_err(|e| e.to_string())?;
        let sink_buf = Buffer::<T>::builder()
            .queue(self.queue.clone())
            .flags(flags::MEM_WRITE_ONLY)
            .len(count)
            .build()
            .map_err(|e| e.to_string())?;
        let kernel = ocl::Kernel::builder()
            .program(&self.program)
            .name(name)
            .queue(self.queue.clone())
            .global_work_size(count)
            .arg(&source_buf)
            .arg(&sink_buf)
            .arg(&iterations)
            .build()
            .map_err(|e| e.to_string())?;
        unsafe { kernel.enq().map_err(|e| e.to_string())? };
        self.queue.finish().map_err(|e| e.to_string())?;
        let started = Instant::now();
        for _ in 0..repeats {
            unsafe { kernel.enq().map_err(|e| e.to_string())? };
            self.queue.finish().map_err(|e| e.to_string())?;
        }
        Ok(started.elapsed().as_secs_f64() * 1000. / repeats as f64)
    }

    /// Byte buffers carry the repr(C) struct arrays. ocl cannot type-check
    /// them because struct pointers are opaque in the kernel signature.
    fn buffer(&self, bytes: &[u8], flags: flags::MemFlags) -> Result<Buffer<u8>, String> {
        let buffer = Buffer::<u8>::builder()
            .queue(self.queue.clone())
            .flags(flags)
            .len(bytes.len())
            .build()
            .map_err(|e| e.to_string())?;
        buffer.cmd().write(bytes).enq().map_err(|e| e.to_string())?;
        Ok(buffer)
    }

    fn empty_buffer(&self, len: usize, flags: flags::MemFlags) -> Result<Buffer<u8>, String> {
        Buffer::<u8>::builder()
            .queue(self.queue.clone())
            .flags(flags)
            .len(len)
            .build()
            .map_err(|e| e.to_string())
    }
}

pub struct Job {
    queue: Queue,
    _params_buf: Buffer<u8>,
    cases_buf: Buffer<u8>,
    rows_buf: Buffer<u8>,
    kernel: ocl::Kernel,
}

impl Job {
    pub fn upload(&self, cases: &[KernelCase]) -> Result<f64, String> {
        let bytes = cases.iter().flat_map(|case| struct_bytes(case)).collect::<Vec<_>>();
        let started = Instant::now();
        self.cases_buf.cmd().write(&bytes).enq().map_err(|e| e.to_string())?;
        Ok(started.elapsed().as_secs_f64() * 1000.)
    }

    pub fn run(&self) -> Result<f64, String> {
        let started = Instant::now();
        unsafe { self.kernel.enq().map_err(|e| e.to_string())? };
        self.queue.finish().map_err(|e| e.to_string())?;
        Ok(started.elapsed().as_secs_f64() * 1000.)
    }

    pub fn read(&self) -> Result<(Vec<KernelRow>, f64), String> {
        let mut bytes = vec![0u8; self.rows_buf.len()];
        let started = Instant::now();
        self.rows_buf.cmd().read(&mut bytes).enq().map_err(|e| e.to_string())?;
        let ms = started.elapsed().as_secs_f64() * 1000.;
        Ok((rows_from_bytes(&bytes), ms))
    }
}

fn struct_bytes<T: Copy>(value: &T) -> Vec<u8> {
    // All spike structs are repr(C) f32/i32 with no padding. Host and device are little-endian.
    unsafe {
        std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()).to_vec()
    }
}

fn rows_from_bytes(bytes: &[u8]) -> Vec<KernelRow> {
    bytes
        .chunks_exact(std::mem::size_of::<KernelRow>())
        .map(|chunk| unsafe { std::ptr::read_unaligned(chunk.as_ptr() as *const KernelRow) })
        .collect()
}

#[cfg(test)]
mod program_cache_tests {
    use super::*;

    fn device() -> DeviceReport {
        DeviceReport {
            platform: "test platform".into(),
            name: "test device".into(),
            driver_version: "0.0.0.0".into(),
            compute_units: 0,
            max_work_group_size: 0,
            single_fp_config: 0,
            double_fp_config: 0,
        }
    }

    fn options() -> &'static str {
        "-D RACE_MAXR=18"
    }

    fn image(device: &DeviceReport, options: &str, bins: &[&[u8]]) -> Vec<u8> {
        let mut out = program_cache_prologue(device, options);
        out.extend_from_slice(&(bins.len() as u32).to_le_bytes());
        for bin in bins {
            out.extend_from_slice(&(bin.len() as u64).to_le_bytes());
            out.extend_from_slice(bin);
        }
        out
    }

    /// Offset of the u64 source digest, right after the u64 source length.
    fn source_digest_at(device: &DeviceReport, options: &str) -> usize {
        program_cache_prologue(device, options).len() - 8
    }

    #[test]
    fn a_saved_image_round_trips_for_its_own_identity() {
        let bytes = image(&device(), options(), &[&[1, 2, 3], &[4, 5]]);
        let bins = parse_program_binary(&bytes, &device(), options()).unwrap();
        assert_eq!(bins, vec![vec![1, 2, 3], vec![4, 5]]);
    }

    #[test]
    fn a_foreign_source_digest_is_rejected() {
        let bytes = image(&device(), options(), &[&[9, 9]]);
        let at = source_digest_at(&device(), options());
        for bit in [1u64, 1 << 32, 1 << 63] {
            let mut bad = bytes.clone();
            let digest = u64::from_le_bytes(bad[at..at + 8].try_into().unwrap()) ^ bit;
            bad[at..at + 8].copy_from_slice(&digest.to_le_bytes());
            let err = parse_program_binary(&bad, &device(), options()).unwrap_err();
            assert!(err.contains("digest"), "{err}");
        }
    }

    #[test]
    fn a_pre_fix_cache_file_is_not_loaded() {
        let mut bytes = image(&device(), options(), &[&[7]]);
        bytes[..8].copy_from_slice(b"GSIMPC01");
        assert!(parse_program_binary(&bytes, &device(), options()).is_err());
    }

    #[test]
    fn a_different_source_length_is_rejected() {
        let mut bytes = image(&device(), options(), &[&[7]]);
        let at = source_digest_at(&device(), options()) - 8;
        let longer = (RACE_KERNEL_SOURCE.len() as u64 + 1).to_le_bytes();
        bytes[at..at + 8].copy_from_slice(&longer);
        assert!(parse_program_binary(&bytes, &device(), options()).is_err());
    }

    #[test]
    fn device_driver_and_options_still_have_to_match() {
        let bytes = image(&device(), options(), &[&[1]]);
        assert!(parse_program_binary(&bytes, &device(), "-D OTHER").is_err());
        let mut other = device();
        other.driver_version = "1.0".into();
        assert!(parse_program_binary(&bytes, &other, options()).is_err());
        let mut renamed = device();
        renamed.name = "other device".into();
        assert!(parse_program_binary(&bytes, &renamed, options()).is_err());
    }

    #[test]
    fn the_file_name_keys_the_source_bytes_and_the_options() {
        let dir = std::path::Path::new("cache");
        let a = program_cache_path(dir, "-D A");
        let b = program_cache_path(dir, "-D B");
        assert_ne!(a, b);
        let want = format!(
            "race-{:016x}-{:016x}.bin",
            fnv1a64(RACE_KERNEL_SOURCE.as_bytes()),
            fnv1a64(b"-D A"),
        );
        assert_eq!(a.file_name().unwrap().to_str().unwrap(), want);
    }
}
