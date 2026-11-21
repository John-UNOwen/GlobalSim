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
/// lanes. The game never fields more than 18, so 18 is the default and keeps a
/// race slot small enough to launch thousands at once. `wide-roster` restores
/// the engine's 120-horse grid band (experimentalGrid 21 to 120) on the GPU;
/// without it those rosters refuse by size and the CPU engine answers them.
#[cfg(feature = "wide-roster")]
pub const ROSTER_CAP: usize = 120;
#[cfg(not(feature = "wide-roster"))]
pub const ROSTER_CAP: usize = 18;

/// `BUILD_OPTIONS` plus the capacity defines, so the kernels' `RACE_MAXR` and
/// `CLS_MAXR` always equal the host's strides.
pub fn build_options() -> String {
    format!("{BUILD_OPTIONS} -D RACE_MAXR={ROSTER_CAP} -D CLS_MAXR={ROSTER_CAP}")
}
// The classic lane carries the bit-exact powf table for the mid-race stat
// refresh (race_solver.rs:1330-1340 recomputes the guts and power derived
// terms when a delivered stat effect lands); construction stays host side.
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
const CL_DEVICE_SINGLE_FP_CONFIG: u32 = 0x101B;
const CL_DEVICE_DOUBLE_FP_CONFIG: u32 = 0x1032;
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
    pub compute_units: u64,
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
                compute_units: read_u64(&device, CL_DEVICE_MAX_COMPUTE_UNITS).unwrap_or(0),
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
            program,
            device: report,
            build_ms,
            build_failures: Cell::new(0),
        })
    }

    /// Builds an extra kernel program (for example rng.cl) on the already
    /// gated device, with the same exactness build options.
    pub fn build_program(&self, source: &str) -> Result<Program, String> {
        Program::builder()
            .cmplr_opt(build_options())
            .source(source)
            .build(&self.context)
            .map_err(|e| e.to_string())
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
