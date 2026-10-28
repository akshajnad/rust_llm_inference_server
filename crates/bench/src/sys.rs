//! Process memory of another process, and the hardware/software provenance every published
//! number carries.

use std::process::Command;

use serde::{Deserialize, Serialize};

/// macOS phys_footprint (counts Metal allocations RSS misses).
#[cfg(target_os = "macos")]
pub fn footprint_bytes(pid: u32) -> Option<u64> {
    // SAFETY: proc_pid_rusage fills the zeroed struct for the given flavor.
    unsafe {
        let mut info: libc::rusage_info_v4 = std::mem::zeroed();
        let rc = libc::proc_pid_rusage(
            pid as i32,
            libc::RUSAGE_INFO_V4,
            (&mut info as *mut libc::rusage_info_v4).cast(),
        );
        (rc == 0).then_some(info.ri_phys_footprint)
    }
}

/// Linux: VmRSS plus the memory in use on the GPU, which RSS doesn't count.
#[cfg(target_os = "linux")]
pub fn footprint_bytes(pid: u32) -> Option<u64> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    let line = status.lines().find(|l| l.starts_with("VmRSS:"))?;
    let rss = line.split_whitespace().nth(1)?.parse::<u64>().ok()? * 1024;
    Some(rss + gpu_memory_used_bytes().unwrap_or(0))
}

/// Memory in use on the first GPU in `CUDA_VISIBLE_DEVICES`, per nvidia-smi. Device-wide, not
/// per process (container PIDs don't match the driver's), which is the server's as long as
/// only one server runs on the GPU at a time.
#[cfg(target_os = "linux")]
fn gpu_memory_used_bytes() -> Option<u64> {
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default();
    let index = visible
        .split(',')
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or("0");
    let mib = cmd(
        "nvidia-smi",
        &[
            "-i",
            index,
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
    )?;
    Some(mib.trim().parse::<u64>().ok()? << 20)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn footprint_bytes(_pid: u32) -> Option<u64> {
    None
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    pub label: String,
    pub engine: String,
    pub model: String,
    pub quantization: String,
    pub gpu: String,
    pub gpu_driver: String,
    pub os: String,
    pub commit: String,
    pub timestamp: String,
}

fn cmd(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// GPU name and driver/OS-level version string.
pub fn gpu() -> (String, String) {
    if let Some(nv) = cmd(
        "nvidia-smi",
        &["--query-gpu=name,driver_version", "--format=csv,noheader"],
    ) {
        let mut parts = nv.lines().next().unwrap_or_default().split(", ");
        let name = parts.next().unwrap_or_default().to_string();
        let driver = parts.next().unwrap_or_default().to_string();
        let cuda = cmd("nvcc", &["--version"])
            .and_then(|v| v.lines().last().map(str::to_string))
            .unwrap_or_default();
        return (name, format!("NVIDIA driver {driver}; {cuda}"));
    }
    let sp = cmd("system_profiler", &["SPDisplaysDataType"]).unwrap_or_default();
    let field = |key: &str| {
        sp.lines()
            .find_map(|l| l.trim().strip_prefix(key))
            .map(|v| v.trim().to_string())
            .unwrap_or_default()
    };
    let name = format!(
        "{} ({} GPU cores)",
        field("Chipset Model:"),
        field("Total Number of Cores:")
    );
    (
        name,
        format!(
            "{} (macOS {})",
            field("Metal Support:"),
            cmd("sw_vers", &["-productVersion"]).unwrap_or_default()
        ),
    )
}

pub fn os() -> String {
    cmd("uname", &["-srm"]).unwrap_or_default()
}

/// Short commit hash, with `-dirty` when the working tree has changes.
pub fn commit() -> String {
    let hash = cmd("git", &["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into());
    match cmd("git", &["status", "--porcelain", "--untracked-files=no"]) {
        Some(s) if !s.is_empty() => format!("{hash}-dirty"),
        _ => hash,
    }
}

pub fn timestamp() -> String {
    cmd("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]).unwrap_or_default()
}
