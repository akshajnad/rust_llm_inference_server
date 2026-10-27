//! Process memory as the OS accounts it. On Apple Silicon, GPU buffers come out of the same
//! unified memory but plain RSS misses them, so macOS uses the phys_footprint the `footprint`
//! tool and Activity Monitor report.

#[derive(Debug, Clone, Copy)]
pub struct Footprint {
    pub current_bytes: u64,
    pub peak_bytes: u64,
}

#[cfg(target_os = "macos")]
pub fn footprint() -> Option<Footprint> {
    // SAFETY: proc_pid_rusage fills the struct for the given flavor; it is zeroed beforehand.
    unsafe {
        let mut info: libc::rusage_info_v4 = std::mem::zeroed();
        let rc = libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            (&mut info as *mut libc::rusage_info_v4).cast(),
        );
        (rc == 0).then_some(Footprint {
            current_bytes: info.ri_phys_footprint,
            peak_bytes: info.ri_lifetime_max_phys_footprint,
        })
    }
}

/// Resident set size (VmRSS) and its high-water mark (VmHWM).
#[cfg(target_os = "linux")]
pub fn footprint() -> Option<Footprint> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kib = |key: &str| -> Option<u64> {
        let line = status.lines().find(|l| l.starts_with(key))?;
        line.split_whitespace()
            .nth(1)?
            .parse::<u64>()
            .ok()
            .map(|k| k * 1024)
    };
    Some(Footprint {
        current_bytes: kib("VmRSS:")?,
        peak_bytes: kib("VmHWM:")?,
    })
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn footprint() -> Option<Footprint> {
    None
}

#[cfg(test)]
mod tests {
    #[test]
    fn reports_a_plausible_footprint() {
        let f = super::footprint().expect("footprint available on this OS");
        assert!(f.current_bytes > 1 << 20, "{f:?}");
        assert!(f.peak_bytes >= f.current_bytes, "{f:?}");
    }
}
