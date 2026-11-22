use std::path::PathBuf;

/// Locate the CUDA toolkit: `$CUDA_PATH`, else the standard install dir.
/// Emits `no_nvjpeg` cfg when nvjpeg.lib is absent so the crate (and its
/// dependents) still build on CUDA-less machines — the safe API then
/// reports unavailable and callers use their CPU fallback.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(no_nvjpeg)");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    let dir: Option<PathBuf> = std::env::var_os("CUDA_PATH")
        .map(PathBuf::from)
        .filter(|p| p.is_dir())
        .or_else(|| {
            let d = PathBuf::from(
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3",
            );
            d.is_dir().then_some(d)
        });
    match dir {
        Some(d) if d.join("lib").join("x64").join("nvjpeg.lib").is_file() => {
            println!("cargo:rustc-link-search=native={}\\lib\\x64", d.display());
            println!("cargo:rustc-link-lib=nvjpeg");
            println!("cargo:rustc-link-lib=cudart");
        }
        _ => {
            println!("cargo:rustc-cfg=no_nvjpeg");
        }
    }
}
