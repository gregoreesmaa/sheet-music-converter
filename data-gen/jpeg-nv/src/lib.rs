//! Minimal safe wrapper over nvJPEG encode (interleaved RGB in, JPEG out).
//!
//! `encode_rgb_jpeg` uploads the pixels, encodes on the GPU (synchronous
//! NULL-stream calls), and downloads the bitstream. One nvJPEG handle per
//! process, one encoder state per thread. Any failure returns `Err` and the
//! caller uses its CPU fallback — including when built without CUDA
//! (`no_nvjpeg` cfg from build.rs) or when no GPU exists at runtime.

use std::ffi::c_void;
use std::os::raw::{c_int, c_uchar};
use std::sync::OnceLock;

#[cfg(not(no_nvjpeg))]
mod imp {
    use super::*;

    pub type NvjpegHandle = *mut c_void;
    pub type NvjpegState = *mut c_void;
    pub type NvjpegParams = *mut c_void;
    pub type CudaStream = *mut c_void;

    #[repr(C)]
    pub struct NvjpegImage {
        pub channel: [*mut c_uchar; 4],
        pub pitch: [usize; 4],
    }

    pub const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
    pub const NVJPEG_INPUT_RGBI: c_int = 5;
    pub const NVJPEG_CSS_420: c_int = 2;
    pub const NVJPEG_CSS_444: c_int = 0;

    extern "C" {
        pub fn cudaSetDevice(device: c_int) -> c_int;
        pub fn cudaMalloc(devptr: *mut *mut c_void, size: usize) -> c_int;
        pub fn cudaMemcpy(
            dst: *mut c_void,
            src: *const c_void,
            count: usize,
            kind: c_int,
        ) -> c_int;
        pub fn cudaFree(devptr: *mut c_void) -> c_int;
        pub fn nvjpegCreateSimple(handle: *mut NvjpegHandle) -> c_int;
        pub fn nvjpegEncoderStateCreate(
            handle: NvjpegHandle,
            state: *mut NvjpegState,
            stream: CudaStream,
        ) -> c_int;
        pub fn nvjpegEncoderStateDestroy(state: NvjpegState) -> c_int;
        pub fn nvjpegEncoderParamsCreate(
            handle: NvjpegHandle,
            params: *mut NvjpegParams,
            stream: CudaStream,
        ) -> c_int;
        pub fn nvjpegEncoderParamsDestroy(params: NvjpegParams) -> c_int;
        pub fn nvjpegEncoderParamsSetQuality(
            params: NvjpegParams,
            quality: c_int,
            stream: CudaStream,
        ) -> c_int;
        pub fn nvjpegEncoderParamsSetSamplingFactors(
            params: NvjpegParams,
            subsampling: c_int,
            stream: CudaStream,
        ) -> c_int;
        #[allow(clippy::too_many_arguments)]
        pub fn nvjpegEncodeImage(
            handle: NvjpegHandle,
            state: NvjpegState,
            params: NvjpegParams,
            source: *const NvjpegImage,
            input_format: c_int,
            image_width: c_int,
            image_height: c_int,
            stream: CudaStream,
        ) -> c_int;
        pub fn nvjpegEncodeRetrieveBitstream(
            handle: NvjpegHandle,
            state: NvjpegState,
            data: *mut c_uchar,
            length: *mut usize,
            stream: CudaStream,
        ) -> c_int;
    }

    /// Opaque nvJPEG handle. Safe to move/share: the handle itself is just
    /// a token; all mutable work goes through per-thread encoder states,
    /// and nvJPEG handles support concurrent use with separate states.
    #[derive(Clone, Copy)]
    struct Handle(NvjpegHandle);
    unsafe impl Send for Handle {}
    unsafe impl Sync for Handle {}

    fn handle() -> Result<NvjpegHandle, String> {
        static HANDLE: OnceLock<Result<Handle, String>> = OnceLock::new();
        match HANDLE.get_or_init(|| unsafe {
            if cudaSetDevice(0) != 0 {
                return Err("nvjpeg: no CUDA device".to_string());
            }
            let mut h: NvjpegHandle = std::ptr::null_mut();
            if nvjpegCreateSimple(&mut h) != 0 || h.is_null() {
                return Err("nvjpeg: nvjpegCreateSimple failed".to_string());
            }
            Ok(Handle(h))
        }) {
            Ok(h) => Ok(h.0),
            Err(e) => Err(e.clone()),
        }
    }

    struct ThreadState {
        state: NvjpegState,
        device_set: bool,
    }

    impl Drop for ThreadState {
        fn drop(&mut self) {
            unsafe {
                nvjpegEncoderStateDestroy(self.state);
            }
        }
    }

    thread_local! {
        static STATE: std::cell::RefCell<Option<ThreadState>> =
            std::cell::RefCell::new(None);
        /// Persistent per-thread device scratch for the upload. Pages are
        /// near-uniform size, so the buffer is allocated once and reused;
        /// it only grows when a larger frame arrives. Freed at thread exit.
        static DEVICE_BUF: std::cell::RefCell<DeviceBuf> =
            std::cell::RefCell::new(DeviceBuf { ptr: std::ptr::null_mut(), cap: 0 });
    }

    /// Owned device allocation with thread-exit cleanup.
    struct DeviceBuf {
        ptr: *mut c_void,
        cap: usize,
    }

    impl Drop for DeviceBuf {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe {
                    cudaFree(self.ptr);
                }
                self.ptr = std::ptr::null_mut();
                self.cap = 0;
            }
        }
    }

    /// Borrow the persistent device buffer, growing it when `need` exceeds
    /// the current capacity. The returned pointer stays valid until the next
    /// growth (same thread) or thread exit.
    fn device_scratch(need: usize) -> Result<*mut c_void, String> {
        DEVICE_BUF.with(|cell| unsafe {
            let mut slot = cell.borrow_mut();
            if slot.cap < need {
                if !slot.ptr.is_null() {
                    check(cudaFree(slot.ptr), "cudaFree(grow)")?;
                    slot.ptr = std::ptr::null_mut();
                    slot.cap = 0;
                }
                let mut dptr: *mut c_void = std::ptr::null_mut();
                check(cudaMalloc(&mut dptr, need), "cudaMalloc")?;
                slot.ptr = dptr;
                slot.cap = need;
            }
            Ok(slot.ptr)
        })
    }

    #[cfg(test)]
    pub fn device_scratch_capacity() -> usize {
        DEVICE_BUF.with(|cell| cell.borrow().cap)
    }

    fn check(code: c_int, what: &str) -> Result<(), String> {
        if code == 0 {
            Ok(())
        } else {
            Err(format!("nvjpeg: {what} failed (code {code})"))
        }
    }

    /// Encode interleaved RGB (`w*h*3` bytes) to a JPEG blob at `quality`
    /// 1..=100. Mirrors the jpeg-encoder crate's chroma rule (4:2:0 below
    /// q90, 4:4:4 at/above) so output size/quality stay comparable.
    pub fn encode_rgb_jpeg(
        pixels: &[u8],
        w: u32,
        h: u32,
        quality: u8,
    ) -> Result<Vec<u8>, String> {
        if !(1..=100).contains(&quality) {
            return Err(format!("nvjpeg: bad quality {quality}"));
        }
        let w64 = w as u64;
        let h64 = h as u64;
        let need = w64
            .checked_mul(h64)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| "nvjpeg: dims overflow".to_string())?;
        if pixels.len() as u64 != need {
            return Err(format!(
                "nvjpeg: pixel len {} != {w}x{h}x3",
                pixels.len()
            ));
        }
        if w > 65535 || h > 65535 {
            return Err("nvjpeg: dims exceed 65535".to_string());
        }
        let handle = handle()?;
        unsafe {
            STATE.with(|cell| {
                let mut slot = cell.borrow_mut();
                if slot.as_ref().is_none_or(|s| !s.device_set) {
                    // (Re)establish this thread's device + state.
                    check(cudaSetDevice(0), "cudaSetDevice")?;
                    let mut st: NvjpegState = std::ptr::null_mut();
                    check(
                        nvjpegEncoderStateCreate(handle, &mut st, std::ptr::null_mut()),
                        "nvjpegEncoderStateCreate",
                    )?;
                    *slot = Some(ThreadState {
                        state: st,
                        device_set: true,
                    });
                }
                let st = slot.as_ref().unwrap().state;
                // Fresh params per call: quality varies per file.
                let mut params: NvjpegParams = std::ptr::null_mut();
                check(
                    nvjpegEncoderParamsCreate(handle, &mut params, std::ptr::null_mut()),
                    "nvjpegEncoderParamsCreate",
                )?;
                let rc = (|| {
                    check(
                        nvjpegEncoderParamsSetQuality(
                            params,
                            quality as c_int,
                            std::ptr::null_mut(),
                        ),
                        "nvjpegEncoderParamsSetQuality",
                    )?;
                    check(
                        nvjpegEncoderParamsSetSamplingFactors(
                            params,
                            if quality < 90 {
                                NVJPEG_CSS_420
                            } else {
                                NVJPEG_CSS_444
                            },
                            std::ptr::null_mut(),
                        ),
                        "nvjpegEncoderParamsSetSamplingFactors",
                    )?;
                    // Persistent per-thread device scratch (grows on
                    // demand): removes the per-file cudaMalloc/cudaFree
                    // pair from the hot path. NULL stream stays
                    // synchronous, so no explicit sync is needed.
                    let dptr = device_scratch(need as usize)?;
                    let rc = (|| {
                        check(
                            cudaMemcpy(
                                dptr,
                                pixels.as_ptr() as *const c_void,
                                need as usize,
                                CUDA_MEMCPY_HOST_TO_DEVICE,
                            ),
                            "cudaMemcpy H2D",
                        )?;
                        let desc = NvjpegImage {
                            channel: [dptr as *mut c_uchar, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut()],
                            pitch: [(w as usize) * 3, 0, 0, 0],
                        };
                        check(
                            nvjpegEncodeImage(
                                handle,
                                st,
                                params,
                                &desc,
                                NVJPEG_INPUT_RGBI,
                                w as c_int,
                                h as c_int,
                                std::ptr::null_mut(),
                            ),
                            "nvjpegEncodeImage",
                        )?;
                        // Size query, then the real download (NULL stream =
                        // synchronous, so no explicit sync needed).
                        let mut len: usize = 0;
                        check(
                            nvjpegEncodeRetrieveBitstream(
                                handle,
                                st,
                                std::ptr::null_mut(),
                                &mut len,
                                std::ptr::null_mut(),
                            ),
                            "nvjpegEncodeRetrieveBitstream(size)",
                        )?;
                        if len == 0 || len > 1 << 31 {
                            return Err(format!("nvjpeg: bad bitstream len {len}"));
                        }
                        let mut out = vec![0u8; len];
                        let mut got = len;
                        check(
                            nvjpegEncodeRetrieveBitstream(
                                handle,
                                st,
                                out.as_mut_ptr(),
                                &mut got,
                                std::ptr::null_mut(),
                            ),
                            "nvjpegEncodeRetrieveBitstream(data)",
                        )?;
                        out.truncate(got);
                        Ok(out)
                    })();
                    rc
                })();
                nvjpegEncoderParamsDestroy(params);
                rc
            })
        }
    }

    /// True when a CUDA device + nvJPEG initialized on this process.
    /// Probe failure is cached; callers fall back to CPU in that case.
    pub fn is_available() -> bool {
        handle().is_ok()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn device_buffer_persists_across_encodes() {
            // Skips cleanly on CUDA-less machines (same pattern as callers).
            if !is_available() {
                eprintln!("SKIP device_buffer_persists_across_encodes: no CUDA/nvJPEG here");
                return;
            }
            let rgb = vec![128u8; 64 * 64 * 3];
            encode_rgb_jpeg(&rgb, 64, 64, 90).unwrap();
            let c1 = device_scratch_capacity();
            assert!(c1 >= 64 * 64 * 3, "scratch must hold the frame, got {c1}");
            encode_rgb_jpeg(&rgb, 64, 64, 90).unwrap();
            assert_eq!(
                device_scratch_capacity(),
                c1,
                "second same-size encode must reuse the buffer"
            );
            // Growth path: a larger frame reallocates once, then reuses.
            let big = vec![128u8; 128 * 128 * 3];
            encode_rgb_jpeg(&big, 128, 128, 90).unwrap();
            let c2 = device_scratch_capacity();
            assert!(c2 >= 128 * 128 * 3, "scratch must grow, got {c2}");
            encode_rgb_jpeg(&big, 128, 128, 90).unwrap();
            assert_eq!(device_scratch_capacity(), c2, "grown buffer must persist");
        }
    }
}

#[cfg(not(no_nvjpeg))]
pub use imp::{encode_rgb_jpeg, is_available};

#[cfg(no_nvjpeg)]
mod stub {
    /// CUDA-less build: always unavailable; callers use the CPU fallback.
    pub fn is_available() -> bool {
        false
    }

    /// CUDA-less build: always fails; callers use the CPU fallback.
    pub fn encode_rgb_jpeg(
        _pixels: &[u8],
        _w: u32,
        _h: u32,
        _quality: u8,
    ) -> Result<Vec<u8>, String> {
        Err("nvjpeg unavailable (built without CUDA)".to_string())
    }
}

#[cfg(no_nvjpeg)]
pub use stub::{encode_rgb_jpeg, is_available};


