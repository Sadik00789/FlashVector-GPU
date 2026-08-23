use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr::NonNull;
use crate::ffi::{
    gpu_free_device, gpu_free_host, gpu_malloc_device, gpu_malloc_host, gpu_memcpy_d2h_async,
    gpu_memcpy_h2d_async, CudaError,
};

/// RAII Page-Locked Pinned Host Memory Buffer for fast zero-copy / async DMA transfers
pub struct PinnedBuffer<T> {
    ptr: NonNull<T>,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for PinnedBuffer<T> {}
unsafe impl<T: Sync> Sync for PinnedBuffer<T> {}

impl<T> PinnedBuffer<T> {
    pub fn new(len: usize) -> Result<Self, CudaError> {
        if len == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len: 0,
                _marker: PhantomData,
            });
        }

        let raw_ptr = gpu_malloc_host::<T>(len)?;
        let non_null = NonNull::new(raw_ptr).ok_or(CudaError::DriverError(-1))?;

        // Zero-initialize pinned memory
        unsafe {
            std::ptr::write_bytes(raw_ptr, 0, len);
        }

        Ok(Self {
            ptr: non_null,
            len,
            _marker: PhantomData,
        })
    }

    pub fn from_slice(slice: &[T]) -> Result<Self, CudaError>
    where
        T: Clone,
    {
        let mut buf = Self::new(slice.len())?;
        buf.copy_from_slice(slice);
        Ok(buf)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }

    pub fn copy_from_slice(&mut self, slice: &[T])
    where
        T: Clone,
    {
        assert_eq!(self.len, slice.len(), "PinnedBuffer slice length mismatch");
        self.as_mut_slice().clone_from_slice(slice);
    }
}

impl<T> Deref for PinnedBuffer<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for PinnedBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T> Index<usize> for PinnedBuffer<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<T> IndexMut<usize> for PinnedBuffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}

impl<T> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if self.len > 0 {
            let _ = gpu_free_host(self.ptr.as_ptr());
        }
    }
}

/// RAII GPU Device VRAM Buffer
pub struct DeviceBuffer<T> {
    ptr: NonNull<T>,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for DeviceBuffer<T> {}
unsafe impl<T: Sync> Sync for DeviceBuffer<T> {}

impl<T> DeviceBuffer<T> {
    pub fn new(len: usize) -> Result<Self, CudaError> {
        if len == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len: 0,
                _marker: PhantomData,
            });
        }

        let raw_ptr = gpu_malloc_device::<T>(len)?;
        let non_null = NonNull::new(raw_ptr).ok_or(CudaError::DriverError(-1))?;

        Ok(Self {
            ptr: non_null,
            len,
            _marker: PhantomData,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn copy_from_host_async(
        &mut self,
        src: &[T],
        stream: *mut libc::c_void,
    ) -> Result<(), CudaError> {
        assert!(self.len >= src.len(), "DeviceBuffer capacity exceeded");
        gpu_memcpy_h2d_async(self.ptr.as_ptr(), src.as_ptr(), src.len(), stream)
    }

    pub fn copy_to_host_async(
        &self,
        dst: &mut [T],
        stream: *mut libc::c_void,
    ) -> Result<(), CudaError> {
        assert!(self.len >= dst.len(), "DeviceBuffer capacity exceeded");
        gpu_memcpy_d2h_async(dst.as_mut_ptr(), self.ptr.as_ptr(), dst.len(), stream)
    }

    pub fn copy_from_pinned_async(
        &mut self,
        pinned: &PinnedBuffer<T>,
        stream: *mut libc::c_void,
    ) -> Result<(), CudaError> {
        assert!(self.len >= pinned.len(), "DeviceBuffer capacity exceeded");
        gpu_memcpy_h2d_async(self.ptr.as_ptr(), pinned.as_ptr(), pinned.len(), stream)
    }

    pub fn copy_to_pinned_async(
        &self,
        pinned: &mut PinnedBuffer<T>,
        stream: *mut libc::c_void,
    ) -> Result<(), CudaError> {
        assert!(self.len >= pinned.len(), "DeviceBuffer capacity exceeded");
        gpu_memcpy_d2h_async(pinned.as_mut_ptr(), self.ptr.as_ptr(), pinned.len(), stream)
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.len > 0 {
            let _ = gpu_free_device(self.ptr.as_ptr());
        }
    }
}
