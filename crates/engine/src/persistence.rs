//! Zero-copy index persistence via 64-byte aligned binary snapshots and OS `mmap`.
//!
//! File layout:
//! - `[0..8)`: magic `b"FLSHVEC1"`
//! - `[8..16)`: little-endian `u64` JSON header length `H`
//! - `[16..16 + H)`: UTF-8 JSON `SnapshotManifest` describing config + sections
//! - `[data_start ..)`: zero-padded to 64-byte boundary; each binary section is
//!   64-byte aligned so its slice can be DMA-transferred directly to VRAM via
//!   `cudaMemcpyAsync`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::ffi::{CudaError, IndexConfig};

const MAGIC: &[u8; 8] = b"FLSHVEC1";
const ALIGNMENT: usize = 64;

#[inline]
fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionMeta {
    pub offset: usize, // relative to `data_start`
    pub byte_len: usize,
    pub dtype: String,
    pub len: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotManifest {
    pub version: u32,
    pub config: IndexConfig,
    pub num_vectors: usize,
    pub entry_point: u32,
    pub cagra_degree: usize,
    pub cagra_seeds: Vec<u32>,
    pub rabitq_padded_dim: usize,
    pub fp16_padded_dim: usize,
    pub sparse_num_docs: usize,
    pub sparse_nnz: usize,
    pub next_logical_id: u32,
    pub sections: BTreeMap<String, SectionMeta>,
}

/// Write a 64-byte-aligned binary snapshot file from named raw byte slices.
pub fn write_aligned_snapshot(
    path: &Path,
    mut manifest: SnapshotManifest,
    sections: Vec<(&str, &str, usize, &[u8])>, // (name, dtype, elem_len, bytes)
) -> Result<(), CudaError> {
    let mut cursor = 0usize;
    for (name, dtype, elem_len, bytes) in &sections {
        cursor = align_up(cursor, ALIGNMENT);
        manifest.sections.insert(
            (*name).to_string(),
            SectionMeta {
                offset: cursor,
                byte_len: bytes.len(),
                dtype: (*dtype).to_string(),
                len: *elem_len,
            },
        );
        cursor += bytes.len();
    }

    let json_bytes = serde_json::to_vec(&manifest)
        .map_err(|e| CudaError::IoError(format!("manifest serialize failed: {e}")))?;
    let header_end = 16 + json_bytes.len();
    let data_start = align_up(header_end, ALIGNMENT);
    let header_pad = data_start - header_end;

    let file = File::create(path)
        .map_err(|e| CudaError::IoError(format!("create snapshot {}: {e}", path.display())))?;
    let mut writer = BufWriter::new(file);
    let write_err = |e: std::io::Error| CudaError::IoError(format!("write snapshot: {e}"));

    writer.write_all(MAGIC).map_err(write_err)?;
    writer
        .write_all(&(json_bytes.len() as u64).to_le_bytes())
        .map_err(write_err)?;
    writer.write_all(&json_bytes).map_err(write_err)?;
    if header_pad > 0 {
        writer
            .write_all(&vec![0u8; header_pad])
            .map_err(write_err)?;
    }

    let mut written = 0usize;
    let zeros = [0u8; ALIGNMENT];
    for (_, _, _, bytes) in sections {
        let aligned = align_up(written, ALIGNMENT);
        if aligned > written {
            writer
                .write_all(&zeros[..aligned - written])
                .map_err(write_err)?;
            written = aligned;
        }
        writer.write_all(bytes).map_err(write_err)?;
        written += bytes.len();
    }

    writer.flush().map_err(write_err)?;
    Ok(())
}

/// View raw typed slice as little-endian bytes (host little-endian platforms).
#[inline]
pub fn as_byte_slice<T: Copy>(slice: &[T]) -> &[u8] {
    if slice.is_empty() {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

/// Read-only memory-mapped snapshot file backed by OS virtual memory.
pub struct MmapSnapshot {
    mapping: OsMmap,
    pub manifest: SnapshotManifest,
    data_start: usize,
}

impl MmapSnapshot {
    pub fn open(path: &Path) -> Result<Self, CudaError> {
        let mapping = OsMmap::open_readonly(path)?;
        let bytes = mapping.as_slice();
        if bytes.len() < 16 || &bytes[0..8] != MAGIC {
            return Err(CudaError::IoError("invalid snapshot magic header".into()));
        }
        let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let header_end = 16usize
            .checked_add(header_len)
            .ok_or_else(|| CudaError::IoError("snapshot header overflow".into()))?;
        if header_end > bytes.len() {
            return Err(CudaError::IoError("truncated snapshot header".into()));
        }
        let manifest: SnapshotManifest = serde_json::from_slice(&bytes[16..header_end])
            .map_err(|e| CudaError::IoError(format!("invalid manifest JSON: {e}")))?;
        let data_start = align_up(header_end, ALIGNMENT);
        if data_start > bytes.len() {
            return Err(CudaError::IoError("truncated snapshot payload".into()));
        }

        for (name, sec) in &manifest.sections {
            let end = data_start
                .checked_add(sec.offset)
                .and_then(|v| v.checked_add(sec.byte_len))
                .ok_or_else(|| CudaError::IoError(format!("section overflow: {name}")))?;
            if end > bytes.len() {
                return Err(CudaError::IoError(format!("truncated section: {name}")));
            }
        }

        Ok(Self {
            mapping,
            manifest,
            data_start,
        })
    }

    pub fn section_bytes(&self, name: &str) -> Option<&[u8]> {
        let sec = self.manifest.sections.get(name)?;
        let start = self.data_start + sec.offset;
        Some(&self.mapping.as_slice()[start..start + sec.byte_len])
    }

    /// Zero-copy typed view directly into the memory-mapped 64-byte-aligned file.
    pub fn section_typed<T: Copy>(&self, name: &str) -> Result<Option<&[T]>, CudaError> {
        let Some(bytes) = self.section_bytes(name) else {
            return Ok(None);
        };
        let elem_size = std::mem::size_of::<T>();
        if elem_size == 0 || !bytes.len().is_multiple_of(elem_size) {
            return Err(CudaError::IoError(format!(
                "section {name} byte size {} incompatible with element size {}",
                bytes.len(),
                elem_size
            )));
        }
        if bytes.is_empty() {
            return Ok(Some(&[]));
        }
        let ptr = bytes.as_ptr();
        if !(ptr as usize).is_multiple_of(std::mem::align_of::<T>()) {
            return Err(CudaError::IoError(format!(
                "section {name} is misaligned for {}",
                std::any::type_name::<T>()
            )));
        }
        let count = bytes.len() / elem_size;
        // SAFETY: `bytes` points into a live read-only mmap region whose length
        // and alignment were validated above, and `T` is plain Copy data.
        Ok(Some(unsafe {
            std::slice::from_raw_parts(ptr as *const T, count)
        }))
    }
}

struct OsMmap {
    ptr: *const u8,
    len: usize,
    #[cfg(windows)]
    mapping_handle: isize,
    #[cfg(windows)]
    file_handle: isize,
}

unsafe impl Send for OsMmap {}
unsafe impl Sync for OsMmap {}

impl OsMmap {
    #[cfg(unix)]
    fn open_readonly(path: &Path) -> Result<Self, CudaError> {
        use std::os::unix::io::AsRawFd;
        let file = File::open(path)
            .map_err(|e| CudaError::IoError(format!("open snapshot {}: {e}", path.display())))?;
        let len = file
            .metadata()
            .map_err(|e| CudaError::IoError(format!("stat snapshot: {e}")))?
            .len() as usize;
        if len == 0 {
            return Err(CudaError::IoError("empty snapshot file".into()));
        }
        let mapped = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if mapped == libc::MAP_FAILED {
            return Err(CudaError::IoError(format!(
                "mmap failed: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(Self {
            ptr: mapped as *const u8,
            len,
        })
    }

    #[cfg(windows)]
    fn open_readonly(path: &Path) -> Result<Self, CudaError> {
        use std::os::windows::io::IntoRawHandle;

        const PAGE_READONLY: u32 = 0x02;
        const FILE_MAP_READ: u32 = 0x04;

        extern "system" {
            fn CreateFileMappingW(
                hFile: isize,
                lpAttributes: *const libc::c_void,
                flProtect: u32,
                dwMaximumSizeHigh: u32,
                dwMaximumSizeLow: u32,
                lpName: *const u16,
            ) -> isize;
            fn MapViewOfFile(
                hFileMappingObject: isize,
                dwDesiredAccess: u32,
                dwFileOffsetHigh: u32,
                dwFileOffsetLow: u32,
                dwNumberOfBytesToMap: usize,
            ) -> *const u8;
            fn CloseHandle(hObject: isize) -> i32;
        }

        let file = File::open(path)
            .map_err(|e| CudaError::IoError(format!("open snapshot {}: {e}", path.display())))?;
        let len = file
            .metadata()
            .map_err(|e| CudaError::IoError(format!("stat snapshot: {e}")))?
            .len() as usize;
        if len == 0 {
            return Err(CudaError::IoError("empty snapshot file".into()));
        }
        let file_handle = file.into_raw_handle() as isize;
        let mapping_handle = unsafe {
            CreateFileMappingW(
                file_handle,
                std::ptr::null(),
                PAGE_READONLY,
                0,
                0,
                std::ptr::null(),
            )
        };
        if mapping_handle == 0 {
            unsafe { CloseHandle(file_handle) };
            return Err(CudaError::IoError(format!(
                "CreateFileMappingW failed: {}",
                std::io::Error::last_os_error()
            )));
        }
        let ptr = unsafe { MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, len) };
        if ptr.is_null() {
            unsafe {
                CloseHandle(mapping_handle);
                CloseHandle(file_handle);
            }
            return Err(CudaError::IoError(format!(
                "MapViewOfFile failed: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(Self {
            ptr,
            len,
            mapping_handle,
            file_handle,
        })
    }

    #[inline]
    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for OsMmap {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
        #[cfg(windows)]
        unsafe {
            extern "system" {
                fn UnmapViewOfFile(lpBaseAddress: *const u8) -> i32;
                fn CloseHandle(hObject: isize) -> i32;
            }
            UnmapViewOfFile(self.ptr);
            CloseHandle(self.mapping_handle);
            CloseHandle(self.file_handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_snapshot_roundtrip_mmap() {
        let mut dir = std::env::temp_dir();
        dir.push(format!("flashvector_snap_test_{}.fv", std::process::id()));

        let vectors: Vec<f32> = (0..64).map(|i| i as f32 * 0.5).collect();
        let adj: Vec<u32> = (0..32).collect();
        let manifest = SnapshotManifest {
            version: 1,
            config: IndexConfig::default(),
            num_vectors: 16,
            entry_point: 3,
            cagra_degree: 32,
            cagra_seeds: vec![3, 7],
            rabitq_padded_dim: 128,
            fp16_padded_dim: 16,
            sparse_num_docs: 0,
            sparse_nnz: 0,
            next_logical_id: 16,
            sections: BTreeMap::new(),
        };

        write_aligned_snapshot(
            &dir,
            manifest,
            vec![
                ("vectors", "f32", vectors.len(), as_byte_slice(&vectors)),
                ("adjacency", "u32", adj.len(), as_byte_slice(&adj)),
            ],
        )
        .unwrap();

        let snap = MmapSnapshot::open(&dir).unwrap();
        assert_eq!(snap.manifest.entry_point, 3);
        let mapped_vecs = snap.section_typed::<f32>("vectors").unwrap().unwrap();
        let mapped_adj = snap.section_typed::<u32>("adjacency").unwrap().unwrap();
        assert_eq!(mapped_vecs, &vectors[..]);
        assert_eq!(mapped_adj, &adj[..]);
        assert_eq!((mapped_vecs.as_ptr() as usize) % 64, 0);
        assert_eq!((mapped_adj.as_ptr() as usize) % 64, 0);

        drop(snap);
        let _ = std::fs::remove_file(&dir);
    }
}
