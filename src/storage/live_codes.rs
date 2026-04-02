use memmap2::{MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::path::PathBuf;

const GROW_SLOTS: usize = 16384;

/// Memory-mapped slab for quantized vectors or raw float vectors.
///
/// # Windows safety
/// On Windows, a file cannot be renamed or overwritten while any handle (mmap
/// *or* the underlying `File`) is open against it.  Call [`release_handles`]
/// before any `fs::rename` / overwrite that targets this file; the next call
/// to any mutating method will reopen the handle automatically.
pub struct LiveCodesFile {
    path: PathBuf,
    file: Option<File>,
    mmap: Option<MmapMut>,
    stride: usize,
    capacity: usize,
    len: usize,
}

impl LiveCodesFile {
    pub fn open(
        path: PathBuf,
        stride: usize,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)?;

        let metadata = file.metadata()?;
        let file_size = metadata.len() as usize;
        let capacity = if stride > 0 { file_size / stride } else { 0 };
        let len = capacity;

        let mut live_codes = Self {
            path,
            file: Some(file),
            mmap: None,
            stride,
            capacity,
            len,
        };

        if file_size > 0 {
            live_codes.remap()?;
        }

        Ok(live_codes)
    }

    /// Ensure the file handle is open, reopening from the stored path if needed.
    fn ensure_open(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.file.is_none() {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(&self.path)?;
            self.file = Some(file);
        }
        Ok(())
    }

    fn remap(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.mmap = None;
        self.ensure_open()?;
        if self.capacity > 0 {
            let mmap = unsafe {
                MmapOptions::new().map_mut(self.file.as_ref().expect("file handle open"))?
            };
            self.mmap = Some(mmap);
        }
        Ok(())
    }

    pub fn get_slot(&self, slot: usize) -> &[u8] {
        let start = slot * self.stride;
        let end = start + self.stride;
        &self.mmap.as_ref().expect("mmap not initialized")[start..end]
    }

    pub fn get_slot_mut(&mut self, slot: usize) -> &mut [u8] {
        let start = slot * self.stride;
        let end = start + self.stride;
        &mut self.mmap.as_mut().expect("mmap not initialized")[start..end]
    }

    pub fn alloc_slot(&mut self) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        if self.len >= self.capacity {
            let new_capacity = self.capacity + GROW_SLOTS;
            self.mmap = None; // mmap must be dropped before set_len on Windows
            self.ensure_open()?;
            self.file.as_ref().unwrap().set_len((new_capacity * self.stride) as u64)?;
            self.capacity = new_capacity;
            self.remap()?;
        }
        let slot = self.len;
        self.len += 1;
        Ok(slot)
    }

    pub fn truncate_to(
        &mut self,
        new_len: usize,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.mmap = None; // mmap must be dropped before set_len on Windows
        self.ensure_open()?;
        self.file.as_ref().unwrap().set_len((new_len * self.stride) as u64)?;
        self.capacity = new_len;
        self.len = new_len;
        self.remap()?;
        Ok(())
    }

    pub fn flush(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(mmap) = &self.mmap {
            mmap.flush()?;
        }
        Ok(())
    }

    /// Release both the mmap and the OS file handle.
    ///
    /// Required on Windows before any `fs::rename` or overwrite targeting
    /// this file.  The handle is reopened automatically on the next access.
    pub fn release_handles(&mut self) {
        self.mmap = None;
        self.file = None;
    }

    /// Release the mmap only (kept for callers that don't need full handle release).
    pub fn release_mmap(&mut self) {
        self.mmap = None;
    }

    /// Hint the OS that this mmap will be accessed randomly (not sequentially).
    /// Disables read-ahead on Linux, reducing page-cache pressure for sparse ANN lookups.
    /// No-op on platforms that don't support madvise.
    #[allow(unused_variables)]
    pub fn advise_random(&self) {
        #[cfg(unix)]
        if let Some(mmap) = &self.mmap {
            let _ = mmap.advise(memmap2::Advice::Random);
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn byte_len(&self) -> usize {
        self.len * self.stride
    }

    /// Returns a read-only view of the entire mmap, safe to share across threads.
    pub fn as_bytes(&self) -> &[u8] {
        match &self.mmap {
            Some(m) => &m[..],
            None => &[],
        }
    }

    pub fn clear(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.mmap = None; // mmap must be dropped before set_len on Windows
        self.ensure_open()?;
        self.file.as_ref().unwrap().set_len(0)?;
        self.len = 0;
        self.capacity = 0;
        Ok(())
    }
}
