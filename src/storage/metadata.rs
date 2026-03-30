use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Cursor, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct VectorMetadata {
    pub properties: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub document: Option<String>,
}

pub struct MetadataStore {
    path: PathBuf,
    data: HashMap<u32, VectorMetadata>,
    dirty: bool,
}

impl MetadataStore {
    pub fn open(path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let path = PathBuf::from(path);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let data = if path.exists() {
            Self::load_from_file(&path)?
        } else {
            HashMap::new()
        };
        Ok(Self {
            path,
            data,
            dirty: false,
        })
    }

    pub fn put(
        &mut self,
        slot: u32,
        meta: &VectorMetadata,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.data.insert(slot, meta.clone());
        self.dirty = true;
        Ok(())
    }

    pub fn put_many(
        &mut self,
        entries: &[(u32, VectorMetadata)],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if entries.is_empty() {
            return Ok(());
        }
        for (slot, meta) in entries {
            self.data.insert(*slot, meta.clone());
        }
        self.dirty = true;
        Ok(())
    }

    pub fn get(
        &self,
        slot: u32,
    ) -> Result<Option<VectorMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(self.data.get(&slot).cloned())
    }

    pub fn get_many(
        &self,
        slots: &[u32],
    ) -> Result<HashMap<u32, VectorMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        let mut out = HashMap::with_capacity(slots.len());
        for slot in slots {
            if let Some(meta) = self.data.get(slot) {
                out.insert(*slot, meta.clone());
            }
        }
        Ok(out)
    }

    pub fn delete(&mut self, slot: u32) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.data.remove(&slot);
        self.dirty = true;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn approx_bytes(&self) -> usize {
        self.data
            .iter()
            .map(|(_slot, meta)| {
                let payload = serde_json::to_vec(meta).map(|v| v.len()).unwrap_or(0);
                4 + payload
            })
            .sum()
    }

    pub fn flush(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !self.dirty {
            return Ok(());
        }

        let tmp = self.path.with_extension("tmp");
        let mut writer = BufWriter::new(File::create(&tmp)?);
        writer.write_all(b"M2S1")?;
        writer.write_all(&(self.data.len() as u64).to_le_bytes())?;

        for (slot, meta) in &self.data {
            let meta_bytes = serde_json::to_vec(meta)?;
            writer.write_all(&slot.to_le_bytes())?;
            writer.write_all(&(meta_bytes.len() as u32).to_le_bytes())?;
            writer.write_all(&meta_bytes)?;
        }

        writer.flush()?;
        std::fs::rename(&tmp, &self.path)?;
        self.dirty = false;
        Ok(())
    }

    fn load_from_file(
        path: &Path,
    ) -> Result<HashMap<u32, VectorMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        let bytes = std::fs::read(path)?;
        let mut cur = Cursor::new(bytes);

        let mut magic = [0u8; 4];
        cur.read_exact(&mut magic)?;
        if &magic != b"M2S1" {
            return Ok(HashMap::new());
        }

        let mut count_buf = [0u8; 8];
        cur.read_exact(&mut count_buf)?;
        let count = u64::from_le_bytes(count_buf) as usize;

        let mut map = HashMap::with_capacity(count);
        for _ in 0..count {
            let mut slot_buf = [0u8; 4];
            cur.read_exact(&mut slot_buf)?;
            let slot = u32::from_le_bytes(slot_buf);

            let mut meta_len_buf = [0u8; 4];
            cur.read_exact(&mut meta_len_buf)?;
            let meta_len = u32::from_le_bytes(meta_len_buf) as usize;
            let mut meta_bytes = vec![0u8; meta_len];
            cur.read_exact(&mut meta_bytes)?;
            let meta: VectorMetadata = serde_json::from_slice(&meta_bytes)?;
            map.insert(slot, meta);
        }

        Ok(map)
    }
}
