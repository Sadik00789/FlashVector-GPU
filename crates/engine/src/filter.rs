//! Device-resident metadata filtering via Roaring-style compressed bitsets.

use std::collections::{BTreeMap, HashMap};

const ARRAY_THRESHOLD: usize = 4096;
const BITMAP_WORDS: usize = 1024; // 65,536 bits = 1024 * 64-bit words

#[derive(Debug, Clone, PartialEq, Eq)]
enum Container {
    Array(Vec<u16>),
    Bitmap(Box<[u64; BITMAP_WORDS]>),
}

impl Container {
    fn len(&self) -> usize {
        match self {
            Container::Array(vals) => vals.len(),
            Container::Bitmap(words) => words.iter().map(|w| w.count_ones() as usize).sum(),
        }
    }

    fn contains(&self, low: u16) -> bool {
        match self {
            Container::Array(vals) => vals.binary_search(&low).is_ok(),
            Container::Bitmap(words) => {
                let word = (low >> 6) as usize;
                let bit = low & 63;
                (words[word] & (1u64 << bit)) != 0
            }
        }
    }

    fn insert(&mut self, low: u16) -> bool {
        match self {
            Container::Array(vals) => match vals.binary_search(&low) {
                Ok(_) => false,
                Err(pos) => {
                    vals.insert(pos, low);
                    if vals.len() >= ARRAY_THRESHOLD {
                        let mut words = Box::new([0u64; BITMAP_WORDS]);
                        for &v in vals.iter() {
                            words[(v >> 6) as usize] |= 1u64 << (v & 63);
                        }
                        *self = Container::Bitmap(words);
                    }
                    true
                }
            },
            Container::Bitmap(words) => {
                let word = (low >> 6) as usize;
                let mask = 1u64 << (low & 63);
                let prev = (words[word] & mask) != 0;
                words[word] |= mask;
                !prev
            }
        }
    }

    fn remove(&mut self, low: u16) -> bool {
        match self {
            Container::Array(vals) => match vals.binary_search(&low) {
                Ok(pos) => {
                    vals.remove(pos);
                    true
                }
                Err(_) => false,
            },
            Container::Bitmap(words) => {
                let word = (low >> 6) as usize;
                let mask = 1u64 << (low & 63);
                let prev = (words[word] & mask) != 0;
                words[word] &= !mask;
                let pop: usize = words.iter().map(|w| w.count_ones() as usize).sum();
                if prev && pop < ARRAY_THRESHOLD {
                    let mut arr = Vec::with_capacity(ARRAY_THRESHOLD);
                    for (wi, &w) in words.iter().enumerate() {
                        let mut bits = w;
                        while bits != 0 {
                            let tz = bits.trailing_zeros() as u16;
                            arr.push(((wi as u16) << 6) | tz);
                            bits &= bits - 1;
                        }
                    }
                    *self = Container::Array(arr);
                }
                prev
            }
        }
    }

    fn and(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Container::Array(a), Container::Array(b)) => {
                let mut out = Vec::new();
                let (mut i, mut j) = (0, 0);
                while i < a.len() && j < b.len() {
                    match a[i].cmp(&b[j]) {
                        std::cmp::Ordering::Less => i += 1,
                        std::cmp::Ordering::Greater => j += 1,
                        std::cmp::Ordering::Equal => {
                            out.push(a[i]);
                            i += 1;
                            j += 1;
                        }
                    }
                }
                if out.is_empty() {
                    None
                } else {
                    Some(Container::Array(out))
                }
            }
            (Container::Array(a), Container::Bitmap(b))
            | (Container::Bitmap(b), Container::Array(a)) => {
                let out: Vec<u16> = a
                    .iter()
                    .copied()
                    .filter(|&v| (b[(v >> 6) as usize] & (1u64 << (v & 63))) != 0)
                    .collect();
                if out.is_empty() {
                    None
                } else {
                    Some(Container::Array(out))
                }
            }
            (Container::Bitmap(a), Container::Bitmap(b)) => {
                let mut words = Box::new([0u64; BITMAP_WORDS]);
                let mut pop = 0usize;
                for w in 0..BITMAP_WORDS {
                    let v = a[w] & b[w];
                    words[w] = v;
                    pop += v.count_ones() as usize;
                }
                if pop == 0 {
                    None
                } else if pop < ARRAY_THRESHOLD {
                    Some(Self::bitmap_to_array(&words, pop))
                } else {
                    Some(Container::Bitmap(words))
                }
            }
        }
    }

    fn or(&self, other: &Self) -> Self {
        match (self, other) {
            (Container::Array(a), Container::Array(b)) => {
                let mut out = Vec::with_capacity(a.len() + b.len());
                let (mut i, mut j) = (0, 0);
                while i < a.len() && j < b.len() {
                    match a[i].cmp(&b[j]) {
                        std::cmp::Ordering::Less => {
                            out.push(a[i]);
                            i += 1;
                        }
                        std::cmp::Ordering::Greater => {
                            out.push(b[j]);
                            j += 1;
                        }
                        std::cmp::Ordering::Equal => {
                            out.push(a[i]);
                            i += 1;
                            j += 1;
                        }
                    }
                }
                out.extend_from_slice(&a[i..]);
                out.extend_from_slice(&b[j..]);
                if out.len() >= ARRAY_THRESHOLD {
                    let mut words = Box::new([0u64; BITMAP_WORDS]);
                    for v in out {
                        words[(v >> 6) as usize] |= 1u64 << (v & 63);
                    }
                    Container::Bitmap(words)
                } else {
                    Container::Array(out)
                }
            }
            (Container::Array(a), Container::Bitmap(b))
            | (Container::Bitmap(b), Container::Array(a)) => {
                let mut words = b.clone();
                for &v in a {
                    words[(v >> 6) as usize] |= 1u64 << (v & 63);
                }
                Container::Bitmap(words)
            }
            (Container::Bitmap(a), Container::Bitmap(b)) => {
                let mut words = Box::new([0u64; BITMAP_WORDS]);
                for w in 0..BITMAP_WORDS {
                    words[w] = a[w] | b[w];
                }
                Container::Bitmap(words)
            }
        }
    }

    fn bitmap_to_array(words: &[u64; BITMAP_WORDS], cap: usize) -> Self {
        let mut arr = Vec::with_capacity(cap);
        for (wi, &w) in words.iter().enumerate() {
            let mut bits = w;
            while bits != 0 {
                let tz = bits.trailing_zeros() as u16;
                arr.push(((wi as u16) << 6) | tz);
                bits &= bits - 1;
            }
        }
        Container::Array(arr)
    }

    fn write_packed_u32(&self, high: u16, out: &mut [u32]) {
        let base_bit = (high as usize) << 16;
        let max_bits = out.len() * 32;
        if base_bit >= max_bits {
            return;
        }
        match self {
            Container::Array(vals) => {
                for &low in vals {
                    let id = base_bit + low as usize;
                    if id < max_bits {
                        out[id >> 5] |= 1u32 << (id & 31);
                    }
                }
            }
            Container::Bitmap(words) => {
                let base_word = base_bit >> 5;
                for (wi, &w64) in words.iter().enumerate() {
                    let lo_idx = base_word + wi * 2;
                    if lo_idx < out.len() {
                        out[lo_idx] |= (w64 & 0xFFFF_FFFF) as u32;
                    }
                    let hi_idx = lo_idx + 1;
                    if hi_idx < out.len() {
                        out[hi_idx] |= (w64 >> 32) as u32;
                    }
                }
            }
        }
    }
}

/// Compressed 32-bit Roaring-style bitmap supporting set operations and
/// direct packing into GPU `u32` bitmasks.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RoaringBitset {
    chunks: BTreeMap<u16, Container>,
}

impl RoaringBitset {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_range(len: usize) -> Self {
        let mut bm = Self::new();
        for i in 0..len {
            bm.insert(i as u32);
        }
        bm
    }

    pub fn from_bools(mask: &[bool]) -> Self {
        let mut bm = Self::new();
        for (i, &allow) in mask.iter().enumerate() {
            if allow {
                bm.insert(i as u32);
            }
        }
        bm
    }

    pub fn insert(&mut self, id: u32) -> bool {
        let high = (id >> 16) as u16;
        let low = (id & 0xFFFF) as u16;
        match self.chunks.get_mut(&high) {
            Some(c) => c.insert(low),
            None => {
                self.chunks.insert(high, Container::Array(vec![low]));
                true
            }
        }
    }

    pub fn remove(&mut self, id: u32) -> bool {
        let high = (id >> 16) as u16;
        let low = (id & 0xFFFF) as u16;
        if let Some(c) = self.chunks.get_mut(&high) {
            let removed = c.remove(low);
            if c.len() == 0 {
                self.chunks.remove(&high);
            }
            removed
        } else {
            false
        }
    }

    pub fn contains(&self, id: u32) -> bool {
        let high = (id >> 16) as u16;
        let low = (id & 0xFFFF) as u16;
        self.chunks
            .get(&high)
            .map(|c| c.contains(low))
            .unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.chunks.values().map(Container::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn intersect(&self, other: &Self) -> Self {
        let mut out = BTreeMap::new();
        for (high, c1) in &self.chunks {
            if let Some(c2) = other.chunks.get(high) {
                if let Some(res) = c1.and(c2) {
                    out.insert(*high, res);
                }
            }
        }
        Self { chunks: out }
    }

    pub fn union(&self, other: &Self) -> Self {
        let mut out = self.chunks.clone();
        for (high, c2) in &other.chunks {
            out.entry(*high)
                .and_modify(|c1| *c1 = c1.or(c2))
                .or_insert_with(|| c2.clone());
        }
        Self { chunks: out }
    }

    /// Materialize all set IDs in ascending order.
    pub fn to_vec(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.len());
        for (&high, container) in &self.chunks {
            let base = (high as u32) << 16;
            match container {
                Container::Array(vals) => {
                    for &low in vals {
                        out.push(base | low as u32);
                    }
                }
                Container::Bitmap(words) => {
                    for (wi, &w) in words.iter().enumerate() {
                        let mut bits = w;
                        while bits != 0 {
                            let tz = bits.trailing_zeros();
                            out.push(base | ((wi as u32) << 6) | tz);
                            bits &= bits - 1;
                        }
                    }
                }
            }
        }
        out
    }

    /// Pack into `[ceil(num_bits / 32)]` `u32` words for GPU device bitmasks.
    pub fn to_packed_u32_words(&self, num_bits: usize) -> Vec<u32> {
        let num_words = num_bits.div_ceil(32);
        let mut words = vec![0u32; num_words];
        for (&high, container) in &self.chunks {
            container.write_packed_u32(high, &mut words);
        }
        words
    }
}

/// Structured metadata record for RAG filtering (`tenant_id`, `timestamp`, ACL tags).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct VectorMetadata {
    pub tenant_id: Option<String>,
    pub timestamp: Option<i64>,
    pub acl_tags: Vec<String>,
}

/// Filter query compiled into a `RoaringBitset` / GPU bitmask.
#[derive(Debug, Clone, Default)]
pub struct MetadataFilter {
    pub tenant_id: Option<String>,
    pub min_timestamp: Option<i64>,
    pub max_timestamp: Option<i64>,
    pub any_acl_tags: Vec<String>,
}

/// Inverted bitmap index over vector metadata for fast device-mask construction.
#[derive(Debug, Clone, Default)]
pub struct MetadataIndex {
    num_docs: usize,
    by_tenant: HashMap<String, RoaringBitset>,
    by_acl: HashMap<String, RoaringBitset>,
    timestamps: Vec<Option<i64>>,
}

impl MetadataIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, id: u32, meta: &VectorMetadata) {
        let idx = id as usize;
        if idx >= self.num_docs {
            self.num_docs = idx + 1;
            self.timestamps.resize(self.num_docs, None);
        }
        self.timestamps[idx] = meta.timestamp;
        if let Some(ref tenant) = meta.tenant_id {
            self.by_tenant.entry(tenant.clone()).or_default().insert(id);
        }
        for tag in &meta.acl_tags {
            self.by_acl.entry(tag.clone()).or_default().insert(id);
        }
    }

    pub fn evaluate(&self, filter: &MetadataFilter) -> RoaringBitset {
        let mut result: Option<RoaringBitset> = None;

        if let Some(ref tenant) = filter.tenant_id {
            let b = self.by_tenant.get(tenant).cloned().unwrap_or_default();
            result = Some(b);
        }

        if !filter.any_acl_tags.is_empty() {
            let mut acl_union = RoaringBitset::new();
            for tag in &filter.any_acl_tags {
                if let Some(b) = self.by_acl.get(tag) {
                    acl_union = acl_union.union(b);
                }
            }
            result = Some(match result {
                Some(cur) => cur.intersect(&acl_union),
                None => acl_union,
            });
        }

        if filter.min_timestamp.is_some() || filter.max_timestamp.is_some() {
            let mut ts_set = RoaringBitset::new();
            match result {
                Some(ref cur) => {
                    for (i, ts_opt) in self.timestamps.iter().enumerate() {
                        let id = i as u32;
                        if cur.contains(id) && Self::matches_ts(*ts_opt, filter) {
                            ts_set.insert(id);
                        }
                    }
                    result = Some(ts_set);
                }
                None => {
                    for (i, ts_opt) in self.timestamps.iter().enumerate() {
                        if Self::matches_ts(*ts_opt, filter) {
                            ts_set.insert(i as u32);
                        }
                    }
                    result = Some(ts_set);
                }
            }
        }

        result.unwrap_or_else(|| RoaringBitset::from_range(self.num_docs))
    }

    #[inline]
    fn matches_ts(ts_opt: Option<i64>, filter: &MetadataFilter) -> bool {
        let Some(ts) = ts_opt else {
            return false;
        };
        if let Some(min_ts) = filter.min_timestamp {
            if ts < min_ts {
                return false;
            }
        }
        if let Some(max_ts) = filter.max_timestamp {
            if ts > max_ts {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roaring_array_bitmap_and_packed_words() {
        let mut a = RoaringBitset::new();
        for i in (0..5000u32).step_by(2) {
            a.insert(i);
        }
        a.insert(70_000);
        assert!(a.contains(4096));
        assert!(!a.contains(4097));
        assert!(a.contains(70_000));

        let mut b = RoaringBitset::new();
        b.insert(4096);
        b.insert(70_000);
        b.insert(1);

        let inter = a.intersect(&b);
        assert_eq!(inter.len(), 2);
        assert!(inter.contains(4096));
        assert!(inter.contains(70_000));

        let words = inter.to_packed_u32_words(70_005);
        assert_ne!(words[4096 / 32] & (1u32 << (4096 & 31)), 0);
        assert_ne!(words[70_000 / 32] & (1u32 << (70_000 & 31)), 0);
        assert_eq!(words[0], 0);
    }

    #[test]
    fn test_metadata_index_tenant_acl_timestamp() {
        let mut idx = MetadataIndex::new();
        idx.set(
            0,
            &VectorMetadata {
                tenant_id: Some("acme".into()),
                timestamp: Some(100),
                acl_tags: vec!["eng".into(), "read".into()],
            },
        );
        idx.set(
            1,
            &VectorMetadata {
                tenant_id: Some("acme".into()),
                timestamp: Some(200),
                acl_tags: vec!["sales".into()],
            },
        );
        idx.set(
            2,
            &VectorMetadata {
                tenant_id: Some("beta".into()),
                timestamp: Some(200),
                acl_tags: vec!["eng".into()],
            },
        );

        let bm = idx.evaluate(&MetadataFilter {
            tenant_id: Some("acme".into()),
            min_timestamp: Some(150),
            max_timestamp: None,
            any_acl_tags: vec!["sales".into(), "admin".into()],
        });
        assert_eq!(bm.len(), 1);
        assert!(bm.contains(1));
    }
}
