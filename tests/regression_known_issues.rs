use ndarray::Array1;
use serde_json::json;
use std::collections::HashMap;
use tempfile::tempdir;
use tqdb::storage::engine::TurboQuantEngine;

fn make_vec(d: usize, val: f64) -> Array1<f64> {
    Array1::from_elem(d, val)
}

fn no_meta() -> HashMap<String, serde_json::Value> {
    HashMap::new()
}

#[test]
fn delete_then_reinsert_persists_after_reopen() {
    // Repro mirrors Python behavior:
    // insert/upsert same ID, delete it, then reinsert same ID before close.
    // After reopen, the ID should still exist.
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;

    let mut e = TurboQuantEngine::open(p, p, d, 4, 42).unwrap();
    let mut first_meta = no_meta();
    first_meta.insert("phase".into(), json!(1));
    e.upsert_with_document(
        "x".into(),
        &make_vec(d, 1.0),
        first_meta,
        Some("first".into()),
    )
    .unwrap();
    assert_eq!(e.stats().vector_count, 1);

    let deleted = e.delete("x".into()).unwrap();
    assert!(deleted);
    assert_eq!(e.stats().vector_count, 0);

    let mut second_meta = no_meta();
    second_meta.insert("phase".into(), json!(2));
    e.upsert_with_document(
        "x".into(),
        &make_vec(d, 2.0),
        second_meta,
        Some("second".into()),
    )
    .unwrap();
    assert_eq!(e.stats().vector_count, 1);
    assert!(e.get("x").unwrap().is_some());

    e.close().unwrap();

    let reopened = TurboQuantEngine::open(p, p, d, 4, 42).unwrap();
    assert_eq!(
        reopened.stats().vector_count,
        1,
        "reinserted id should persist after reopen",
    );
    assert!(
        reopened.get("x").unwrap().is_some(),
        "reinserted id missing after reopen",
    );
    let got = reopened.get("x").unwrap().unwrap();
    assert_eq!(got.metadata.get("phase"), Some(&json!(2)));
    assert_eq!(got.document.as_deref(), Some("second"));
}

// ---------------------------------------------------------------------------
// Issue #102 — close() must release mappings; a truncated live slab must be
// reported as an error instead of panicking on the first query.
// ---------------------------------------------------------------------------

#[test]
fn close_releases_live_codes_mapping() {
    // After close() the mapping must be gone so the file can be truncated or
    // replaced. On Windows a live mapping section fails this with os error 1224.
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;

    let mut e = TurboQuantEngine::open(p, p, d, 4, 42).unwrap();
    e.insert("a".into(), &make_vec(d, 1.0), no_meta()).unwrap();
    e.close().unwrap();

    let codes = dir.path().join("live_codes.bin");
    std::fs::File::create(&codes).expect("live_codes.bin must be truncatable after close()");
}

#[test]
fn close_is_idempotent() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;

    let mut e = TurboQuantEngine::open(p, p, d, 4, 42).unwrap();
    e.insert("a".into(), &make_vec(d, 1.0), no_meta()).unwrap();
    e.close().unwrap();
    assert!(e.is_closed());
    e.close().expect("second close() is a no-op");
}

#[test]
fn search_after_close_errors_instead_of_panicking() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;

    let mut e = TurboQuantEngine::open(p, p, d, 4, 42).unwrap();
    e.insert("a".into(), &make_vec(d, 1.0), no_meta()).unwrap();
    e.close().unwrap();

    let err = e
        .search(&make_vec(d, 1.0), 1)
        .expect_err("search on a closed database must return an error");
    assert!(
        err.to_string().contains("closed"),
        "unexpected error: {err}"
    );
}

#[test]
fn open_reports_truncated_live_codes_instead_of_panicking_later() {
    // An interrupted write can leave live_codes.bin empty while live_ids.bin
    // still references its slots. That used to open fine and panic on the first
    // search ("range end index N out of range for slice of length 0"), which
    // PyO3 surfaces as an uncatchable PanicException.
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;

    let mut e = TurboQuantEngine::open(p, p, d, 4, 42).unwrap();
    e.insert("a".into(), &make_vec(d, 1.0), no_meta()).unwrap();
    e.close().unwrap();

    std::fs::write(dir.path().join("live_codes.bin"), b"").unwrap();

    let msg = match TurboQuantEngine::open(p, p, d, 4, 42) {
        Ok(_) => panic!("opening a store with a truncated live_codes.bin must fail"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("corrupt store"), "unexpected error: {msg}");
    assert!(msg.contains("live_codes.bin"), "unexpected error: {msg}");
}

// ---------------------------------------------------------------------------
// Issue #110 — a database written by tqdb <= 0.8.3 must say so, not report EOF.
//
// The fixture in tests/fixtures/legacy_0_8_3_store was produced by an actual
// v0.8.3 build (tag v0.8.3, d=8, dense rotation stored as f32). 0.8.4 changed
// that matrix to bf16, so this build cannot decode it — the question is only
// whether the failure is diagnosable.
// ---------------------------------------------------------------------------

fn copy_fixture(name: &str, dst: &std::path::Path) {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(&src).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), dst.join(entry.file_name())).unwrap();
    }
}

#[test]
fn pre_0_8_4_store_reports_the_format_change_not_an_eof() {
    let dir = tempdir().unwrap();
    let store = dir.path().join("legacy");
    copy_fixture("legacy_0_8_3_store", &store);
    let p = store.to_str().unwrap();

    let msg = match TurboQuantEngine::open(p, p, 8, 4, 42) {
        Ok(_) => panic!("a pre-0.8.4 dense store cannot be decoded by this build"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("quantizer.bin"), "must name the file: {msg}");
    assert!(msg.contains("0.8.4"), "must name the release that changed it: {msg}");
    assert!(
        msg.contains("Regenerate"),
        "must say what to do about it: {msg}"
    );
    assert!(
        msg.contains("damaged"),
        "must acknowledge the corruption case too: {msg}"
    );
}

#[test]
fn current_quantizer_state_round_trips_with_its_header() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;

    let mut e = TurboQuantEngine::open(p, p, d, 4, 42).unwrap();
    e.insert("a".into(), &make_vec(d, 1.0), no_meta()).unwrap();
    e.close().unwrap();

    // The file written by this build carries the magic + version header ...
    let bytes = std::fs::read(dir.path().join("quantizer.bin")).unwrap();
    assert_eq!(&bytes[..4], b"TQQZ", "new files must carry the magic");
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 1);

    // ... and reopening reads it back.
    let reopened = TurboQuantEngine::open(p, p, d, 4, 42).unwrap();
    assert_eq!(reopened.stats().vector_count, 1);
}

#[test]
fn quantizer_state_from_a_newer_format_says_to_upgrade() {
    let dir = tempdir().unwrap();
    let p = dir.path().to_str().unwrap();
    let d = 16;

    let mut e = TurboQuantEngine::open(p, p, d, 4, 42).unwrap();
    e.insert("a".into(), &make_vec(d, 1.0), no_meta()).unwrap();
    e.close().unwrap();

    // Forge a file from a hypothetical future version.
    let path = dir.path().join("quantizer.bin");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();

    let msg = match TurboQuantEngine::open(p, p, d, 4, 42) {
        Ok(_) => panic!("a newer quantizer format must not be silently accepted"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("v99"), "must name the format it found: {msg}");
    assert!(msg.contains("upgrade tqdb"), "must say what to do: {msg}");
}
