use candle_core::{npy::NpzTensors, pickle::PthTensors, DType, Device, Error, Result, Tensor};
use std::fs::File;
use std::path::{Path, PathBuf};

fn upstream(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(name)
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/zip-fixtures")
        .join(name)
}

fn assert_npy_values(tensor: &Tensor) -> Result<()> {
    assert_eq!(
        tensor.to_dtype(DType::U8)?.to_vec1::<u8>()?,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    );
    Ok(())
}

fn assert_zip_error<T>(result: Result<T>) {
    assert!(matches!(result, Err(Error::Zip(_))));
}

#[test]
fn upstream_npy() -> Result<()> {
    assert_npy_values(&Tensor::read_npy(upstream("test.npy"))?)
}

#[test]
fn upstream_npz_eager() -> Result<()> {
    let tensors = Tensor::read_npz(upstream("test.npz"))?;
    assert_eq!(tensors.len(), 2);
    assert_eq!(tensors[0].0, "x");
    assert_eq!(tensors[1].0, "x_plus_one");
    assert_npy_values(&tensors[0].1)?;
    assert_eq!(
        tensors[1].1.to_dtype(DType::U8)?.to_vec1::<u8>()?,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    );
    Ok(())
}

#[test]
fn upstream_npz_lazy() -> Result<()> {
    let tensors = NpzTensors::new(upstream("test.npz"))?;
    let mut names = tensors
        .names()
        .into_iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    names.sort_unstable();
    assert_eq!(names, ["x", "x_plus_one"]);
    assert_npy_values(&tensors.get("x")?.unwrap())?;
    let (shape, dtype) = tensors.get_shape_and_dtype("x")?;
    assert_eq!(shape.dims(), [10]);
    assert_eq!(dtype, DType::I64);
    assert!(tensors.get("missing")?.is_none());
    Ok(())
}

#[test]
fn npz_by_name_preserves_order_and_missing_error() -> Result<()> {
    let tensors = Tensor::read_npz_by_name(upstream("test.npz"), &["x_plus_one", "x"])?;
    assert_eq!(tensors[0].to_dtype(DType::U8)?.to_vec1::<u8>()?[0], 1);
    assert_npy_values(&tensors[1])?;
    assert!(matches!(
        Tensor::read_npz_by_name(upstream("test.npz"), &["missing"]),
        Err(Error::Npy(_))
    ));
    Ok(())
}

#[test]
fn unicode_npz_eager_lazy_and_by_name() -> Result<()> {
    let path = fixture("unicode.npz");
    let tensors = Tensor::read_npz(&path)?;
    assert_eq!(tensors[0].0, "café");
    assert_eq!(tensors[1].0, "日本語");
    let lazy = NpzTensors::new(&path)?;
    for name in ["café", "日本語"] {
        assert_npy_values(&lazy.get(name)?.unwrap())?;
        assert_npy_values(&Tensor::read_npz_by_name(&path, &[name])?[0])?;
    }
    Ok(())
}

#[test]
fn npz_suffix_and_nested_member_names() -> Result<()> {
    let path = fixture("suffix.npz");
    let tensors = Tensor::read_npz(&path)?;
    assert_eq!(tensors[0].0, "plain");
    assert_eq!(tensors[1].0, "inner/name");
    let lazy = NpzTensors::new(&path)?;
    assert_npy_values(&lazy.get("plain")?.unwrap())?;
    assert_npy_values(&lazy.get("inner/name")?.unwrap())?;
    Ok(())
}

#[test]
fn non_utf8_npz_name_uses_zip9_cp437_fallback() -> Result<()> {
    let path = fixture("malformed-name.npz");
    let tensors = Tensor::read_npz(&path)?;
    assert_eq!(tensors[0].0, "\u{00a0}ad");
    assert_npy_values(&tensors[0].1)?;
    let lazy = NpzTensors::new(&path)?;
    assert_npy_values(&lazy.get("\u{00a0}ad")?.unwrap())?;
    assert_npy_values(&Tensor::read_npz_by_name(&path, &["\u{00a0}ad"])?[0])?;
    Ok(())
}

#[test]
fn npz_write_read_round_trip() -> Result<()> {
    let path = std::env::temp_dir().join(format!("candle-round-trip-{}.npz", std::process::id()));
    let first = Tensor::new(&[1f32, 2f32, 3f32], &Device::Cpu)?;
    let second = Tensor::new(&[4i64, 5i64], &Device::Cpu)?;
    Tensor::write_npz(&[("café", &first), ("second", &second)], &path)?;
    let tensors = Tensor::read_npz(&path)?;
    assert_eq!(tensors[0].0, "café");
    assert_eq!(tensors[0].1.to_vec1::<f32>()?, [1., 2., 3.]);
    assert_eq!(tensors[1].1.to_vec1::<i64>()?, [4, 5]);
    std::fs::remove_file(path)?;
    Ok(())
}

#[test]
fn upstream_pth_plain_and_nested_state_dict() -> Result<()> {
    for (name, key) in [
        ("test.pt", None),
        ("test_with_key.pt", Some("model_state_dict")),
    ] {
        let tensors = PthTensors::new(upstream(name), key)?;
        let tensor = tensors.get("test")?.unwrap();
        assert_eq!(tensor.elem_count(), 8);
        assert_eq!(
            tensor.flatten_all()?.to_vec1::<i64>()?,
            [1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert!(tensors.get("missing")?.is_none());
        assert_eq!(tensors.tensor_infos().len(), 1);
    }
    Ok(())
}

#[test]
fn upstream_pth_fortran_contiguous() -> Result<()> {
    let tensors = PthTensors::new(upstream("fortran_tensor_3d.pth"), None)?;
    let tensor = tensors.get("tensor_fortran")?.unwrap();
    assert_eq!(tensor.dims3()?, (2, 3, 4));
    assert_eq!(
        tensor.to_vec3::<i64>()?,
        [
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ]
    );
    Ok(())
}

#[test]
fn unicode_pth_archive_prefix() -> Result<()> {
    let original = PthTensors::new(upstream("test.pt"), None)?
        .get("test")?
        .unwrap();
    let unicode = PthTensors::new(fixture("unicode.pt"), None)?
        .get("test")?
        .unwrap();
    assert_eq!(original.dims(), unicode.dims());
    assert_eq!(
        original.flatten_all()?.to_vec1::<i64>()?,
        unicode.flatten_all()?.to_vec1::<i64>()?
    );
    Ok(())
}

#[test]
fn non_utf8_pth_member_uses_zip9_cp437_fallback() -> Result<()> {
    let tensors = PthTensors::new(fixture("malformed-name.pt"), None)?;
    assert_eq!(
        tensors
            .get("test")?
            .unwrap()
            .flatten_all()?
            .to_vec1::<i64>()?,
        [1, 2, 3, 4, 5, 6, 7, 8]
    );
    Ok(())
}

#[test]
fn zip_filename_iterator_and_payload() -> Result<()> {
    let path = fixture("unicode.npz");
    let mut archive = zip::ZipArchive::new(File::open(path)?)?;
    let names = archive
        .file_names()
        .map(|name| name.map(|name| name.into_owned()))
        .collect::<zip::result::ZipResult<Vec<_>>>()?;
    assert_eq!(names, ["café.npy", "日本語.npy"]);
    assert_eq!(archive.by_name("café.npy")?.name()?.as_ref(), "café.npy");
    Ok(())
}

#[test]
fn deflate_feature_behavior() -> Result<()> {
    let result = Tensor::read_npz(fixture("deflated.npz"));
    if std::env::var_os("CANDLE_TEST_EXPECT_DEFLATE").is_some() {
        assert_npy_values(&result?[0].1)?;
        return Ok(());
    }
    match result {
        Ok(tensors) => assert_npy_values(&tensors[0].1)?,
        Err(Error::Zip(zip::result::ZipError::CompressionMethodNotSupported(8))) => (),
        Err(error) => return Err(error),
    }
    Ok(())
}

#[test]
fn legacy_cp437_npz_all_readers() -> Result<()> {
    let path = fixture("cp437.npz");
    let tensors = Tensor::read_npz(&path)?;
    assert_eq!(tensors[0].0, "café");
    assert_npy_values(&tensors[0].1)?;
    assert_npy_values(&NpzTensors::new(&path)?.get("café")?.unwrap())?;
    assert_npy_values(&Tensor::read_npz_by_name(&path, &["café"])?[0])?;
    Ok(())
}

#[test]
fn legacy_lookup_scans_metadata_without_opening_unrelated_compressed_members() -> Result<()> {
    assert_npy_values(
        &Tensor::read_npz_by_name(fixture("cp437-after-compressed.npz"), &["café"])?[0],
    )
}

#[test]
fn legacy_cp437_pth_actual_prefix_and_storage_retrieval() -> Result<()> {
    for (name, key) in [
        ("cp437.pt", None),
        ("cp437-with-key.pt", Some("model_state_dict")),
    ] {
        let path = fixture(name);
        let infos = candle_core::pickle::read_pth_tensor_info(&path, false, key)?;
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].path, "café/data/0");
        let tensors = PthTensors::new(path, key)?;
        assert_eq!(
            tensors
                .get("test")?
                .unwrap()
                .flatten_all()?
                .to_vec1::<i64>()?,
            [1, 2, 3, 4, 5, 6, 7, 8]
        );
    }
    Ok(())
}

#[test]
fn pth_invalid_utf8_flag_actual_prefix_retains_zip9_cp437_fallback() -> Result<()> {
    let tensors = PthTensors::new(fixture("malformed-prefix.pt"), None)?;
    assert_eq!(
        tensors
            .get("test")?
            .unwrap()
            .flatten_all()?
            .to_vec1::<i64>()?,
        [1, 2, 3, 4, 5, 6, 7, 8]
    );
    Ok(())
}

#[test]
fn pth_missing_storage_retains_zip_file_not_found() -> Result<()> {
    let tensors = PthTensors::new(fixture("missing-storage.pt"), None)?;
    assert!(matches!(
        tensors.get("test"),
        Err(Error::Zip(zip::result::ZipError::FileNotFound))
    ));
    Ok(())
}

#[test]
fn corrupt_archives_propagate_zip_errors() {
    assert_zip_error(Tensor::read_npz(fixture("corrupt.zip")));
    assert_zip_error(Tensor::read_npz_by_name(fixture("corrupt.zip"), &["test"]));
    assert_zip_error(NpzTensors::new(fixture("corrupt.zip")));
    assert_zip_error(PthTensors::new(fixture("corrupt.zip"), None));
}

#[test]
fn exact_raw_duplicate_names_use_last_payload() -> Result<()> {
    let path = fixture("duplicate.npz");
    let tensors = Tensor::read_npz(&path)?;
    assert_eq!(tensors.len(), 1);
    assert_eq!(tensors[0].0, "x");
    assert_eq!(
        tensors[0].1.to_vec1::<i64>()?,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    );
    assert_eq!(
        NpzTensors::new(&path)?
            .get("x")?
            .unwrap()
            .to_vec1::<i64>()?,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    );
    assert_eq!(
        Tensor::read_npz_by_name(&path, &["x"])?[0].to_vec1::<i64>()?,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    );
    Ok(())
}

#[test]
fn mixed_encoding_collision_uses_last_central_entry_consistently() -> Result<()> {
    for (name, expected) in [
        ("mixed-collision.npz", (1..=10).collect::<Vec<i64>>()),
        (
            "mixed-collision-reversed.npz",
            (0..10).collect::<Vec<i64>>(),
        ),
    ] {
        let path = fixture(name);
        let tensors = Tensor::read_npz(&path)?;
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].0, "café");
        assert_eq!(tensors[0].1.to_vec1::<i64>()?, expected);
        assert_eq!(
            Tensor::read_npz_by_name(&path, &["café"])?[0].to_vec1::<i64>()?,
            expected
        );
        assert_eq!(
            NpzTensors::new(&path)?
                .get("café")?
                .unwrap()
                .to_vec1::<i64>()?,
            expected
        );
    }
    Ok(())
}

#[test]
fn interleaved_raw_duplicates_preserve_last_central_entry() -> Result<()> {
    let path = fixture("interleaved-duplicate.npz");
    assert_eq!(
        Tensor::read_npz_by_name(&path, &["café"])?[0].to_vec1::<i64>()?,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    );
    assert_eq!(
        NpzTensors::new(&path)?
            .get("café")?
            .unwrap()
            .to_vec1::<i64>()?,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    );
    Ok(())
}

#[test]
fn cp437_bytes_that_form_utf8_retain_zip9_naming_difference() -> Result<()> {
    let path = fixture("cp437-valid-utf8-bytes.npz");
    assert_eq!(Tensor::read_npz(&path)?[0].0, "café");
    assert_npy_values(&Tensor::read_npz_by_name(&path, &["café"])?[0])?;
    assert!(matches!(
        Tensor::read_npz_by_name(&path, &["caf├⌐"]),
        Err(Error::Npy(_))
    ));
    Ok(())
}

#[test]
fn suffix_alias_retains_previous_lazy_insertion_behavior() -> Result<()> {
    let path = fixture("suffix-alias.npz");
    let eager = Tensor::read_npz(&path)?;
    assert_eq!(eager.len(), 2);
    assert_npy_values(&eager[0].1)?;
    assert_eq!(
        NpzTensors::new(&path)?
            .get("x")?
            .unwrap()
            .to_vec1::<i64>()?,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    );
    assert_npy_values(&Tensor::read_npz_by_name(&path, &["x"])?[0])?;
    Ok(())
}

#[test]
fn pth_mixed_and_interleaved_aliases_have_one_winning_tensor_info() -> Result<()> {
    for (name, base) in [("mixed-collision.pt", 11), ("interleaved-duplicate.pt", 21)] {
        let path = fixture(name);
        let infos = candle_core::pickle::read_pth_tensor_info(&path, false, None)?;
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].name, "test");
        assert_eq!(infos[0].path, "café/data/0");
        let pth = PthTensors::new(path, None)?;
        assert_eq!(pth.tensor_infos().len(), 1);
        assert_eq!(
            pth.get("test")?.unwrap().flatten_all()?.to_vec1::<i64>()?,
            (base..base + 8).collect::<Vec<_>>()
        );
    }
    Ok(())
}

#[test]
fn selective_npz_preserves_unsupported_compression_error() {
    assert!(matches!(
        Tensor::read_npz_by_name(fixture("unsupported-compression.npz"), &["x"]),
        Err(Error::Zip(
            zip::result::ZipError::CompressionMethodNotSupported(99)
        ))
    ));
}
