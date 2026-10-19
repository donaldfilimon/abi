//! CPU-only archive compatibility checks for the patched Candle dependency.

use candle_core::{Device, Error, Tensor, npy::NpzTensors};
use std::path::PathBuf;

struct Archive(PathBuf);

impl Archive {
    fn new() -> Self {
        let path = abi_foundation::temp_path::temp_file_path("abi_archive_compat", "npz");
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .expect("create unique scratch archive");
        Self(path)
    }
}

impl Drop for Archive {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn zip_upgrade_preserves_utf8_and_cp437_tensor_names() {
    for (name, expected, cp437) in [("données", "données", false), ("a", "é", true)] {
        let archive = Archive::new();
        let tensor = Tensor::new(&[2_u32, 4, 6], &Device::Cpu).expect("CPU tensor");
        Tensor::write_npz(&[(name, &tensor)], &archive.0).expect("write scratch tensor");
        if cp437 {
            let mut bytes = std::fs::read(&archive.0).expect("read own archive");
            let positions: Vec<_> = bytes
                .windows(5)
                .enumerate()
                .filter_map(|(index, bytes)| (bytes == b"a.npy").then_some(index))
                .collect();
            assert_eq!(positions.len(), 2, "local and central filename records");
            for index in positions {
                bytes[index] = 0x82;
            }
            std::fs::write(&archive.0, bytes).expect("write CP437 filename fixture");
        }

        let eager = Tensor::read_npz(&archive.0).expect("read NPZ tensor");
        assert_eq!(eager.len(), 1);
        assert_eq!(eager[0].0, expected);
        assert_eq!(
            eager[0].1.to_vec1::<u32>().expect("eager values"),
            [2, 4, 6]
        );
        let lazy = NpzTensors::new(&archive.0).expect("index NPZ tensor");
        assert_eq!(lazy.names().len(), 1);
        let loaded = lazy
            .get(expected)
            .expect("load by decoded name")
            .expect("tensor exists");
        assert_eq!(loaded.to_vec1::<u32>().expect("lazy values"), [2, 4, 6]);
    }
}

#[test]
fn malformed_archives_remain_errors_in_eager_lazy_and_pth_readers() {
    let archive = Archive::new();
    std::fs::write(&archive.0, b"not a zip archive").expect("write malformed fixture");
    assert!(matches!(Tensor::read_npz(&archive.0), Err(Error::Zip(_))));
    assert!(matches!(NpzTensors::new(&archive.0), Err(Error::Zip(_))));
    assert!(matches!(
        candle_core::pickle::read_pth_tensor_info(&archive.0, false, None),
        Err(Error::Zip(_))
    ));
}
