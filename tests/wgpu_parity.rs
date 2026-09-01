//! GPU-vs-CPU bit-exact parity on the golden fixture.
//!
//! GPU work on this host must run under `training.slice` (workspace
//! GPU-compute policy); run this test via:
//!   systemd-run --user --slice=training.slice --wait --pipe \
//!     bash -c 'cd dally-eval && nix develop --command cargo test --test wgpu_parity'
//! Skips gracefully when no adapter is available.

use std::fs;

use dally_eval::ir::Program;
use dally_eval::runner::BatchRunner;
use dally_eval::{CpuRunner, CubeRunner};

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn cubecl_matches_cpu_bit_exact() {
    let text = fs::read_to_string(fixture("siswalk1_cap2.ir")).unwrap();
    let prog = Program::parse(&text).unwrap();
    let bin = fs::read(fixture("parity32.bin")).unwrap();
    let mut off = 0usize;
    let n = u32::from_le_bytes(bin[off..off + 4].try_into().unwrap()) as usize;
    off += 4;
    let width = u32::from_le_bytes(bin[off..off + 4].try_into().unwrap()) as usize;
    off += 8;
    let inputs = bin[off..off + n * width].to_vec();

    let cpu = CpuRunner.run_batch(&prog, &inputs, n).unwrap();
    let runner = match CubeRunner::new(&prog, n) {
        Ok(r) => r,
        Err(dally_eval::RunError::GpuUnavailable(msg)) => {
            eprintln!("SKIP: no GPU available: {msg}");
            return;
        }
        Err(e) => panic!("{e:?}"),
    };
    let gpu = match runner.run(&prog, &inputs, n) {
        Ok(g) => g,
        Err((_, dally_eval::RunError::GpuUnavailable(msg))) => {
            eprintln!("SKIP: no GPU available: {msg}");
            return;
        }
        Err((i, e)) => panic!("instance {i}: {e:?}"),
    };
    assert_eq!(cpu.len(), gpu.len());
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert_eq!(c, g, "instance {i} differs between CPU and CubeCL");
    }
}
