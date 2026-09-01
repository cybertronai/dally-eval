//! GPU throughput harness: CubeRunner (CubeCL kernel, static buffer
//! reuse) end-to-end on the real 73k-op benchmark program.
//!
//! Harness-free so it soft-skips without an adapter and uses its own
//! timing loop. Run under the host's resource-management policy.

use std::fs;
use std::time::Instant;

use dally_eval::ir::Program;
use dally_eval::runner::BatchRunner;
use dally_eval::{CubeRunner, LdsRunner, RunError};

fn main() {
    let text = fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/siswalk1_cap2.ir"),
    )
    .expect("fixture present");
    let prog = Program::parse(&text).unwrap();
    let width = prog.inputs.len();

    // probe: build the runner (compiles the kernel) on a 1-instance batch
    let runner = match CubeRunner::new(&prog, 100_000) {
        Ok(r) => r,
        Err(RunError::GpuUnavailable(m)) => {
            eprintln!("SKIP: no GPU adapter ({m})");
            return;
        }
        Err(e) => panic!("{e:?}"),
    };
    match runner.run(&prog, &vec![0u8; width], 1) {
        Ok(_) => {}
        Err((_, RunError::GpuUnavailable(m))) => {
            eprintln!("SKIP: {m}");
            return;
        }
        Err((i, e)) => panic!("probe instance {i}: {e:?}"),
    }

    for &n in &[1_000usize, 10_000, 50_000, 100_000] {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut inputs = vec![0u8; n * width];
        for b in inputs.iter_mut() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *b = (seed >> 33) as u8;
        }
        // correctness spot check vs CPU on a slice
        let gpu = runner.run(&prog, &inputs, n).expect("gpu run");
        let cpu = dally_eval::CpuRunner
            .run_batch(&prog, &inputs, n.min(256))
            .expect("cpu run");
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert_eq!(g, c, "GPU/CPU divergence at instance {i}");
        }

        let iters = if n <= 1_000 {
            50
        } else if n <= 10_000 {
            10
        } else {
            3
        };
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = runner.run(&prog, &inputs, n).unwrap();
        }
        let per = t0.elapsed() / iters;
        println!(
            "raw    {n:>6}: {per:>10.3?}/batch   {:>10.0} instances/s   (ops {})",
            n as f64 / per.as_secs_f64(),
            prog.len()
        );

        // scored mode: answer key uploaded per call, grading on device,
        // one flag word per instance comes back
        let expected_flat: Vec<u8> = gpu.iter().flat_map(|r| r.iter().copied()).collect();
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = runner
                .run_scored(&prog, &inputs, &expected_flat, n)
                .unwrap();
        }
        let per = t0.elapsed() / iters;
        println!(
            "scored {n:>6}: {per:>10.3?}/batch   {:>10.0} instances/s",
            n as f64 / per.as_secs_f64()
        );

        // LDS kernel: cells in workgroup shared memory
        let lds_runner = match LdsRunner::new(&prog, 100_000) {
            Ok(r) => {
                eprintln!(
                    "lds tiling: lanes={} cell_words={} lds_bytes={}",
                    r.tiling().lanes,
                    r.tiling().cell_words,
                    r.tiling().lds_bytes
                );
                r
            }
            Err(RunError::GpuUnavailable(m)) => {
                eprintln!("SKIP lds: {m}");
                return;
            }
            Err(e) => panic!("{e:?}"),
        };
        let gpu2 = lds_runner.run(&prog, &inputs, n).expect("lds run");
        for (i, (g, c)) in gpu2.iter().zip(cpu.iter()).enumerate() {
            assert_eq!(g, c, "LDS divergence at instance {i}");
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = lds_runner.run(&prog, &inputs, n).unwrap();
        }
        let per = t0.elapsed() / iters;
        println!(
            "lds    {n:>6}: {per:>10.3?}/batch   {:>10.0} instances/s",
            n as f64 / per.as_secs_f64()
        );

        // LDS scored mode (grading on device)
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = lds_runner
                .run_scored(&prog, &inputs, &expected_flat, n)
                .unwrap();
        }
        let per = t0.elapsed() / iters;
        println!(
            "lds+sc {n:>6}: {per:>10.3?}/batch   {:>10.0} instances/s",
            n as f64 / per.as_secs_f64()
        );

        // occupancy sweep: smaller tilings -> more workgroups per CU
        for &cap in &[32_768usize, 16_384, 8_192] {
            let r = match LdsRunner::with_lds_cap(&prog, 100_000, cap) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("skip cap {cap}: {e:?}");
                    continue;
                }
            };
            let out = r.run(&prog, &inputs, n).expect("lds run");
            for (i, (g, c)) in out.iter().zip(cpu.iter()).enumerate() {
                assert_eq!(g, c, "LDS(cap={cap}) divergence at {i}");
            }
            let it = 3.max(iters / 2);
            let t0 = Instant::now();
            for _ in 0..it {
                let _ = r.run(&prog, &inputs, n).unwrap();
            }
            let per = t0.elapsed() / it;
            println!(
                "lds@{cap:<6} (lanes={}): {per:>10.3?}/batch  {:>8.0} inst/s",
                r.tiling().lanes,
                n as f64 / per.as_secs_f64()
            );
        }
    }
}
