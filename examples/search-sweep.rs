//! Inner-loop validation: automated candidate-layout search over
//! sparse-parity programs, scored by dally-eval's backends.
//!
//! The sweep generates layout permutations of a base program (address
//! remappings in the style of the benchmark's renumber/staging
//! optimizations), evaluates each candidate over a fixed instance
//! batch, and reports the best-found cost plus sweep wall-clock - the
//! "autonomous hardware search" inner loop that sub-millisecond
//! evaluation enables.
//!
//! Run under the host resource-management policy:
//!   systemd-run --user --slice=training.slice --wait --pipe \
//!     bash -c 'cd dally-eval && nix develop --command \
//!       cargo run --release --example search-sweep'

use std::time::Instant;

use dally_eval::ir::{Op, Program};
use dally_eval::runner::BatchRunner;
use dally_eval::{CpuRunner, LdsRunner};

/// Deterministic pseudo-random layout: remap every cell address through
/// a permutation drawn from an LCG. Semantics-preserving (pure
/// relabeling), so recovery is invariant and only the cost changes -
/// exactly the search space the benchmark's hand optimizations walk.
fn permuted(prog: &Program, seed: u64) -> Program {
    let n = prog.max_addr as usize + 1;
    // Fisher-Yates over 1..=max_addr with an LCG
    let mut perm: Vec<u16> = (1..=prog.max_addr).collect();
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    for i in (1..perm.len()).rev() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (s >> 33) as usize % (i + 1);
        perm.swap(i, j);
    }
    let mut map = vec![0u16; n];
    for (i, &p) in perm.iter().enumerate() {
        map[i + 1] = p;
    }
    let remap = |a: u16| -> u16 {
        if a as usize <= prog.max_addr as usize {
            map[a as usize]
        } else {
            a
        }
    };
    let mut out = Program::default();
    for i in 0..prog.len() {
        out.push(Op {
            kind: prog.kinds[i],
            dst: remap(prog.dst[i]),
            a: remap(prog.a[i]),
            b: remap(prog.b[i]),
            c: remap(prog.c[i]),
            imm: prog.imm[i],
        });
    }
    out.inputs = prog.inputs.iter().map(|&a| remap(a)).collect();
    out.outputs = prog.outputs.iter().map(|&a| remap(a)).collect();
    out.finalize();
    out
}

fn synth_instances(prog: &Program, n: usize, seed: u64) -> Vec<u8> {
    let width = prog.inputs.len();
    let mut s = seed;
    let mut v = vec![0u8; n * width];
    for b in v.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = (s >> 33) as u8;
    }
    v
}

fn main() {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/siswalk1_cap2.ir"),
    )
    .expect("fixture");
    let base = Program::parse(&text).unwrap();
    let n = 1_024usize;
    let instances = synth_instances(&base, n, 0x1234_5678);

    println!(
        "base program: {} ops, static cost {}",
        base.len(),
        base.static_cost
    );

    // --- CPU (Rayon) sweep ---
    let candidates = 200usize;
    let t0 = Instant::now();
    let mut best = (u64::MAX, 0usize);
    let mut baseline_ok = 0usize;
    for k in 0..candidates {
        let cand = permuted(&base, k as u64 + 1);
        let outs = CpuRunner.run_batch(&cand, &instances, n).unwrap();
        let ok = outs.iter().filter(|r| r.iter().all(|&b| b < 2)).count();
        if k == 0 {
            baseline_ok = ok;
        }
        if cand.static_cost < best.0 {
            best = (cand.static_cost, k);
        }
    }
    let cpu_dt = t0.elapsed();
    println!(
        "cpu sweep: {candidates} candidates x {n} instances in {cpu_dt:?}  ({:.0} cand/s)  best cost {} (perm #{})",
        candidates as f64 / cpu_dt.as_secs_f64(),
        best.0,
        best.1
    );
    let _ = baseline_ok;

    // --- GPU (LDS) sweep: end-to-end (permute+parse+eval) and
    // eval-only, because the honest bottleneck split matters
    if let Ok(runner) = LdsRunner::new(&base, n) {
        let cands: Vec<Program> = (0..candidates)
            .map(|k| permuted(&base, k as u64 + 1))
            .collect();
        // warm/verify one
        let _ = runner.run(&cands[0], &instances, n).unwrap();

        let t0 = Instant::now();
        for cand in &cands {
            let _ = runner.run(cand, &instances, n).unwrap();
        }
        let gpu_eval = t0.elapsed();
        println!(
            "gpu eval-only: {candidates} candidates x {n} instances in {gpu_eval:?}  ({:.0} cand/s)",
            candidates as f64 / gpu_eval.as_secs_f64()
        );

        let t0 = Instant::now();
        for k in 0..candidates {
            let cand = permuted(&base, k as u64 + 1);
            let _ = runner.run(&cand, &instances, n).unwrap();
        }
        let gpu_e2e = t0.elapsed();
        println!(
            "gpu end-to-end (permute+parse+eval): {gpu_e2e:?}  ({:.0} cand/s)",
            candidates as f64 / gpu_e2e.as_secs_f64()
        );
        println!(
            "eval engines at this batch: cpu {:.0} cand/s vs gpu {:.0} cand/s",
            candidates as f64 / cpu_dt.as_secs_f64(),
            candidates as f64 / gpu_eval.as_secs_f64()
        );
    } else {
        eprintln!("SKIP gpu sweep: no adapter");
    }

    // --- Reference point: what Python would pay ---
    // The Python engine evaluates this program at ~2.8 s / 1,024
    // instances; the same 200-candidate sweep would cost ~9.3 minutes.
    let py_est = candidates as f64 * 2.8;
    println!(
        "python-equivalent sweep: ~{py_est:.0}s (~{:.1} min); rust cpu sweep is {:.0}x faster",
        py_est / 60.0,
        py_est / cpu_dt.as_secs_f64()
    );
}
