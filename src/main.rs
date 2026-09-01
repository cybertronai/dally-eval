//! CLI: static cost, fixture validation, and rough benchmarking.
//!
//! Subcommands:
//!   dally-eval cost <program.ir>
//!   dally-eval run <program.ir> <fixture.bin>
//!   dally-eval bench <program.ir> [instances] [iters]
//!
//! Fixture format (little-endian): u32 n, u32 n_inputs, u64 reference
//! cost, n rows of n_inputs bytes, u32 out_words (= n * out_width),
//! then out_words bytes of expected outputs.

use std::fs;
use std::time::Instant;

use dally_eval::ir::Program;
use dally_eval::runner::{BatchRunner, CpuRunner};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("cost") => cmd_cost(&args[2..]),
        Some("run") => cmd_run(&args[2..]),
        Some("bench") => cmd_bench(&args[2..]),
        Some("verify") => cmd_verify(&args[2..]),
        _ => {
            eprintln!(
                "usage: dally-eval <cost|run|bench|verify> ...\n  verify: IR on stdin -> JSON {{cost, ops, inputs, outputs}}"
            );
            std::process::exit(2);
        }
    }
}

/// Machine-friendly mode for embedding from Python: reads the IR text
/// on stdin, writes one JSON object with the static scoring facts.
fn cmd_verify(_args: &[String]) {
    use std::io::Read;
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap_or_else(|e| die(e));
    let prog = Program::parse(&text).unwrap_or_else(|e| die(e));
    println!(
        "{{\"cost\":{},\"ops\":{},\"inputs\":{},\"outputs\":{}}}",
        prog.static_cost,
        prog.len(),
        prog.inputs.len(),
        prog.outputs.len()
    );
}

fn cmd_cost(args: &[String]) {
    let path = &args[0];
    let text = fs::read_to_string(path).unwrap_or_else(|e| die(e));
    let prog = Program::parse(&text).unwrap_or_else(|e| die(e));
    println!("{}", prog.static_cost);
}

fn cmd_run(args: &[String]) {
    let (ir_path, bin_path) = (&args[0], &args[1]);
    let prog = Program::parse(&fs::read_to_string(ir_path).unwrap_or_else(|e| die(e)))
        .unwrap_or_else(|e| die(e));
    let bin = fs::read(bin_path).unwrap_or_else(|e| die(e));
    let (n, width, ref_cost, inputs, expected) = parse_fixture(&bin);
    if prog.static_cost != ref_cost {
        die(format!(
            "static cost mismatch: rust {} vs reference {}",
            prog.static_cost, ref_cost
        ));
    }
    let t0 = Instant::now();
    let outs = CpuRunner
        .run_batch(&prog, &inputs, n)
        .unwrap_or_else(|(i, e)| die(format!("instance {i} trapped: {e:?}")));
    let dt = t0.elapsed();
    let out_w = prog.outputs.len();
    let correct = outs
        .iter()
        .zip(expected.chunks(out_w))
        .filter(|(got, want)| got == want)
        .count();
    println!(
        "instances {n}  width {width}  static_cost {}  correct {}/{} ({:.2}%)  {:?}",
        prog.static_cost,
        correct,
        n,
        100.0 * correct as f64 / n as f64,
        dt
    );
}

fn cmd_bench(args: &[String]) {
    let ir_path = &args[0];
    let reps: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(1024);
    let iters: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(10);
    let prog = Program::parse(&fs::read_to_string(ir_path).unwrap_or_else(|e| die(e)))
        .unwrap_or_else(|e| die(e));
    // synthetic instances from a simple LCG so the branchy paths vary
    let width = prog.inputs.len();
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut inputs = vec![0u8; reps * width];
    for b in inputs.iter_mut() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = (seed >> 33) as u8;
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = CpuRunner.run_batch(&prog, &inputs, reps).unwrap();
    }
    let total = t0.elapsed();
    let per = total / iters as u32;
    println!(
        "instances {reps} x {iters} iters  avg {per:?}/iter  {:.0} instances/s  ops {}",
        reps as f64 / per.as_secs_f64(),
        prog.len()
    );
}

fn parse_fixture(bin: &[u8]) -> (usize, usize, u64, Vec<u8>, Vec<u8>) {
    let mut off = 0;
    #[allow(clippy::redundant_closure)]
    let u32at = |off: &mut usize| {
        let v = u32::from_le_bytes(bin[*off..*off + 4].try_into().unwrap());
        *off += 4;
        v as usize
    };
    let n = u32at(&mut off);
    let width = u32at(&mut off);
    let ref_cost = u64::from_le_bytes(bin[off..off + 8].try_into().unwrap());
    off += 8;
    let inputs = bin[off..off + n * width].to_vec();
    off += n * width;
    let out_words = u32at(&mut off);
    let expected = bin[off..off + out_words].to_vec();
    (n, width, ref_cost, inputs, expected)
}

fn die<E: std::fmt::Display>(e: E) -> ! {
    eprintln!("error: {e}");
    std::process::exit(1);
}
