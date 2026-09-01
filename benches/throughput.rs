//! Throughput benchmarks on the real 72,780-op benchmark program.
//!
//! The Python reference evaluates 1,024 dev-suite instances of this
//! program in ~2.8 s (batched numpy). These benches measure the Rust
//! engine on the same workload shape.

use std::fs;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use dally_eval::eval::Machine;
use dally_eval::ir::Program;
use dally_eval::runner::{BatchRunner, CpuRunner};

fn fixture_ir() -> String {
    fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/siswalk1_cap2.ir"),
    )
    .unwrap()
}

fn synth_instances(prog: &Program, n: usize) -> Vec<u8> {
    let width = prog.inputs.len();
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut v = vec![0u8; n * width];
    for b in v.iter_mut() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = (seed >> 33) as u8;
    }
    v
}

fn bench_parse(c: &mut Criterion) {
    let text = fixture_ir();
    c.bench_function("parse_72k_ops", |b| {
        b.iter(|| Program::parse(&text).unwrap())
    });
}

fn bench_serial(c: &mut Criterion) {
    let prog = Program::parse(&fixture_ir()).unwrap();
    let inputs = synth_instances(&prog, 1024);
    let width = prog.inputs.len();
    let mut group = c.benchmark_group("eval_serial");
    group.throughput(Throughput::Elements(1024));
    group.bench_function(BenchmarkId::new("siswalk1_cap2", 1024), |b| {
        b.iter(|| {
            let mut m = Machine::new(prog.max_addr);
            let mut ok = 0usize;
            for row in inputs.chunks_exact(width) {
                if m.run(&prog, row).unwrap().len() == 32 {
                    ok += 1;
                }
            }
            ok
        })
    });
    group.finish();
}

fn bench_parallel(c: &mut Criterion) {
    let prog = Program::parse(&fixture_ir()).unwrap();
    let inputs = synth_instances(&prog, 1024);
    let mut group = c.benchmark_group("eval_parallel");
    group.throughput(Throughput::Elements(1024));
    group.sample_size(20);
    group.bench_function(BenchmarkId::new("siswalk1_cap2", 1024), |b| {
        b.iter(|| CpuRunner.run_batch(&prog, &inputs, 1024).unwrap())
    });
    group.finish();
}

fn bench_cost_fold(c: &mut Criterion) {
    let prog = Program::parse(&fixture_ir()).unwrap();
    c.bench_function("static_cost_fold_72k", |b| {
        b.iter(|| {
            prog.kinds
                .iter()
                .enumerate()
                .map(|(i, k)| {
                    dally_eval::ir::Op {
                        kind: *k,
                        dst: prog.dst[i],
                        a: prog.a[i],
                        b: prog.b[i],
                        c: prog.c[i],
                        imm: prog.imm[i],
                    }
                    .read_cost()
                })
                .sum::<u64>()
        })
    });
}

criterion_group!(
    benches,
    bench_parse,
    bench_cost_fold,
    bench_serial,
    bench_parallel
);
criterion_main!(benches);
