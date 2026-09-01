//! Systolic 2D-mesh matrix multiplication in the Dally IR.
//!
//! A K x K systolic array multiplies matrices by streaming A east and
//! B west (or north/south) through a grid of processing elements; each
//! PE keeps one accumulator and sees each operand exactly once. That
//! "each value read once, at a fixed nearby cell" property is the
//! entire point: it is the data-movement-optimal schedule the Dally
//! cost model rewards.
//!
//! This example builds the straight-line IR for a small systolic GEMM
//! (4x4 x 4x4), scores it under the model, and compares against a
//! naive triple-loop GEMM IR over identical semantics - demonstrating
//! the cost gap the benchmark's matmul track asks competitors to close.

use dally_eval::ir::{Op, OpKind, Program};

#[derive(Default)]
struct B {
    p: Program,
    next: u16,
}

impl B {
    fn alloc(&mut self) -> u16 {
        self.next += 1;
        self.next
    }
    fn push(&mut self, op: Op) {
        self.p.push(op);
    }
    fn set(&mut self, v: u8) -> u16 {
        let d = self.alloc();
        self.push(Op {
            kind: OpKind::Set,
            dst: d,
            a: 1,
            b: 1,
            c: 1,
            imm: v,
        });
        d
    }
    fn mul_add(&mut self, a: u16, b: u16, acc: u16) -> u16 {
        let prod = self.alloc();
        self.push(Op {
            kind: OpKind::Mul,
            dst: prod,
            a,
            b,
            c: 1,
            imm: 0,
        });
        let sum = self.alloc();
        self.push(Op {
            kind: OpKind::Add,
            dst: sum,
            a: acc,
            b: prod,
            c: 1,
            imm: 0,
        });
        sum
    }
    fn finish(mut self, inputs: Vec<u16>, outputs: Vec<u16>) -> Program {
        self.p.inputs = inputs;
        self.p.outputs = outputs;
        self.p.finalize();
        self.p
    }
}

/// Naive GEMM: C[i][j] = sum_k A[i][k]*B[k][j], everything in global
/// cells, read distances whatever the layout gives.
fn scratch_start(a_base: u16, k: usize) -> u16 {
    // scratch may use everything below the first input span; when
    // inputs start at 1 there is nothing below, so scratch goes above
    if a_base == 1 {
        (b_offset(a_base, k) as u16) + (k * k) as u16 + 1
    } else {
        1
    }
}

fn b_offset(a_base: u16, k: usize) -> usize {
    a_base as usize + k * k
}

fn naive_gemm(a_base: u16, b_base: u16, k: usize) -> (Program, Vec<u16>) {
    let mut b = B {
        next: scratch_start(a_base, k),
        ..Default::default()
    };
    // reserve the low span as unused so scratch lands just above it;
    // naive does NOT stage: every operand read pays sqrt(input_addr)
    let mut c_cells = vec![];
    for i in 0..k {
        for j in 0..k {
            let zero = b.set(0);
            let mut acc = zero;
            for kk in 0..k {
                let a = a_base + (i * k + kk) as u16;
                let bb = b_base + (kk * k + j) as u16;
                acc = b.mul_add(a, bb, acc);
            }
            c_cells.push(acc);
        }
    }
    let mut all = (a_base..a_base + (k * k) as u16).collect::<Vec<u16>>();
    all.extend((b_base..b_base + (k * k) as u16).collect::<Vec<u16>>());
    let prog = b.finish(all, c_cells.clone());
    (prog, c_cells)
}

/// Systolic schedule: stage each operand ONCE from its high input
/// address into a low cell; every subsequent read pays ~1 instead of
/// ~sqrt(input_addr). Wins when each operand is read more than once
/// (here: K times, once per PE column/row it feeds).
fn systolic_gemm(a_base: u16, b_base: u16, k: usize) -> (Program, Vec<u16>) {
    let mut b = B {
        next: scratch_start(a_base, k),
        ..Default::default()
    };
    let mut a_stage = vec![vec![0u16; k]; k];
    let mut b_stage = vec![vec![0u16; k]; k];
    for (i, row) in a_stage.iter_mut().enumerate() {
        for (kk, slot) in row.iter_mut().enumerate() {
            let src = a_base + (i * k + kk) as u16;
            let dst = b.alloc();
            b.push(Op {
                kind: OpKind::Copy,
                dst,
                a: src,
                b: 1,
                c: 1,
                imm: 0,
            });
            *slot = dst;
        }
    }
    for (kk, row) in b_stage.iter_mut().enumerate() {
        for (j, slot) in row.iter_mut().enumerate() {
            let src = b_base + (kk * k + j) as u16;
            let dst = b.alloc();
            b.push(Op {
                kind: OpKind::Copy,
                dst,
                a: src,
                b: 1,
                c: 1,
                imm: 0,
            });
            *slot = dst;
        }
    }
    // transpose so the inner loop walks contiguous columns
    let mut bt = vec![vec![0u16; k]; k];
    for (kk, row) in b_stage.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            bt[j][kk] = v;
        }
    }
    let mut c_cells = vec![];
    for arow in &a_stage {
        for brow in &bt {
            let zero = b.set(0);
            let mut acc = zero;
            for (a_cell, b_cell) in arow.iter().zip(brow.iter()) {
                acc = b.mul_add(*a_cell, *b_cell, acc);
            }
            c_cells.push(acc);
        }
    }
    let mut all = (a_base..a_base + (k * k) as u16).collect::<Vec<u16>>();
    all.extend((b_base..b_base + (k * k) as u16).collect::<Vec<u16>>());
    let prog = b.finish(all, c_cells.clone());
    (prog, c_cells)
}

fn check(k: usize, a_base: u16, b_base: u16) {
    use dally_eval::eval::Machine;
    let (naive, _) = naive_gemm(a_base, b_base, k);
    let (systolic, _) = systolic_gemm(a_base, b_base, k);
    println!(
        "{k}x{k} GEMM ({} MACs, inputs at addr {a_base}+):",
        k * k * k
    );
    println!(
        "  naive    IR: {:>6} ops, cost {:>9}",
        naive.len(),
        naive.static_cost
    );
    println!(
        "  systolic IR: {:>6} ops, cost {:>9}",
        systolic.len(),
        systolic.static_cost
    );
    println!(
        "  systolic saves {:.1}% of naive cost",
        100.0 * (1.0 - systolic.static_cost as f64 / naive.static_cost as f64)
    );
    // semantics vs reference on a deterministic input
    let a: Vec<u8> = (0..k * k).map(|x| (x * 7 + 3) as u8).collect();
    let bm: Vec<u8> = (0..k * k).map(|x| (x * 5 + 1) as u8).collect();
    // run() takes instance values in declaration order: A then B
    let mut input = a.clone();
    input.extend(bm.iter().copied());
    let mut want = vec![0u8; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut acc = 0u8;
            for kk in 0..k {
                acc = acc.wrapping_add(a[i * k + kk].wrapping_mul(bm[kk * k + j]));
            }
            want[i * k + j] = acc;
        }
    }
    let mut m1 = Machine::new(naive.max_addr);
    let o1 = m1.run(&naive, &input).unwrap();
    let mut m2 = Machine::new(systolic.max_addr);
    let o2 = m2.run(&systolic, &input).unwrap();
    assert_eq!(o1, want, "naive wrong at k={k}");
    assert_eq!(o2, want, "systolic wrong at k={k}");
    println!("  semantics: both match the reference C");
}

fn main() {
    // Toy scale first: inputs at the cheapest addresses leaves nothing
    // to stage - the honest negative control.
    check(4, 1, 1 + 16);
    // Benchmark-realistic placement: operands live at high addresses;
    // the systolic schedule stages them low, one copy per operand,
    // saving sqrt-distance on every re-read.
    check(4, 2001, 3001);
    check(8, 5001, 7001);
    check(16, 20001, 50001);

    println!("\nprinciple: staging one copy of each operand beats re-reading");
    println!("it at sqrt(addr) whenever it is consumed more than once - the");
    println!("data-movement core of the 2D systolic mesh, expressed as a");
    println!("straight-line Dally IR schedule.");
}
