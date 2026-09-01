//! Task-family validation: 4x4 matmul, 5-input polynomial, and
//! 16/32-bit sparse parity trees, each built as an IR program and
//! checked against a native reference.

use dally_eval::eval::Machine;
use dally_eval::ir::{Op, OpKind, Program};

fn builder(n_inputs: u16) -> ProgramBuilder {
    // inputs occupy cells 1..=n_inputs; scratch allocates above them
    ProgramBuilder {
        next: n_inputs,
        ..Default::default()
    }
}

#[derive(Default)]
struct ProgramBuilder {
    prog: Program,
    next: u16,
}

impl ProgramBuilder {
    fn alloc(&mut self) -> u16 {
        self.next += 1;
        self.next
    }
    fn push(&mut self, op: Op) {
        self.prog.push(op);
    }
    fn build(self, inputs: Vec<u16>, outputs: Vec<u16>) -> Program {
        let mut p = self.prog;
        p.inputs = inputs;
        p.outputs = outputs;
        p.finalize();
        p
    }
}

use OpKind as K;

/// 4x4 matmul C = A x B over wrapping 8-bit arithmetic.
/// Inputs: 16 bytes of A (row-major), then 16 bytes of B.
#[test]
fn matmul_4x4_matches_reference() {
    let mut b = builder(32);
    // cells 1..=32 are A and B inputs
    let a_base = 1u16;
    let b_base = 17u16;
    let mut c_cells = vec![];
    for i in 0..4usize {
        for j in 0..4usize {
            let acc = b.alloc();
            b.push(Op {
                kind: K::Set,
                dst: acc,
                a: 1,
                b: 1,
                c: 1,
                imm: 0,
            });
            for k in 0..4usize {
                let prod = b.alloc();
                b.push(Op {
                    kind: K::Mul,
                    dst: prod,
                    a: a_base + (i * 4 + k) as u16,
                    b: b_base + (k * 4 + j) as u16,
                    c: 1,
                    imm: 0,
                });
                b.push(Op {
                    kind: K::Add,
                    dst: acc,
                    a: acc,
                    b: prod,
                    c: 1,
                    imm: 0,
                });
            }
            c_cells.push(acc);
        }
    }
    let inputs: Vec<u16> = (1..=32).collect();
    let prog = b.build(inputs, c_cells.clone());

    let cases: Vec<(Vec<u8>, Vec<u8>)> = (0..24u32)
        .map(|s| {
            let a: Vec<u8> = (0..16).map(|k| (s * 7 + k as u32 * 13) as u8).collect();
            let mat: Vec<u8> = (0..16).map(|k| (s * 11 + k as u32 * 5 + 1) as u8).collect();
            let mut row = a.clone();
            row.extend(mat.clone());
            // native reference
            let mut want = vec![];
            for i in 0..4 {
                for j in 0..4 {
                    let mut acc: u8 = 0;
                    for k in 0..4 {
                        acc = acc.wrapping_add(a[i * 4 + k].wrapping_mul(mat[k * 4 + j]));
                    }
                    want.push(acc);
                }
            }
            (row, want)
        })
        .collect();

    let mut m = Machine::new(prog.max_addr);
    for (row, want) in &cases {
        let got = m.run(&prog, row).unwrap();
        assert_eq!(&got, want);
    }
    // cost sanity: a real program with priced reads
    assert!(prog.static_cost > 0);
}

/// 5-input Horner polynomial p(x) = (((c4 x + c3) x + c2) x + c1) x + c0.
#[test]
fn polynomial_5_term_matches_reference() {
    let coeffs = [3u8, 200, 77, 150, 9];
    let mut b = builder(32);
    let x = 1u16;
    // acc = c4
    let mut acc = b.alloc();
    b.push(Op {
        kind: K::Set,
        dst: acc,
        a: 1,
        b: 1,
        imm: coeffs[4],
        c: 1,
    });
    for &c in coeffs.iter().rev().skip(1) {
        // acc = acc * x + c
        let prod = b.alloc();
        b.push(Op {
            kind: K::Mul,
            dst: prod,
            a: acc,
            b: x,
            c: 1,
            imm: 0,
        });
        let cst = b.alloc();
        b.push(Op {
            kind: K::Set,
            dst: cst,
            a: 1,
            b: 1,
            imm: c,
            c: 1,
        });
        let sum = b.alloc();
        b.push(Op {
            kind: K::Add,
            dst: sum,
            a: prod,
            b: cst,
            c: 1,
            imm: 0,
        });
        acc = sum;
    }
    let prog = b.build(vec![x], vec![acc]);
    let mut m = Machine::new(prog.max_addr);
    for xv in 0u16..256 {
        let got = m.run(&prog, &[xv as u8]).unwrap();
        // native Horner
        let mut a: u8 = coeffs[4];
        for &c in coeffs.iter().rev().skip(1) {
            a = a.wrapping_mul(xv as u8).wrapping_add(c);
        }
        assert_eq!(got[0], a, "x={xv}");
    }
}

/// Sparse parity tree over n input bits with a fixed secret subset,
/// built as a pure XOR chain; also checks that reordering the tree
/// changes the static cost (the "optimization" dimension).
fn parity_prog(n: usize, secret: &[usize]) -> Program {
    let mut b = builder(32);
    let mut acc = 1u16 + secret[0] as u16;
    for &s in &secret[1..] {
        let out = b.alloc();
        b.push(Op {
            kind: K::Xor,
            dst: out,
            a: acc,
            b: 1 + s as u16,
            c: 1,
            imm: 0,
        });
        acc = out;
    }
    let inputs: Vec<u16> = (1..=n as u16).collect();
    b.build(inputs, vec![acc])
}

#[test]
fn sparse_parity_trees_16_and_32() {
    for (n, secret) in [
        (16usize, vec![2usize, 5, 9, 14]),
        (32, vec![1, 7, 13, 22, 30]),
    ] {
        let prog = parity_prog(n, &secret);
        let mut m = Machine::new(prog.max_addr);
        // deterministic pseudo-random bit vectors
        let mut seed = 0x1234_5678u64 ^ n as u64;
        for _ in 0..64 {
            let mut bits = vec![0u8; n];
            for (i, bit) in bits.iter_mut().enumerate() {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *bit = ((seed >> 60) & 1 == 1 && i % 3 != 0) as u8;
            }
            let want = secret.iter().map(|&s| bits[s]).fold(0u8, |a, v| a ^ v);
            let got = m.run(&prog, &bits).unwrap();
            assert_eq!(got[0], want, "n={n}");
        }
    }
}

#[test]
fn tree_layout_changes_static_cost() {
    // Same XOR tree over cells 1..=32, but a chain that starts at the
    // low addresses costs less than one whose accumulator wanders high:
    // trivially true here because parity_prog allocates fresh cells; we
    // verify the pricing responds to addresses at all.
    let p_low = parity_prog(32, &[1, 2, 3, 4, 5]);
    let p_high = parity_prog(32, &[27, 28, 29, 30, 31]);
    assert!(p_low.static_cost != p_high.static_cost);
}
