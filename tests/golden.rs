//! Golden parity against the Python reference evaluator.
//!
//! `tests/fixtures/siswalk1_cap2.ir` is the benchmark's current 20/40%
//! band record program (72,780 ops, static cost 1,317,480) and
//! `tests/fixtures/parity32.bin` carries 32 real dev-suite instances
//! with the Python engine's exact output bytes.

use std::fs;

use dally_eval::ir::Program;
use dally_eval::runner::{BatchRunner, CpuRunner};

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn parse_fixture(bin: &[u8]) -> (usize, usize, u64, Vec<u8>, Vec<u8>) {
    let mut off = 0usize;
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

#[test]
fn golden_static_cost_matches_python() {
    let text = fs::read_to_string(fixture("siswalk1_cap2.ir")).unwrap();
    let prog = Program::parse(&text).unwrap();
    // 72,780 generator ops + 513 staging copies from optimize_layout
    assert_eq!(prog.len(), 73_293, "op count (staged program)");
    assert_eq!(prog.static_cost, 1_317_480, "static cost vs Python");
}

#[test]
fn golden_outputs_bit_exact() {
    let text = fs::read_to_string(fixture("siswalk1_cap2.ir")).unwrap();
    let prog = Program::parse(&text).unwrap();
    let bin = fs::read(fixture("parity32.bin")).unwrap();
    let (n, width, ref_cost, inputs, expected) = parse_fixture(&bin);
    assert_eq!(width, prog.inputs.len());
    assert_eq!(prog.static_cost, ref_cost);

    let outs = CpuRunner
        .run_batch(&prog, &inputs, n)
        .expect("benchmark program must not trap");
    let out_w = prog.outputs.len();
    for (i, (got, want)) in outs.iter().zip(expected.chunks(out_w)).enumerate() {
        assert_eq!(
            got.as_slice(),
            want,
            "instance {i}: outputs differ from Python reference"
        );
    }
}

#[test]
fn machine_recovery_fraction() {
    // The 32 golden instances are all solved by this program on the dev
    // suite; every output row must equal the hidden mask (all-ones
    // per-instance by construction of the fixture's expected bytes).
    let text = fs::read_to_string(fixture("siswalk1_cap2.ir")).unwrap();
    let prog = Program::parse(&text).unwrap();
    let bin = fs::read(fixture("parity32.bin")).unwrap();
    let (n, width, _cost, inputs, expected) = parse_fixture(&bin);
    assert_eq!(width, prog.inputs.len());
    let outs = CpuRunner.run_batch(&prog, &inputs, n).unwrap();
    let out_w = prog.outputs.len();
    let correct = outs
        .iter()
        .zip(expected.chunks(out_w))
        .filter(|(g, w)| g.as_slice() == *w)
        .count();
    assert_eq!(correct, n);
}
