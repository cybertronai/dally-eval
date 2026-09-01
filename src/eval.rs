//! Zero-allocation simulator and correctness checkers.
//!
//! Semantics are bit-exact with `mask_sparse_parity.py`: 8-bit cells,
//! wrapping `add`/`sub`/`mul`, *floor* signed division (traps on zero),
//! signed compares, `select` as a ternary on a nonzero condition, and
//! every write storing the two's-complement byte.

use crate::ir::{Op, OpKind, Program};

/// Failure modes while interpreting one instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunError {
    /// `div` by zero. The Python reference raises; we trap the instance.
    DivideByZero { op_index: usize },
    /// An operand cell was never written (well-formed programs do not
    /// do this; the Python dict engine would KeyError).
    ReadBeforeWrite { op_index: usize, addr: u16 },
    /// The GPU backend could not initialize (no adapter / device).
    GpuUnavailable(String),
}

/// Reusable machine: owns the cell arena so batch evaluation allocates
/// nothing per instance.
pub struct Machine {
    cells: Vec<u8>,
    /// Highest address written so far (for cheap clears between runs).
    high_water: usize,
}

impl Machine {
    pub fn new(max_addr: u16) -> Self {
        Machine {
            cells: vec![0u8; max_addr as usize + 1],
            high_water: 0,
        }
    }

    /// Run one instance: write `inputs` into the declared input cells,
    /// interpret every op, return the declared output bytes.
    pub fn run(&mut self, prog: &Program, inputs: &[u8]) -> Result<Vec<u8>, RunError> {
        self.reset();
        debug_assert_eq!(inputs.len(), prog.inputs.len());
        for (&addr, &val) in prog.inputs.iter().zip(inputs) {
            self.store(addr, val);
        }
        let cells = &mut self.cells;
        for i in 0..prog.kinds.len() {
            let op = Op {
                kind: prog.kinds[i],
                dst: prog.dst[i],
                a: prog.a[i],
                b: prog.b[i],
                c: prog.c[i],
                imm: prog.imm[i],
            };
            exec_op(&op, i, cells)?;
        }
        Ok(prog.outputs.iter().map(|&o| cells[o as usize]).collect())
    }

    fn reset(&mut self) {
        let n = self.high_water.min(self.cells.len() - 1);
        self.cells[..=n].fill(0);
        self.high_water = 0;
    }

    #[inline]
    fn store(&mut self, addr: u16, val: u8) {
        self.cells[addr as usize] = val;
        self.high_water = self.high_water.max(addr as usize);
    }
}

/// Reference semantics for one instruction over the cell array.
/// Exposed for GPU-parity cross-checks.
#[inline]
pub fn exec_op(op: &Op, i: usize, cells: &mut [u8]) -> Result<(), RunError> {
    use OpKind::*;
    let rd = |addr: u16| -> Result<u8, RunError> {
        let v = cells[addr as usize];
        // The Python dict engine errors on never-written cells. A zero
        // byte is indistinguishable from unwritten after a reset, so we
        // approximate: well-formed benchmark programs always write
        // before read; we do not trap here (documented divergence from
        // the dict engine's KeyError, matching the numpy vector engine,
        // which also treats unwritten cells as 0).
        let _ = v;
        Ok(v)
    };
    match op.kind {
        Set => cells[op.dst as usize] = op.imm,
        Copy => cells[op.dst as usize] = rd(op.a)?,
        Not => cells[op.dst as usize] = !rd(op.a)?,
        Abs => cells[op.dst as usize] = (rd(op.a)? as i8).wrapping_abs() as u8,
        And => cells[op.dst as usize] = rd(op.a)? & rd(op.b)?,
        Or => cells[op.dst as usize] = rd(op.a)? | rd(op.b)?,
        Xor => cells[op.dst as usize] = rd(op.a)? ^ rd(op.b)?,
        Add => cells[op.dst as usize] = rd(op.a)?.wrapping_add(rd(op.b)?),
        Sub => cells[op.dst as usize] = rd(op.a)?.wrapping_sub(rd(op.b)?),
        Mul => cells[op.dst as usize] = rd(op.a)?.wrapping_mul(rd(op.b)?),
        Div => {
            let x = rd(op.a)? as i8;
            let y = rd(op.b)? as i8;
            if y == 0 {
                return Err(RunError::DivideByZero { op_index: i });
            }
            // Python `//` is floor division; Rust `/` truncates.
            let q = x.wrapping_div(y);
            let r = x.wrapping_rem(y);
            let floored = if r != 0 && (r < 0) != (y < 0) {
                q - 1
            } else {
                q
            };
            cells[op.dst as usize] = floored as u8;
        }
        CmpEq => cells[op.dst as usize] = u8::from(rd(op.a)? == rd(op.b)?),
        CmpNe => cells[op.dst as usize] = u8::from(rd(op.a)? != rd(op.b)?),
        CmpLt => cells[op.dst as usize] = u8::from((rd(op.a)? as i8) < (rd(op.b)? as i8)),
        CmpLe => cells[op.dst as usize] = u8::from((rd(op.a)? as i8) <= (rd(op.b)? as i8)),
        CmpGt => cells[op.dst as usize] = u8::from((rd(op.a)? as i8) > (rd(op.b)? as i8)),
        CmpGe => cells[op.dst as usize] = u8::from((rd(op.a)? as i8) >= (rd(op.b)? as i8)),
        Select => {
            let cond = rd(op.a)?;
            let x = rd(op.b)?;
            let y = rd(op.c)?;
            cells[op.dst as usize] = if cond != 0 { x } else { y };
        }
    }
    Ok(())
}

/// Pluggable reference function for correctness checking: decides
/// whether a program's outputs are *correct* for one instance.
pub trait Checker {
    type Instance;
    /// Ground-truth output bytes for an instance, computed natively.
    fn reference(&self, instance: &Self::Instance) -> Vec<u8>;
    /// Number of output cells the checker compares.
    fn out_len(&self) -> usize;
}

/// Sparse-parity mask checker: instance = (input bits, secret mask);
/// correct iff every output byte equals the mask byte.
pub struct MaskChecker {
    pub n_bits: usize,
}

impl Checker for MaskChecker {
    type Instance = (Vec<u8>, Vec<u8>);
    fn reference(&self, instance: &(Vec<u8>, Vec<u8>)) -> Vec<u8> {
        instance.1.clone()
    }
    fn out_len(&self) -> usize {
        self.n_bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(text: &str, inputs: &[u8]) -> Vec<u8> {
        let prog = crate::ir::Program::parse(text).unwrap();
        let mut m = Machine::new(prog.max_addr);
        m.run(&prog, inputs).unwrap()
    }

    #[test]
    fn wrapping_semantics() {
        // 200 + 100 wraps to 44; 0 - 1 wraps to 255
        let out = run(
            "5\nset 1,200\nset 2,100\nadd 3,1,2\nsub 4,1,2\n3,4\n",
            &[0u8; 1],
        );
        assert_eq!(out, vec![44, 100]);
    }

    #[test]
    fn floor_division_matches_python() {
        // Python: -7 // 2 == -4 (floor), 7 // -2 == -4, -8 // 2 == -4
        let out = run(
            "1\nset 1,-7\nset 2,2\ndiv 3,1,2\nset 4,7\nset 5,-2\ndiv 6,4,5\n3,6\n",
            &[0u8; 1],
        );
        // -4 as u8 = 252
        assert_eq!(out, vec![252, 252]);
    }

    #[test]
    fn div_by_zero_traps() {
        let prog = crate::ir::Program::parse("1\nset 1,0\nset 2,5\ndiv 3,2,1\n3\n").unwrap();
        let mut m = Machine::new(prog.max_addr);
        assert!(matches!(
            m.run(&prog, &[0]),
            Err(RunError::DivideByZero { .. })
        ));
    }

    #[test]
    fn signed_compares_and_select() {
        // -1 (255) < 1 -> 1; select picks else-branch on cond 0
        let out = run(
            "1\nset 1,255\nset 2,1\ncmp 3,1,2,lt\nset 4,7\nset 7,0\nselect 5,3,4,2\nselect 6,7,4,2\n3,5,6\n",
            &[0u8; 1],
        );
        assert_eq!(out, vec![1, 7, 1]);
    }

    #[test]
    fn not_abs() {
        let out = run("1\nset 1,0\nnot 2,1\nset 3,128\nabs 4,3\n2,4\n", &[0u8; 1]);
        // !0 = 255; abs(-128) wraps to 128
        assert_eq!(out, vec![255, 128]);
    }
}
