//! dally-eval: high-throughput evaluator for Bill Dally 2D Manhattan
//! communication-cost IR programs.
//!
//! The cost model: a processor sits at the origin of an upper-half-plane
//! 2D memory grid; cell `idx` is at Manhattan distance `ceil(sqrt(idx))`.
//! Reading a cell costs its distance; writes and arithmetic are free.
//! A program is a straight-line (branchless, loop-free) sequence of
//! 3-address instructions over 8-bit cells; its score is the static sum
//! of operand-read costs plus one final read per declared output.
//!
//! Semantics are bit-exact with the Python reference implementation in
//! `sutro-problems/sparse-parity/mask_sparse_parity.py`, which is the
//! normative golden. Notable details: `add`/`sub`/`mul` wrap; `div` is
//! *floor* division on signed 8-bit interpretations (not truncation) and
//! traps on zero; comparisons are signed; every write stores the
//! two's-complement byte.
//!
//! Parallelism shape: within one instance the program is a fixed
//! sequential dataflow chain, but instances (and candidate schedules)
//! are embarrassingly parallel. All runners consume the same flat SoA
//! buffers ([`ir::Program`], row-major instance matrix), so a Rayon CPU
//! runner and a future wgpu compute backend share one interface.

pub mod cost;
pub mod eval;
pub mod ir;
pub mod runner;
pub mod wgpu_runner;

pub use eval::{Checker, Machine, RunError};
pub use ir::{Op, OpKind, Program};
pub use runner::{BatchRunner, CpuRunner};
pub use wgpu_runner::WgpuRunner;
