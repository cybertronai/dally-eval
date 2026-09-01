//! Batch runners: one trait, CPU (Rayon) and GPU (wgpu) backends that
//! consume the same flat buffers.

use crate::eval::RunError;
use crate::ir::Program;

/// Evaluate a program over a batch of instances.
///
/// `instances` is a flat row-major matrix of `n * prog.inputs.len()`
/// bytes. Returns one output row per instance, or the per-instance
/// error index for trapping programs.
pub trait BatchRunner {
    fn run_batch(
        &self,
        prog: &Program,
        instances: &[u8],
        n: usize,
    ) -> Result<Vec<Vec<u8>>, (usize, RunError)>;
}

/// CPU backend: Rayon across instances, one reused [`Machine`] per
/// worker thread (zero allocation in steady state).
pub struct CpuRunner;

impl BatchRunner for CpuRunner {
    fn run_batch(
        &self,
        prog: &Program,
        instances: &[u8],
        n: usize,
    ) -> Result<Vec<Vec<u8>>, (usize, RunError)> {
        use rayon::prelude::*;

        let width = prog.inputs.len();
        debug_assert_eq!(instances.len(), n * width);
        let rows: Vec<Result<Vec<u8>, RunError>> = instances
            .par_chunks_exact(width)
            .map_init(
                || crate::eval::Machine::new(prog.max_addr),
                |m, row| m.run(prog, row),
            )
            .collect();
        rows.into_iter()
            .enumerate()
            .map(|(i, r)| r.map_err(|e| (i, e)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Program;

    #[test]
    fn parallel_batch_matches_serial() {
        let text = "10,11\nset 1,3\nadd 2,10,11\nmul 3,2,1\n3,2\n";
        let prog = Program::parse(text).unwrap();
        let width = prog.inputs.len();
        let instances: Vec<u8> = (0..1024u32)
            .flat_map(|i| vec![(i % 7) as u8, (i % 13) as u8])
            .collect();
        let out = CpuRunner.run_batch(&prog, &instances, 1024).unwrap();
        for (i, row) in out.iter().enumerate() {
            let a = (i as u32 % 7) as u8;
            let b = (i as u32 % 13) as u8;
            assert_eq!(row[0], (a.wrapping_add(b)).wrapping_mul(3));
            assert_eq!(row[1], a.wrapping_add(b));
        }
        assert_eq!(width, 2);
    }
}
