//! s1-dally: the bridge crate between System-1's MCTS planner and
//! dally-eval's evaluators. System-1 proposes schedules; dally-eval
//! scores them exactly; this crate defines the interchange.
//!
//! Design: an MCTS node is a partial schedule (a prefix of IR ops).
//! Expansion appends a legal op; evaluation scores the completed
//! program's static cost (the search signal). The trait below lets
//! System-1 plug any proposal policy while dally-eval stays the sole
//! scorer - single source of truth for cost.

pub use dally_eval::ir::{Op, OpKind, Program};

/// A partial GEMM schedule under construction by the MCTS planner.
#[derive(Clone, Debug, Default)]
pub struct PartialSchedule {
    pub ops: Vec<Op>,
    /// inputs already declared (address list)
    pub inputs: Vec<u16>,
    /// remaining budget (op cap)
    pub budget: usize,
}

impl PartialSchedule {
    pub fn new(inputs: Vec<u16>, op_cap: usize) -> Self {
        Self { ops: Vec::new(), inputs, budget: op_cap }
    }

    /// Append an op if budget allows; returns false when the schedule
    /// is complete (budget exhausted).
    pub fn push(&mut self, op: Op) -> bool {
        if self.budget == 0 {
            return false;
        }
        self.ops.push(op);
        self.budget -= 1;
        true
    }

    /// Compile to a runnable Program with the given outputs.
    pub fn finish(self, outputs: Vec<u16>) -> Program {
        let mut p = Program::default();
        for op in self.ops {
            p.push(op);
        }
        p.inputs = self.inputs;
        p.outputs = outputs;
        p.finalize();
        p
    }
}

/// The MCTS proposal policy: given a partial schedule, propose the next
/// op. Implemented by System-1; this crate provides the interface and
/// the exact scorer.
pub trait ProposalPolicy {
    fn propose(&mut self, partial: &PartialSchedule) -> Option<Op>;
    fn name(&self) -> &'static str;
}

/// Exact score of a completed schedule: static read cost from
/// dally-eval's cost model (the same function the competition uses).
pub fn score(program: &Program) -> u64 {
    program.static_cost
}

/// One MCTS driver: expand via the policy, score when complete.
pub struct SearchStep<P: ProposalPolicy> {
    pub policy: P,
}

impl<P: ProposalPolicy> SearchStep<P> {
    /// Roll out a full schedule and return (program, cost).
    pub fn rollout(
        &mut self,
        inputs: Vec<u16>,
        outputs: Vec<u16>,
        op_cap: usize,
    ) -> (Program, u64) {
        let mut partial = PartialSchedule::new(inputs, op_cap);
        while let Some(op) = self.policy.propose(&partial) {
            if !partial.push(op) {
                break;
            }
        }
        let program = partial.finish(outputs);
        let cost = score(&program);
        (program, cost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedMul;
    impl ProposalPolicy for FixedMul {
        fn propose(&mut self, p: &PartialSchedule) -> Option<Op> {
            if p.ops.is_empty() {
                Some(Op { kind: OpKind::Mul, dst: 40, a: 1, b: 17, c: 1, imm: 0 })
            } else {
                None
            }
        }
        fn name(&self) -> &'static str {
            "fixed-mul"
        }
    }

    #[test]
    fn rollout_scores_one_mul() {
        let inputs: Vec<u16> = (1..=32).collect();
        let mut step = SearchStep { policy: FixedMul };
        let (prog, cost) = step.rollout(inputs, vec![40], 10);
        assert_eq!(prog.len(), 1);
        // mul reads addr 1 (cost 1) + addr 17 (cost 5) + output 40 (7)
        assert_eq!(cost, 1 + 5 + 7);
    }
}
