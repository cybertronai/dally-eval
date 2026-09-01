//! 3-address instruction representation and text-IR parser.
//!
//! Storage is structure-of-arrays: parallel vectors of opcodes,
//! destinations, and operands, which keeps the interpreter's hot loop
//! cache-friendly on 100k..1M-instruction programs and gives GPU/WASM
//! backends flat buffers to upload unchanged.

use crate::cost::cost;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpKind {
    Set,
    Copy,
    Not,
    Abs,
    And,
    Or,
    Xor,
    Add,
    Sub,
    Mul,
    Div,
    CmpEq,
    CmpNe,
    CmpLt,
    CmpLe,
    CmpGt,
    CmpGe,
    Select,
}

/// One instruction. `a`/`b` are cell operands; `select` additionally
/// uses `c` (`select dst, a, b, c` = `a != 0 ? b : c`); `set` uses
/// `imm`. Unused fields are inert (never priced, never read).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Op {
    pub kind: OpKind,
    pub dst: u16,
    pub a: u16,
    pub b: u16,
    pub c: u16,
    pub imm: u8,
}

impl Op {
    /// Read cost of this instruction's operand fetches: every cell
    /// operand is charged its distance; the `set` immediate is free.
    #[inline]
    pub fn read_cost(&self) -> u64 {
        use OpKind::*;
        match self.kind {
            Set => 0,
            Copy | Not | Abs => cost(self.a as u32),
            And | Or | Xor | Add | Sub | Mul | Div => cost(self.a as u32) + cost(self.b as u32),
            CmpEq | CmpNe | CmpLt | CmpLe | CmpGt | CmpGe => {
                cost(self.a as u32) + cost(self.b as u32)
            }
            Select => cost(self.a as u32) + cost(self.b as u32) + cost(self.c as u32),
        }
    }
}

/// A parsed straight-line program: SoA instruction storage plus the
/// input/output address declarations.
#[derive(Clone, Debug, Default)]
pub struct Program {
    pub kinds: Vec<OpKind>,
    pub dst: Vec<u16>,
    pub a: Vec<u16>,
    pub b: Vec<u16>,
    pub c: Vec<u16>,
    pub imm: Vec<u8>,
    pub inputs: Vec<u16>,
    pub outputs: Vec<u16>,
    /// Highest cell address touched (1-based; 0 means empty program).
    pub max_addr: u16,
    /// Static read cost: sum of operand reads + one read per output.
    pub static_cost: u64,
}

impl Program {
    pub fn len(&self) -> usize {
        self.kinds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kinds.is_empty()
    }

    pub fn op(&self, i: usize) -> Op {
        Op {
            kind: self.kinds[i],
            dst: self.dst[i],
            a: self.a[i],
            b: self.b[i],
            c: self.c[i],
            imm: self.imm[i],
        }
    }

    pub fn push(&mut self, op: Op) {
        let touched = match op.kind {
            OpKind::Set => op.dst,
            _ => op.dst.max(op.a).max(op.b).max(op.c),
        };
        self.max_addr = self.max_addr.max(touched);
        self.static_cost += op.read_cost();
        self.kinds.push(op.kind);
        self.dst.push(op.dst);
        self.a.push(op.a);
        self.b.push(op.b);
        self.c.push(op.c);
        self.imm.push(op.imm);
    }

    /// Add the output-read charges and fold the declared input/output
    /// addresses into `max_addr` (operand tracking alone misses them).
    /// Called automatically by `parse`; programmatic builders that used
    /// `push` must call this once.
    pub fn finalize(&mut self) {
        self.static_cost += self.outputs.iter().map(|&o| cost(o as u32)).sum::<u64>();
        if let Some(m) = self.inputs.iter().chain(self.outputs.iter()).max() {
            self.max_addr = self.max_addr.max(*m);
        }
    }

    /// Parse the benchmark text IR: first line = comma-separated input
    /// addresses, last line = comma-separated output addresses, and one
    /// op per line between (`set d,i` / unary `op d,a` / binary
    /// `op d,a,b` / `cmp d,a,b,pred` / `select d,c,x,y`).
    pub fn parse(text: &str) -> Result<Program, String> {
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let first = lines.next().ok_or("empty program")?;
        let mut body: Vec<&str> = lines.collect();
        let last = body
            .pop()
            .ok_or("program has ops but no output declaration")?;
        let mut p = Program::default();
        p.inputs = parse_addr_list(first)?;
        p.outputs = parse_addr_list(last)?;
        if p.inputs.is_empty() {
            return Err("no input addresses declared".into());
        }
        for line in &body {
            p.push(parse_op(line)?);
        }
        p.finalize();
        Ok(p)
    }
}

fn parse_addr_list(line: &str) -> Result<Vec<u16>, String> {
    line.split(',')
        .map(|t| {
            let t = t.trim();
            let v: u16 = t.parse().map_err(|_| format!("bad address {t:?}"))?;
            if v == 0 {
                Err("addresses are 1-based; got 0".to_string())
            } else {
                Ok(v)
            }
        })
        .collect()
}

fn parse_op(line: &str) -> Result<Op, String> {
    let line = line.trim();
    let (head, rest) = line
        .split_once(' ')
        .ok_or_else(|| format!("malformed instruction {line:?}"))?;
    let mut parts = rest.split(',');
    let mut next = |what: &str| -> Result<u16, String> {
        let t = parts
            .next()
            .ok_or_else(|| format!("missing {what} in {line:?}"))?
            .trim();
        let v: u16 = t.parse().map_err(|_| format!("bad {what} {t:?}"))?;
        if v == 0 {
            return Err(format!("addresses are 1-based; got 0 in {line:?}"));
        }
        Ok(v)
    };
    use OpKind::*;
    if head == "cmp" {
        let dst = next("dest")?;
        let a = next("lhs")?;
        let b = next("rhs")?;
        let pred = parts
            .next()
            .ok_or_else(|| format!("cmp needs a predicate in {line:?}"))?
            .trim();
        let kind = match pred {
            "eq" => CmpEq,
            "ne" => CmpNe,
            "lt" => CmpLt,
            "le" => CmpLe,
            "gt" => CmpGt,
            "ge" => CmpGe,
            _ => return Err(format!("bad predicate {pred:?}")),
        };
        return Ok(Op {
            kind,
            dst,
            a,
            b,
            c: 1,
            imm: 0,
        });
    }
    let kind = match head {
        "set" => Set,
        "copy" => Copy,
        "not" => Not,
        "abs" => Abs,
        "and" => And,
        "or" => Or,
        "xor" => Xor,
        "add" => Add,
        "sub" => Sub,
        "mul" => Mul,
        "div" => Div,
        "select" => Select,
        other => return Err(format!("unknown opcode {other:?}")),
    };
    let dst = next("dest")?;
    if kind == Set {
        let t = parts
            .next()
            .ok_or_else(|| format!("set needs an immediate in {line:?}"))?
            .trim();
        let v: i32 = t.parse().map_err(|_| format!("bad immediate {t:?}"))?;
        if !(-128..=255).contains(&v) {
            return Err(format!("immediate {v} out of 8-bit range"));
        }
        return Ok(Op {
            kind,
            dst,
            a: 1,
            b: 1,
            c: 1,
            imm: (v & 0xFF) as u8,
        });
    }
    let rest_ops: Vec<&str> = parts.map(|s| s.trim()).collect();
    let addr = |t: &str, what: &str| -> Result<u16, String> {
        let v: u16 = t.parse().map_err(|_| format!("bad {what} {t:?}"))?;
        if v == 0 {
            return Err(format!("addresses are 1-based; got 0 in {line:?}"));
        }
        Ok(v)
    };
    use OpKind::*;
    match kind {
        Copy | Not | Abs => {
            if rest_ops.len() != 1 {
                return Err(format!("{head:?} needs exactly 1 operand in {line:?}"));
            }
            let a = addr(rest_ops[0], "src1")?;
            Ok(Op {
                kind,
                dst,
                a,
                b: 1,
                c: 1,
                imm: 0,
            })
        }
        Select => {
            if rest_ops.len() != 3 {
                return Err(format!("select needs 3 operands in {line:?}"));
            }
            let a = addr(rest_ops[0], "cond")?;
            let b = addr(rest_ops[1], "true-src")?;
            let c = addr(rest_ops[2], "false-src")?;
            Ok(Op {
                kind,
                dst,
                a,
                b,
                c,
                imm: 0,
            })
        }
        // Binary ops accept 3 operands (dst, s1, s2) or the 2-operand
        // accumulator form (dst, s2) whose first source is dst itself,
        // matching the reference parser (which also charges dst as a
        // read in that form).
        _ => match rest_ops.len() {
            1 => {
                let b = addr(rest_ops[0], "src2")?;
                Ok(Op {
                    kind,
                    dst,
                    a: dst,
                    b,
                    c: 1,
                    imm: 0,
                })
            }
            2 => {
                let a = addr(rest_ops[0], "src1")?;
                let b = addr(rest_ops[1], "src2")?;
                Ok(Op {
                    kind,
                    dst,
                    a,
                    b,
                    c: 1,
                    imm: 0,
                })
            }
            n => Err(format!("{head:?} needs 2 or 3 operands, got {n}")),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_costs_a_tiny_program() {
        let text = "10,11\nset 1,7\nadd 2,10,11\nxor 3,2,1\n2,3\n";
        let p = Program::parse(text).unwrap();
        assert_eq!(p.len(), 3);
        assert_eq!(p.inputs, vec![10, 11]);
        assert_eq!(p.outputs, vec![2, 3]);
        // set 0; add reads 10(4)+11(4)=8; xor reads 2(2)+1(1)=3;
        // output reads 2(2)+3(2)=4. Total 15.
        assert_eq!(p.static_cost, 15);
        assert_eq!(p.max_addr, 11);
    }

    #[test]
    fn select_prices_all_three_operands() {
        let text = "1,2,3,4\nselect 5,1,2,3\n5\n";
        let p = Program::parse(text).unwrap();
        // reads 1(1)+2(2)+3(2)=5, output 5(3) => 8
        assert_eq!(p.static_cost, 8);
    }

    #[test]
    fn accumulator_form_reads_dst_as_first_source() {
        // `or 5,3` == `or 5,5,3`: reads 5 and 3.
        let text = "3\nor 5,3\n5\n";
        let p = Program::parse(text).unwrap();
        assert_eq!(p.static_cost, cost(5) + cost(3) + cost(5));
    }

    #[test]
    fn rejects_zero_addresses_and_bad_ops() {
        assert!(Program::parse("0\nset 1,1\n1\n").is_err());
        assert!(Program::parse("1\nfrobnicate 2,1\n1\n").is_err());
        assert!(Program::parse("1\ncmp 2,1,1,maybe\n1\n").is_err());
        assert!(Program::parse("1\nselect 2,1\n1\n").is_err());
    }
}
