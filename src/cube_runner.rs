//! GPU backend implemented in pure Rust via [CubeCL](https://crates.io/crates/cubecl)
//! (`#[cube]` kernels compiled at runtime to the active backend's ISA).
//! No raw shader text, no manual uniform packing: scalar arguments ride
//! the launch ABI directly.
//!
//! The interpreter kernel is a faithful port of the CPU semantics in
//! pure u32 arithmetic (sign-bias trick for signed compares,
//! magnitude-based floor division), so the same code path works on
//! every backend.
//!
//! [`CubeRunner`] keeps the packed op stream and address tables
//! resident on the device across batches; per batch it uploads only
//! the instance matrix and reads back outputs and trap flags.

use crate::eval::RunError;
use crate::ir::{OpKind, Program};
use crate::runner::BatchRunner;
use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

// Kind codes shared between host packing and the kernel.
const K_SET: u32 = 0;
const K_COPY: u32 = 1;
const K_NOT: u32 = 2;
const K_ABS: u32 = 3;
const K_AND: u32 = 4;
const K_OR: u32 = 5;
const K_XOR: u32 = 6;
const K_ADD: u32 = 7;
const K_SUB: u32 = 8;
const K_MUL: u32 = 9;
const K_DIV: u32 = 10;
const K_EQ: u32 = 11;
const K_NE: u32 = 12;
const K_LT: u32 = 13;
const K_LE: u32 = 14;
const K_GT: u32 = 15;
const K_GE: u32 = 16;
const K_SELECT: u32 = 17;

#[cube]
fn cell_get(cells: &Array<u32>, base: usize, idx: u32) -> u32 {
    let w = cells[base + (idx / 4u32) as usize];
    (w >> ((idx % 4u32) * 8u32)) & 0xFFu32
}

#[cube]
fn cell_set(cells: &mut Array<u32>, base: usize, idx: u32, v: u32) {
    let wi = base + (idx / 4u32) as usize;
    let sh = (idx % 4u32) * 8u32;
    let old = cells[wi];
    cells[wi] = (old & !(0xFFu32 << sh)) | ((v & 0xFFu32) << sh);
}

/// Signed less-than on 8-bit interpretations, in pure u32: sign-extend
/// both bytes to bit 31, flip the sign bit, compare unsigned.
#[cube]
fn signed_lt(x: u32, y: u32) -> bool {
    let sx = (x << 24u32) ^ 0x8000_0000u32;
    let sy = (y << 24u32) ^ 0x8000_0000u32;
    sx < sy
}

#[cube]
fn select_u32(fales: u32, tru: u32, cond: bool) -> u32 {
    let mut out = fales;
    if cond {
        out = tru;
    }
    out
}

#[cube]
fn bool_u32(b: bool) -> u32 {
    select_u32(0u32, 1u32, b)
}

#[cube(launch_unchecked)]
fn dally_kernel(
    ops: &Array<u32>,
    in_addrs: &Array<u32>,
    out_addrs: &Array<u32>,
    inputs: &Array<u32>,
    cells: &mut Array<u32>,
    outputs: &mut Array<u32>,
    traps: &mut Array<u32>,
    n_instances: usize,
    n_ops: usize,
    n_in: usize,
    n_out: usize,
    cell_words: usize,
) {
    let inst = ABSOLUTE_POS;
    if inst >= n_instances {
        terminate!();
    }
    let base = inst * cell_words;

    for i in range(0usize, cell_words) {
        cells[base + i] = 0u32;
    }
    for j in range(0usize, n_in) {
        let flat = inst * n_in + j;
        let w = inputs[flat / 4usize];
        let b = (w >> (((flat % 4usize) * 8usize) as u32)) & 0xFFu32;
        let a = in_addrs[j];
        cell_set(cells, base, a, b);
    }

    let mut trap = 0u32;
    for k in range(0usize, n_ops) {
        if trap == 0u32 {
            let o0 = ops[k * 4usize];
            let o1 = ops[k * 4usize + 1usize];
            let o2 = ops[k * 4usize + 2usize];
            let kind = o0 & 0xFFu32;
            let imm = (o0 >> 8usize) & 0xFFu32;
            let dst = o0 >> 16u32;
            let a = o1 & 0xFFFFu32;
            let b = o1 >> 16u32;
            let c = o2 & 0xFFFFu32;

            if kind == K_SET {
                cell_set(cells, base, dst, imm);
            } else if kind == K_COPY {
                cell_set(cells, base, dst, cell_get(cells, base, a));
            } else if kind == K_NOT {
                cell_set(cells, base, dst, cell_get(cells, base, a) ^ 0xFFu32);
            } else if kind == K_ABS {
                let x = cell_get(cells, base, a);
                let s = x >> 7u32;
                cell_set(
                    cells,
                    base,
                    dst,
                    select_u32(x, (!x + 1u32) & 0xFFu32, s == 1u32),
                );
            } else if kind == K_AND {
                cell_set(
                    cells,
                    base,
                    dst,
                    cell_get(cells, base, a) & cell_get(cells, base, b),
                );
            } else if kind == K_OR {
                cell_set(
                    cells,
                    base,
                    dst,
                    cell_get(cells, base, a) | cell_get(cells, base, b),
                );
            } else if kind == K_XOR {
                cell_set(
                    cells,
                    base,
                    dst,
                    cell_get(cells, base, a) ^ cell_get(cells, base, b),
                );
            } else if kind == K_ADD {
                cell_set(
                    cells,
                    base,
                    dst,
                    cell_get(cells, base, a) + cell_get(cells, base, b),
                );
            } else if kind == K_SUB {
                cell_set(
                    cells,
                    base,
                    dst,
                    cell_get(cells, base, a) - cell_get(cells, base, b),
                );
            } else if kind == K_MUL {
                cell_set(
                    cells,
                    base,
                    dst,
                    cell_get(cells, base, a) * cell_get(cells, base, b),
                );
            } else if kind == K_DIV {
                // floor division on signed interpretations via
                // magnitudes; division by zero traps
                let x = cell_get(cells, base, a);
                let y = cell_get(cells, base, b);
                if y == 0u32 {
                    trap = 1u32;
                } else {
                    let xs = x >> 7u32;
                    let ys = y >> 7u32;
                    let xm = select_u32(x & 0x7Fu32, (!x + 1u32) & 0xFFu32, xs == 1u32);
                    let ym = select_u32(y & 0x7Fu32, (!y + 1u32) & 0xFFu32, ys == 1u32);
                    let mut q = xm / ym;
                    let r = xm % ym;
                    if r != 0u32 && xs != ys {
                        q += 1u32;
                    }
                    let neg = xs != ys;
                    cell_set(cells, base, dst, select_u32(q, (!q + 1u32) & 0xFFu32, neg));
                }
            } else if kind == K_EQ {
                cell_set(
                    cells,
                    base,
                    dst,
                    bool_u32(cell_get(cells, base, a) == cell_get(cells, base, b)),
                );
            } else if kind == K_NE {
                cell_set(
                    cells,
                    base,
                    dst,
                    bool_u32(cell_get(cells, base, a) != cell_get(cells, base, b)),
                );
            } else if kind == K_LT {
                cell_set(
                    cells,
                    base,
                    dst,
                    bool_u32(signed_lt(
                        cell_get(cells, base, a),
                        cell_get(cells, base, b),
                    )),
                );
            } else if kind == K_LE {
                let l = signed_lt(cell_get(cells, base, b), cell_get(cells, base, a));
                cell_set(cells, base, dst, bool_u32(!l));
            } else if kind == K_GT {
                cell_set(
                    cells,
                    base,
                    dst,
                    bool_u32(signed_lt(
                        cell_get(cells, base, b),
                        cell_get(cells, base, a),
                    )),
                );
            } else if kind == K_GE {
                let l = signed_lt(cell_get(cells, base, a), cell_get(cells, base, b));
                cell_set(cells, base, dst, bool_u32(!l));
            } else if kind == K_SELECT {
                let cond = cell_get(cells, base, a);
                let v = select_u32(
                    cell_get(cells, base, c),
                    cell_get(cells, base, b),
                    cond != 0u32,
                );
                cell_set(cells, base, dst, v);
            } else {
                trap = 2u32;
            }
        }
    }

    traps[inst] = trap;
    for j in range(0usize, n_out) {
        let flat = inst * n_out + j;
        let wi = flat / 4usize;
        let wsh = ((flat % 4usize) * 8usize) as u32;
        let v = cell_get(cells, base, out_addrs[j]);
        outputs[wi] |= v << wsh;
    }
}

fn op_kind_code(k: OpKind) -> u32 {
    match k {
        OpKind::Set => K_SET,
        OpKind::Copy => K_COPY,
        OpKind::Not => K_NOT,
        OpKind::Abs => K_ABS,
        OpKind::And => K_AND,
        OpKind::Or => K_OR,
        OpKind::Xor => K_XOR,
        OpKind::Add => K_ADD,
        OpKind::Sub => K_SUB,
        OpKind::Mul => K_MUL,
        OpKind::Div => K_DIV,
        OpKind::CmpEq => K_EQ,
        OpKind::CmpNe => K_NE,
        OpKind::CmpLt => K_LT,
        OpKind::CmpLe => K_LE,
        OpKind::CmpGt => K_GT,
        OpKind::CmpGe => K_GE,
        OpKind::Select => K_SELECT,
    }
}

fn pack_ops(prog: &Program) -> Vec<u32> {
    let mut v = Vec::with_capacity(prog.len() * 4);
    for i in 0..prog.len() {
        v.push(
            op_kind_code(prog.kinds[i])
                | ((prog.imm[i] as u32) << 8)
                | ((prog.dst[i] as u32) << 16),
        );
        v.push(prog.a[i] as u32 | ((prog.b[i] as u32) << 16));
        v.push(prog.c[i] as u32);
        v.push(0);
    }
    v
}

fn pack_bytes(bytes: &[u8]) -> Vec<u32> {
    let mut words = vec![0u32; bytes.len().div_ceil(4)];
    for (i, &b) in bytes.iter().enumerate() {
        words[i / 4] |= (b as u32) << ((i % 4) * 8);
    }
    words
}

fn unpack_bytes(words: &[u32], n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| ((words[i / 4] >> ((i % 4) * 8)) & 0xFF) as u8)
        .collect()
}

/// CubeCL-backed GPU runner with static buffer reuse: the op stream and
/// address tables stay resident on the device for the runner's
/// lifetime; per batch only the instance matrix uploads and the
/// outputs/traps read back.
pub struct CubeRunner {
    client: ComputeClient<cubecl_wgpu::WgpuRuntime>,
    ops: Handle,
    in_addrs: Handle,
    out_addrs: Handle,
    n_ops: usize,
    n_in: usize,
    n_out: usize,
    cell_words: usize,
    capacity: usize,
    cells: Handle,
    traps: Handle,
}

impl CubeRunner {
    pub fn new(prog: &Program, capacity: usize) -> Result<Self, RunError> {
        // cubecl panics (rather than errors) when no adapter exists;
        // catch it so headless environments soft-skip instead of abort.
        let init = std::panic::catch_unwind(|| {
            let device = cubecl_wgpu::WgpuDevice::default();
            <cubecl_wgpu::WgpuRuntime as cubecl::Runtime>::client(&device)
        })
        .map_err(|e| {
            RunError::GpuUnavailable(
                e.downcast_ref::<String>()
                    .cloned()
                    .unwrap_or_else(|| "cubecl init panicked (no adapter?)".into()),
            )
        })?;
        let client = init;
        let ops = client.create_from_slice(bytemuck::cast_slice(&pack_ops(prog)));
        let in_addrs = client.create_from_slice(bytemuck::cast_slice(
            &prog.inputs.iter().map(|&x| x as u32).collect::<Vec<_>>(),
        ));
        let out_addrs = client.create_from_slice(bytemuck::cast_slice(
            &prog.outputs.iter().map(|&x| x as u32).collect::<Vec<_>>(),
        ));
        let cell_words = (prog.max_addr as usize + 1).div_ceil(4);
        let cells = client.empty(capacity * cell_words * 4);
        let traps = client.empty(capacity * 4);
        Ok(Self {
            client,
            ops,
            in_addrs,
            out_addrs,
            n_ops: prog.len(),
            n_in: prog.inputs.len(),
            n_out: prog.outputs.len(),
            cell_words,
            capacity,
            cells,
            traps,
        })
    }

    /// Grow-only outputs reset is handled by the kernel (`|=` into a
    /// zeroed word requires the buffer to start zeroed): allocate the
    /// outputs buffer zeroed via create_from_slice of zeros.
    pub fn run(
        &self,
        prog: &Program,
        instances: &[u8],
        n: usize,
    ) -> Result<Vec<Vec<u8>>, (usize, RunError)> {
        let width = prog.inputs.len();
        debug_assert_eq!(instances.len(), n * width);
        if n > self.capacity {
            return Err((
                0,
                RunError::GpuUnavailable(format!(
                    "batch {n} exceeds runner capacity {}",
                    self.capacity
                )),
            ));
        }
        let inputs = self
            .client
            .create_from_slice(bytemuck::cast_slice(&pack_bytes(instances)));
        // outputs accumulate with |=, so hand the kernel a zeroed view
        let out_words = (n * prog.outputs.len()).div_ceil(4);
        let outputs = self.client.create_from_slice(&vec![0u8; out_words * 4]);
        let groups = n.div_ceil(64) as u32;
        unsafe {
            dally_kernel::launch_unchecked::<cubecl_wgpu::WgpuRuntime>(
                &self.client,
                CubeCount::Static(groups, 1, 1),
                CubeDim::new_1d(64),
                ArrayArg::from_raw_parts(self.ops.clone(), self.n_ops * 4),
                ArrayArg::from_raw_parts(self.in_addrs.clone(), self.n_in),
                ArrayArg::from_raw_parts(self.out_addrs.clone(), self.n_out),
                ArrayArg::from_raw_parts(inputs, n * width.div_ceil(4)),
                ArrayArg::from_raw_parts(self.cells.clone(), self.capacity * self.cell_words),
                ArrayArg::from_raw_parts(outputs.clone(), out_words),
                ArrayArg::from_raw_parts(self.traps.clone(), self.capacity),
                n,
                self.n_ops,
                self.n_in,
                self.n_out,
                self.cell_words,
            );
        }
        let out_bytes = self.client.read_one_unchecked(outputs);
        let trap_bytes = self.client.read_one_unchecked(self.traps.clone());
        let mut traps: Vec<u32> = bytemuck::cast_slice(&trap_bytes).to_vec();
        traps.truncate(n);
        let flat = unpack_bytes(bytemuck::cast_slice(&out_bytes), n * prog.outputs.len());
        let mut rows = Vec::with_capacity(n);
        for (i, trap) in traps.iter().enumerate() {
            if *trap != 0 {
                // op index is not tracked on the GPU side
                return Err((
                    i,
                    RunError::DivideByZero {
                        op_index: usize::MAX,
                    },
                ));
            }
            rows.push(flat[i * prog.outputs.len()..(i + 1) * prog.outputs.len()].to_vec());
        }
        Ok(rows)
    }
}

impl BatchRunner for CubeRunner {
    fn run_batch(
        &self,
        prog: &Program,
        instances: &[u8],
        n: usize,
    ) -> Result<Vec<Vec<u8>>, (usize, RunError)> {
        if n == 0 {
            return Ok(Vec::new());
        }
        self.run(prog, instances, n)
    }
}
