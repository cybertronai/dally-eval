//! LDS (workgroup shared memory) compute kernel: per-instance cell state
//! lives in shared memory at ~1-2 cycle latency instead of global
//! memory, collapsing the dependent-load chain that made the global
//! kernel latency-bound.
//!
//! Tiling is parameterized by the adapter's limits at runner
//! construction:
//! `lanes = prev_pow2(max_shared_memory_bytes / instance_cell_bytes)`,
//! clamped to `[1, max_units_per_cube]`. Power-of-two alignment keeps
//! the workgroup a clean subdivision of the device wavefront (Wave32 /
//! Wave64 / Warp32); sub-wave workgroups trade slot efficiency for the
//! LDS fit, which is the right trade for a latency-bound kernel. Each
//! lane owns a private `cell_words`-wide slice of the workgroup's
//! shared allocation, so no barriers are needed.
//!
//! Programs whose cell array alone exceeds the LDS budget are rejected
//! at construction; callers fall back to the global-memory
//! [`CubeRunner`](crate::CubeRunner).

use crate::eval::RunError;
use crate::ir::{OpKind, Program};
use crate::runner::BatchRunner;
use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

#[cube]
fn lds_get(lds: &SharedMemory<u32>, base: usize, idx: u32) -> u32 {
    let w = lds[base + (idx / 4u32) as usize];
    (w >> ((idx % 4u32) * 8u32)) & 0xFFu32
}

#[cube]
fn lds_set(lds: &mut SharedMemory<u32>, base: usize, idx: u32, v: u32) {
    let wi = base + (idx / 4u32) as usize;
    let sh = (idx % 4u32) * 8u32;
    let old = lds[wi];
    lds[wi] = (old & !(0xFFu32 << sh)) | ((v & 0xFFu32) << sh);
}

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
fn dally_lds_kernel(
    ops: &Array<u32>,
    in_addrs: &Array<u32>,
    out_addrs: &Array<u32>,
    inputs: &Array<u32>,
    outputs: &mut Array<u32>,
    traps: &mut Array<u32>,
    expected: &Array<u32>,
    flags: &mut Array<u32>,
    n_instances: usize,
    n_ops: usize,
    n_in: usize,
    n_out: usize,
    #[comptime] cell_words: usize,
    #[comptime] lanes: usize,
    mode: u32,
) {
    // one workgroup handles `lanes` instances; lane = UNIT_POS
    let inst = (CUBE_POS_X * lanes as u32 + UNIT_POS) as usize;
    if inst >= n_instances {
        terminate!();
    }
    let base = UNIT_POS as usize * cell_words;
    let mut lds = SharedMemory::<u32>::new(lanes * cell_words);

    // zero this lane's slice
    for i in range(0usize, cell_words) {
        lds[base + i] = 0u32;
    }
    // load inputs into declared cells (byte-packed instance rows)
    for j in range(0usize, n_in) {
        let flat = inst * n_in + j;
        let w = inputs[flat / 4usize];
        let b = (w >> (((flat % 4usize) * 8usize) as u32)) & 0xFFu32;
        let a = in_addrs[j];
        lds_set(&mut lds, base, a, b);
    }

    let mut trap = 0u32;
    for k in range(0usize, n_ops) {
        if trap == 0u32 {
            let o0 = ops[k * 4usize];
            let o1 = ops[k * 4usize + 1usize];
            let o2 = ops[k * 4usize + 2usize];
            let kind = o0 & 0xFFu32;
            let imm = (o0 >> 8u32) & 0xFFu32;
            let dst = o0 >> 16u32;
            let a = o1 & 0xFFFFu32;
            let b = o1 >> 16u32;
            let c = o2 & 0xFFFFu32;

            if kind == 0u32 {
                lds_set(&mut lds, base, dst, imm);
            } else if kind == 1u32 {
                let v = lds_get(&lds, base, a);
                lds_set(&mut lds, base, dst, v);
            } else if kind == 2u32 {
                let v = lds_get(&lds, base, a) ^ 0xFFu32;
                lds_set(&mut lds, base, dst, v);
            } else if kind == 3u32 {
                let x = lds_get(&lds, base, a);
                let s = x >> 7u32;
                lds_set(
                    &mut lds,
                    base,
                    dst,
                    select_u32(x, (!x + 1u32) & 0xFFu32, s == 1u32),
                );
            } else if kind == 4u32 {
                let x = lds_get(&lds, base, a);
                let y = lds_get(&lds, base, b);
                lds_set(&mut lds, base, dst, x & y);
            } else if kind == 5u32 {
                let x = lds_get(&lds, base, a);
                let y = lds_get(&lds, base, b);
                lds_set(&mut lds, base, dst, x | y);
            } else if kind == 6u32 {
                let x = lds_get(&lds, base, a);
                let y = lds_get(&lds, base, b);
                lds_set(&mut lds, base, dst, x ^ y);
            } else if kind == 7u32 {
                let x = lds_get(&lds, base, a);
                let y = lds_get(&lds, base, b);
                lds_set(&mut lds, base, dst, x + y);
            } else if kind == 8u32 {
                let x = lds_get(&lds, base, a);
                let y = lds_get(&lds, base, b);
                lds_set(&mut lds, base, dst, x - y);
            } else if kind == 9u32 {
                let x = lds_get(&lds, base, a);
                let y = lds_get(&lds, base, b);
                lds_set(&mut lds, base, dst, x * y);
            } else if kind == 10u32 {
                let x = lds_get(&lds, base, a);
                let y = lds_get(&lds, base, b);
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
                    lds_set(
                        &mut lds,
                        base,
                        dst,
                        select_u32(q, (!q + 1u32) & 0xFFu32, neg),
                    );
                }
            } else if kind == 11u32 {
                let v = lds_get(&lds, base, a) == lds_get(&lds, base, b);
                lds_set(&mut lds, base, dst, bool_u32(v));
            } else if kind == 12u32 {
                let v = lds_get(&lds, base, a) != lds_get(&lds, base, b);
                lds_set(&mut lds, base, dst, bool_u32(v));
            } else if kind == 13u32 {
                let v = signed_lt(lds_get(&lds, base, a), lds_get(&lds, base, b));
                lds_set(&mut lds, base, dst, bool_u32(v));
            } else if kind == 14u32 {
                let v = signed_lt(lds_get(&lds, base, b), lds_get(&lds, base, a));
                lds_set(&mut lds, base, dst, bool_u32(!v));
            } else if kind == 15u32 {
                let v = signed_lt(lds_get(&lds, base, b), lds_get(&lds, base, a));
                lds_set(&mut lds, base, dst, bool_u32(v));
            } else if kind == 16u32 {
                let v = signed_lt(lds_get(&lds, base, a), lds_get(&lds, base, b));
                lds_set(&mut lds, base, dst, bool_u32(!v));
            } else if kind == 17u32 {
                let cond = lds_get(&lds, base, a);
                let v = select_u32(lds_get(&lds, base, c), lds_get(&lds, base, b), cond != 0u32);
                lds_set(&mut lds, base, dst, v);
            } else {
                trap = 2u32;
            }
        }
    }

    traps[inst] = trap;
    if mode == 0u32 {
        for j in range(0usize, n_out) {
            let flat = inst * n_out + j;
            let wi = flat / 4usize;
            let wsh = ((flat % 4usize) * 8usize) as u32;
            let v = lds_get(&lds, base, out_addrs[j]);
            outputs[wi] |= v << wsh;
        }
    } else {
        let mut ok = 1u32;
        if trap != 0u32 {
            ok = 0u32;
        }
        for j in range(0usize, n_out) {
            let flat = inst * n_out + j;
            let eb = (expected[flat / 4usize] >> (((flat % 4usize) * 8usize) as u32)) & 0xFFu32;
            let v = lds_get(&lds, base, out_addrs[j]);
            if v != eb {
                ok = 0u32;
            }
        }
        flags[inst] = ok;
    }
}

/// Per-binding storage ceiling guard (wgpu maxStorageBufferBindingSize
/// is commonly 128 MiB; stay well under, honoring the workspace VRAM
/// pre-calculation invariant).
const MAX_CELLS_BUFFER: usize = 96 * 1024 * 1024;

/// Largest batch whose flat cells buffer stays under the binding cap.
fn chunk_capacity(cell_bytes_per_instance: usize, reserved: usize) -> usize {
    let budget = MAX_CELLS_BUFFER.saturating_sub(reserved);
    (budget / cell_bytes_per_instance.max(1)).max(1)
}

/// Cap per-dispatch work: very-long-op programs (hundreds of thousands
/// of dependent ops per instance) hang the GPU when too many lanes are
/// dispatched at once (device-lost on 709k ops x 34k lanes, observed
/// on RDNA2). Coarse guard: chunk_lanes * n_ops <= WORK_CAP.
const WORK_CAP: usize = 8_000_000_000;

fn chunk_for(n_ops: usize, cell_bytes_per_instance: usize, reserved: usize) -> usize {
    let by_bytes = chunk_capacity(cell_bytes_per_instance, reserved);
    let by_work = (WORK_CAP / n_ops.max(1)).max(1);
    by_bytes.min(by_work)
}

fn prev_pow2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    1 << (usize::BITS - 1 - n.leading_zeros())
}

/// Workgroup tiling derived from adapter limits: shared-memory budget
/// divided by per-instance cell bytes, power-of-two aligned (a clean
/// wavefront subdivision), clamped to the device's workgroup size.
#[derive(Clone, Copy, Debug)]
pub struct Tiling {
    pub lanes: usize,
    pub cell_words: usize,
    pub lds_bytes: usize,
    pub max_lds_bytes: usize,
    pub max_workgroup: usize,
}

impl Tiling {
    pub fn from_limits(max_lds_bytes: usize, max_workgroup: usize, cell_bytes: usize) -> Self {
        let cell_words = cell_bytes.div_ceil(4);
        let by_budget = (max_lds_bytes / 4) / cell_words.max(1);
        let lanes = prev_pow2(by_budget).clamp(1, max_workgroup.max(1));
        Tiling {
            lanes,
            cell_words,
            lds_bytes: lanes * cell_words * 4,
            max_lds_bytes,
            max_workgroup,
        }
    }
}

fn op_kind_code(k: OpKind) -> u32 {
    use OpKind::*;
    match k {
        Set => 0,
        Copy => 1,
        Not => 2,
        Abs => 3,
        And => 4,
        Or => 5,
        Xor => 6,
        Add => 7,
        Sub => 8,
        Mul => 9,
        Div => 10,
        CmpEq => 11,
        CmpNe => 12,
        CmpLt => 13,
        CmpLe => 14,
        CmpGt => 15,
        CmpGe => 16,
        Select => 17,
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

/// Global buffers stay resident exactly like [`CubeRunner`]; the
/// difference is the kernel's cell array lives in workgroup shared
/// memory.
pub struct LdsRunner {
    client: ComputeClient<cubecl_wgpu::WgpuRuntime>,
    tiling: Tiling,
    ops: Handle,
    in_addrs: Handle,
    out_addrs: Handle,
    n_ops: usize,
    n_in: usize,
    n_out: usize,
    capacity: usize,
    traps: Handle,
    flags: Handle,
}

impl LdsRunner {
    /// Fails with [`RunError::GpuUnavailable`] when no adapter exists,
    /// or when the program's cell array exceeds the entire LDS budget
    /// (single lane still does not fit) - fall back to `CubeRunner`.
    pub fn new(prog: &Program, capacity: usize) -> Result<Self, RunError> {
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
        let props = client.properties().clone();
        let (max_lds, max_units) = (
            props.hardware.max_shared_memory_size,
            props.hardware.max_units_per_cube as usize,
        );
        let cell_bytes = prog.max_addr as usize + 1;
        let tiling = Tiling::from_limits(max_lds, max_units, cell_bytes);
        if tiling.cell_words * 4 > max_lds {
            return Err(RunError::GpuUnavailable(format!(
                "program cells ({} bytes) exceed LDS budget ({} bytes); use CubeRunner",
                tiling.cell_words * 4,
                max_lds
            )));
        }
        let ops = client.create_from_slice(bytemuck::cast_slice(&pack_ops(prog)));
        let in_addrs = client.create_from_slice(bytemuck::cast_slice(
            &prog.inputs.iter().map(|&x| x as u32).collect::<Vec<_>>(),
        ));
        let out_addrs = client.create_from_slice(bytemuck::cast_slice(
            &prog.outputs.iter().map(|&x| x as u32).collect::<Vec<_>>(),
        ));
        // persistent per-instance buffers are sized to the chunk
        // capacity, not the requested capacity: each dispatch only
        // touches m <= chunk instances, and capacity-sized buffers
        // would blow the binding limit for 100k-instance runners on
        // the big record programs
        let chunk = chunk_for(prog.len(), tiling.cell_words * 4, 0).min(capacity);
        let traps = client.create_from_slice(&vec![0u8; chunk * 4]);
        let flags = client.create_from_slice(&vec![0u8; chunk * 4]);
        Ok(Self {
            client,
            tiling,
            ops,
            in_addrs,
            out_addrs,
            n_ops: prog.len(),
            n_in: prog.inputs.len(),
            n_out: prog.outputs.len(),
            capacity: chunk,
            traps,
            flags,
        })
    }

    pub fn tiling(&self) -> &Tiling {
        &self.tiling
    }

    /// Constructor with an artificial LDS budget cap (minimum of the
    /// device budget and `cap_bytes`), for occupancy experiments: a
    /// smaller tiling fits more workgroups per compute unit.
    pub fn with_lds_cap(
        prog: &Program,
        capacity: usize,
        cap_bytes: usize,
    ) -> Result<Self, RunError> {
        let mut r = Self::new(prog, capacity)?;
        let cell_bytes = prog.max_addr as usize + 1;
        r.tiling = Tiling::from_limits(
            r.tiling.max_lds_bytes.min(cap_bytes),
            r.tiling.max_workgroup,
            cell_bytes,
        );
        Ok(r)
    }

    fn check_capacity(&self, n: usize) -> Result<(), (usize, RunError)> {
        // chunked dispatch handles any n; persistent buffers are
        // chunk-sized and re-used per sub-batch
        let _ = n;
        Ok(())
    }

    pub fn run(
        &self,
        prog: &Program,
        instances: &[u8],
        n: usize,
    ) -> Result<Vec<Vec<u8>>, (usize, RunError)> {
        let width = prog.inputs.len();
        debug_assert_eq!(instances.len(), n * width);
        self.check_capacity(n)?;
        // chunked dispatch: each sub-batch's cells/outputs bindings
        // stay under the storage cap, so 100k+ instance batches on the
        // big record programs dispatch transparently (VRAM
        // pre-calculation invariant)
        let cell_bytes = self.tiling.cell_words * 4;
        let chunk = chunk_for(self.n_ops, cell_bytes, 0).min(n);
        let mut rows_all = Vec::with_capacity(n);
        let mut start = 0usize;
        while start < n {
            let end = (start + chunk).min(n);
            let m = end - start;
            let sub_in = &instances[start * width..end * width];
            let inputs = self
                .client
                .create_from_slice(bytemuck::cast_slice(&pack_bytes(sub_in)));
            let out_words = m * prog.outputs.len().div_ceil(4);
            let outputs = self.client.create_from_slice(&vec![0u8; out_words * 4]);
            let groups = m.div_ceil(self.tiling.lanes) as u32;
            unsafe {
                dally_lds_kernel::launch_unchecked::<cubecl_wgpu::WgpuRuntime>(
                    &self.client,
                    CubeCount::Static(groups, 1, 1),
                    CubeDim::new_1d(self.tiling.lanes as u32),
                    ArrayArg::from_raw_parts(self.ops.clone(), self.n_ops * 4),
                    ArrayArg::from_raw_parts(self.in_addrs.clone(), self.n_in),
                    ArrayArg::from_raw_parts(self.out_addrs.clone(), self.n_out),
                    ArrayArg::from_raw_parts(inputs, m * width.div_ceil(4)),
                    ArrayArg::from_raw_parts(outputs.clone(), out_words),
                    ArrayArg::from_raw_parts(self.traps.clone(), self.capacity),
                    ArrayArg::from_raw_parts(outputs.clone(), out_words),
                    ArrayArg::from_raw_parts(self.flags.clone(), self.capacity),
                    m,
                    self.n_ops,
                    self.n_in,
                    self.n_out,
                    self.tiling.cell_words,
                    self.tiling.lanes,
                    0u32,
                );
            }
            let both = self.client.read(vec![outputs, self.traps.clone()]);
            match self.collect_raw(prog, &both[0], &both[1], m) {
                Ok(mut r) => rows_all.append(&mut r),
                Err((i, e)) => return Err((start + i, e)),
            }
            start = end;
        }
        Ok(rows_all)
    }

    /// Scored mode: grading on device against a pre-uploaded answer
    /// key; one flag word per instance comes back.
    pub fn run_scored(
        &self,
        prog: &Program,
        instances: &[u8],
        expected_flat: &[u8],
        n: usize,
    ) -> Result<usize, (usize, RunError)> {
        let width = prog.inputs.len();
        let out_w = prog.outputs.len();
        debug_assert_eq!(instances.len(), n * width);
        debug_assert_eq!(expected_flat.len(), n * out_w);
        self.check_capacity(n)?;
        let cell_bytes = self.tiling.cell_words * 4;
        let chunk = chunk_for(self.n_ops, cell_bytes, 0).min(n);
        let mut correct = 0usize;
        let mut start = 0usize;
        while start < n {
            let end = (start + chunk).min(n);
            let m = end - start;
            let sub_in = &instances[start * width..end * width];
            let sub_exp = &expected_flat[start * out_w..end * out_w];
            let inputs = self
                .client
                .create_from_slice(bytemuck::cast_slice(&pack_bytes(sub_in)));
            let expected = self
                .client
                .create_from_slice(bytemuck::cast_slice(&pack_bytes(sub_exp)));
            let groups = m.div_ceil(self.tiling.lanes) as u32;
            unsafe {
                dally_lds_kernel::launch_unchecked::<cubecl_wgpu::WgpuRuntime>(
                    &self.client,
                    CubeCount::Static(groups, 1, 1),
                    CubeDim::new_1d(self.tiling.lanes as u32),
                    ArrayArg::from_raw_parts(self.ops.clone(), self.n_ops * 4),
                    ArrayArg::from_raw_parts(self.in_addrs.clone(), self.n_in),
                    ArrayArg::from_raw_parts(self.out_addrs.clone(), self.n_out),
                    ArrayArg::from_raw_parts(inputs, m * width.div_ceil(4)),
                    ArrayArg::from_raw_parts(expected.clone(), m * out_w.div_ceil(4)),
                    ArrayArg::from_raw_parts(self.traps.clone(), self.capacity),
                    ArrayArg::from_raw_parts(expected.clone(), m * out_w.div_ceil(4)),
                    ArrayArg::from_raw_parts(self.flags.clone(), self.capacity),
                    m,
                    self.n_ops,
                    self.n_in,
                    self.n_out,
                    self.tiling.cell_words,
                    self.tiling.lanes,
                    1u32,
                );
            }
            let both = self
                .client
                .read(vec![self.flags.clone(), self.traps.clone()]);
            let flags: Vec<u32> = bytemuck::cast_slice(&both[0]).to_vec();
            let mut traps: Vec<u32> = bytemuck::cast_slice(&both[1]).to_vec();
            traps.truncate(m);
            for (j, trap) in traps.iter().enumerate() {
                if *trap != 0 {
                    return Err((
                        start + j,
                        RunError::DivideByZero {
                            op_index: usize::MAX,
                        },
                    ));
                }
            }
            correct += flags.iter().take(m).filter(|&&f| f == 1).count();
            start = end;
        }
        Ok(correct)
    }

    fn collect_raw(
        &self,
        prog: &Program,
        out_bytes: &[u8],
        trap_bytes: &[u8],
        n: usize,
    ) -> Result<Vec<Vec<u8>>, (usize, RunError)> {
        let mut traps: Vec<u32> = bytemuck::cast_slice(trap_bytes).to_vec();
        traps.truncate(n);
        for (i, trap) in traps.iter().enumerate() {
            if *trap != 0 {
                return Err((
                    i,
                    RunError::DivideByZero {
                        op_index: usize::MAX,
                    },
                ));
            }
        }
        let flat = unpack_bytes(bytemuck::cast_slice(out_bytes), n * prog.outputs.len());
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            rows.push(flat[i * prog.outputs.len()..(i + 1) * prog.outputs.len()].to_vec());
        }
        Ok(rows)
    }
}

impl BatchRunner for LdsRunner {
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
