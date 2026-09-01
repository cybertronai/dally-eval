//! GPU compute-shader backend: the same [`BatchRunner`] contract over
//! the same flat buffers, one workgroup lane per instance.
//!
//! The interpreter kernel walks the packed op stream exactly like the
//! CPU engine. GPU-specific semantics: division by zero sets a
//! per-instance trap flag instead of unwinding; the host maps trapped
//! instances back to [`RunError::DivideByZero`].
//!
//! Batch sizes above a few thousand instances allocate GPU memory; run
//! large batches within your host's resource-management policy.

use crate::eval::RunError;
use crate::ir::{OpKind, Program};
use crate::runner::BatchRunner;

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read>       ops      : array<vec4<u32>>;
@group(0) @binding(1) var<storage, read>       in_addrs : array<u32>;
@group(0) @binding(2) var<storage, read>       out_addrs: array<u32>;
@group(0) @binding(3) var<storage, read>       inputs   : array<u32>;
@group(0) @binding(4) var<storage, read_write> cells    : array<u32>;
@group(0) @binding(5) var<storage, read_write> outputs  : array<u32>;
@group(0) @binding(6) var<storage, read_write> traps    : array<u32>;

struct Params {
    n_instances: u32,
    n_ops: u32,
    n_in: u32,
    n_out: u32,
    cell_words: u32,
};
@group(0) @binding(7) var<uniform> params : Params;

fn get_cell(base: u32, idx: u32) -> u32 {
    let w = cells[base + (idx >> 2u)];
    return (w >> ((idx & 3u) * 8u)) & 0xFFu;
}
fn set_cell(base: u32, idx: u32, v: u32) {
    let wi = base + (idx >> 2u);
    let sh = (idx & 3u) * 8u;
    let w = cells[wi];
    cells[wi] = (w & ~(0xFFu << sh)) | ((v & 0xFFu) << sh);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let inst = gid.x;
    if (inst >= params.n_instances) { return; }
    let base = inst * params.cell_words;
    for (var i: u32 = 0u; i < params.cell_words; i = i + 1u) {
        cells[base + i] = 0u;
    }
    for (var j: u32 = 0u; j < params.n_in; j = j + 1u) {
        let flat = inst * params.n_in + j;
        let w = inputs[flat >> 2u];
        let b = (w >> ((flat & 3u) * 8u)) & 0xFFu;
        set_cell(base, in_addrs[j], b);
    }
    var trap = 0u;
    for (var k: u32 = 0u; k < params.n_ops; k = k + 1u) {
        if (trap != 0u) { break; }
        let op = ops[k];
        let kind = op.x & 0xFFu;
        let imm  = (op.x >> 8u) & 0xFFu;
        let dst  = op.x >> 16u;
        let a    = op.y & 0xFFFFu;
        let b    = op.y >> 16u;
        let c    = op.z & 0xFFFFu;
        switch (kind) {
            case 0u: { set_cell(base, dst, imm); }
            case 1u: { set_cell(base, dst, get_cell(base, a)); }
            case 2u: { set_cell(base, dst, (~get_cell(base, a)) & 0xFFu); }
            case 3u: {
                let s = i32(get_cell(base, a) << 24u) >> 24;
                let v = select(s, -s, s < 0);
                set_cell(base, dst, u32(v) & 0xFFu);
            }
            case 4u: { set_cell(base, dst, get_cell(base, a) & get_cell(base, b)); }
            case 5u: { set_cell(base, dst, get_cell(base, a) | get_cell(base, b)); }
            case 6u: { set_cell(base, dst, get_cell(base, a) ^ get_cell(base, b)); }
            case 7u: { set_cell(base, dst, (get_cell(base, a) + get_cell(base, b)) & 0xFFu); }
            case 8u: { set_cell(base, dst, (get_cell(base, a) - get_cell(base, b)) & 0xFFu); }
            case 9u: { set_cell(base, dst, (get_cell(base, a) * get_cell(base, b)) & 0xFFu); }
            case 10u: {
                let x = i32(get_cell(base, a) << 24u) >> 24;
                let y = i32(get_cell(base, b) << 24u) >> 24;
                if (y == 0) { trap = 1u; } else {
                    var q = x / y;
                    let r = x % y;
                    if (r != 0 && ((r < 0) != (y < 0))) { q = q - 1; }
                    set_cell(base, dst, u32(q) & 0xFFu);
                }
            }
            case 11u: { set_cell(base, dst, u32(get_cell(base, a) == get_cell(base, b))); }
            case 12u: { set_cell(base, dst, u32(get_cell(base, a) != get_cell(base, b))); }
            case 13u: {
                let x = i32(get_cell(base, a) << 24u) >> 24;
                let y = i32(get_cell(base, b) << 24u) >> 24;
                set_cell(base, dst, u32(x < y));
            }
            case 14u: {
                let x = i32(get_cell(base, a) << 24u) >> 24;
                let y = i32(get_cell(base, b) << 24u) >> 24;
                set_cell(base, dst, u32(x <= y));
            }
            case 15u: {
                let x = i32(get_cell(base, a) << 24u) >> 24;
                let y = i32(get_cell(base, b) << 24u) >> 24;
                set_cell(base, dst, u32(x > y));
            }
            case 16u: {
                let x = i32(get_cell(base, a) << 24u) >> 24;
                let y = i32(get_cell(base, b) << 24u) >> 24;
                set_cell(base, dst, u32(x >= y));
            }
            case 17u: {
                let cond = get_cell(base, a);
                let v = select(get_cell(base, c), get_cell(base, b), cond != 0u);
                set_cell(base, dst, v);
            }
            default: { trap = 2u; }
        }
    }
    traps[inst] = trap;
    for (var j: u32 = 0u; j < params.n_out; j = j + 1u) {
        let flat = inst * params.n_out + j;
        let wi = flat >> 2u;
        let sh = (flat & 3u) * 8u;
        let v = select(0u, get_cell(base, out_addrs[j]), trap == 0u);
        outputs[wi] = (outputs[wi] & ~(0xFFu << sh)) | ((v & 0xFFu) << sh);
    }
}
"#;

fn op_kind_code(k: OpKind) -> u32 {
    match k {
        OpKind::Set => 0,
        OpKind::Copy => 1,
        OpKind::Not => 2,
        OpKind::Abs => 3,
        OpKind::And => 4,
        OpKind::Or => 5,
        OpKind::Xor => 6,
        OpKind::Add => 7,
        OpKind::Sub => 8,
        OpKind::Mul => 9,
        OpKind::Div => 10,
        OpKind::CmpEq => 11,
        OpKind::CmpNe => 12,
        OpKind::CmpLt => 13,
        OpKind::CmpLe => 14,
        OpKind::CmpGt => 15,
        OpKind::CmpGe => 16,
        OpKind::Select => 17,
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    n_instances: u32,
    n_ops: u32,
    n_in: u32,
    n_out: u32,
    cell_words: u32,
}

pub struct WgpuRunner;

impl BatchRunner for WgpuRunner {
    fn run_batch(
        &self,
        prog: &Program,
        instances: &[u8],
        n: usize,
    ) -> Result<Vec<Vec<u8>>, (usize, RunError)> {
        if n == 0 {
            return Ok(Vec::new());
        }
        pollster::block_on(run_on_gpu(prog, instances, n))
    }
}

async fn run_on_gpu(
    prog: &Program,
    instances: &[u8],
    n: usize,
) -> Result<Vec<Vec<u8>>, (usize, RunError)> {
    let gpu_err = |msg: String| (0usize, RunError::GpuUnavailable(msg));
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .ok_or_else(|| gpu_err("no suitable adapter".into()))?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .map_err(|e| gpu_err(e.to_string()))?;

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("dally-eval interpreter"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("dally-eval pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let layout = pipeline.get_bind_group_layout(0);
    use wgpu::util::DeviceExt;

    let n_in = prog.inputs.len() as u32;
    let n_out = prog.outputs.len() as u32;
    let cell_words = (prog.max_addr as u32 + 1).div_ceil(4);
    let params = GpuParams {
        n_instances: n as u32,
        n_ops: prog.len() as u32,
        n_in,
        n_out,
        cell_words,
    };

    let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let ops_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&pack_ops(prog)),
        usage: storage,
    });
    let in_addrs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&prog.inputs.iter().map(|&x| x as u32).collect::<Vec<_>>()),
        usage: storage,
    });
    let out_addrs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&prog.outputs.iter().map(|&x| x as u32).collect::<Vec<_>>()),
        usage: storage,
    });
    let inputs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&pack_bytes(instances)),
        usage: storage,
    });
    let cells_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &vec![0u8; (n as u32 * cell_words * 4) as usize],
        usage: storage,
    });
    let out_words = (n * prog.outputs.len()).div_ceil(4);
    let outputs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &vec![0u8; out_words * 4],
        usage: storage | wgpu::BufferUsages::COPY_SRC,
    });
    let traps_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &vec![0u8; n * 4],
        usage: storage | wgpu::BufferUsages::COPY_SRC,
    });
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ops_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: in_addrs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_addrs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: inputs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: cells_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: outputs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: traps_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n.div_ceil(64) as u32, 1, 1);
    }
    let out_read = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (out_words * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let traps_read = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&outputs_buf, 0, &out_read, 0, (out_words * 4) as u64);
    encoder.copy_buffer_to_buffer(&traps_buf, 0, &traps_read, 0, (n * 4) as u64);
    queue.submit(Some(encoder.finish()));

    let (out_snd, out_rcv) = std::sync::mpsc::channel();
    out_read.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = out_snd.send(r);
    });
    let _ = device.poll(wgpu::Maintain::Wait);
    let (t_snd, t_rcv) = std::sync::mpsc::channel();
    traps_read
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| {
            let _ = t_snd.send(r);
        });
    let _ = device.poll(wgpu::Maintain::Wait);

    let out_bytes = out_read.slice(..).get_mapped_range().to_vec();
    let trap_bytes = traps_read.slice(..).get_mapped_range().to_vec();
    let traps: Vec<u32> = bytemuck::cast_slice(&trap_bytes).to_vec();
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
    let _ = (out_rcv, t_rcv);
    Ok(rows)
}
