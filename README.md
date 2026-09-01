# dally-eval

High-throughput evaluator for Bill Dally 2D Manhattan communication-cost
IR programs: the compute core of the sparse-parity benchmark family.
Semantics are bit-exact with the Python reference
(`sutro-problems/sparse-parity/mask_sparse_parity.py`), validated by
golden fixtures generated from that engine.

## Cost model

Processor at the origin of an upper-half-plane memory grid; cell `idx`
sits at distance `ceil(sqrt(idx))` (computed as `isqrt(idx-1)+1`).
Operand reads are priced, writes and arithmetic are free, and each
declared output cell is charged one final read. Program cost is static:
the sum over the op stream. Binary ops accept a 3-operand form
(`op dst,s1,s2`) and a 2-operand accumulator form (`op dst,s2`) whose
first source is the destination (also charged as a read), matching the
reference parser.

Execution semantics: 8-bit cells; `add`/`sub`/`mul` wrap; `div` is
floor division on signed interpretations and traps on zero (surfaces
as `RunError::DivideByZero` per instance); compares are signed;
`select dst,c,x,y` is `c != 0 ? x : y`; every write stores the
two's-complement byte.

## Layout

- `src/cost.rs` - exact integer distance pricing (no LUT; `isqrt` is a
  few cycles and a table would cost more cache than it saves)
- `src/ir.rs` - SoA flat buffers (`Vec<u16>` addresses, `Vec<u8>`/enum
  opcodes) plus the text-IR parser
- `src/eval.rs` - zero-allocation interpreter over a reused cell arena
  and the `Checker` trait for reference-function validation
- `src/runner.rs` - `BatchRunner` trait; `CpuRunner` fans instances out
  across Rayon workers with one machine per worker
- `src/cube_runner.rs` - `CubeRunner`: the same batch contract on a
  CubeCL `#[cube]` kernel written in pure Rust (no shader text), one
  workgroup lane per instance, div-by-zero as a per-instance trap
  flag, with the op stream and address tables resident across batches
- `src/lds_runner.rs` - `LdsRunner`: cells in workgroup shared memory
  (LDS) with adapter-limit-parameterized tiling; falls back to
  `CubeRunner` for programs whose cells exceed the LDS budget
- `tests/` - golden parity (real 73,293-op benchmark program, 32 real
  instances, byte-exact expected outputs), task families (4x4 matmul,
  5-input polynomial, 16/32-bit sparse parity trees), GPU-vs-CPU parity
- `benches/throughput.rs` - criterion suite on the real program

## Setup

NixOS:
```
nix develop
cargo test
cargo bench
```

The devShell pins Rust via oxalica/rust-overlay from
`rust-toolchain.toml` (stable channel) and wires headless GPU access
(Vulkan loader and ICD discovery).

macOS (Apple Silicon: wgpu uses the Metal backend out of the box):
```
rustup default stable
cargo test
cargo bench
```
GPU tests pick up Metal automatically and soft-skip only if no adapter
is found.

Non-Nix Linux: same rustup path; the GPU backend needs a Vulkan driver
(RADV/AMD, ANV/Intel, or NVIDIA proprietary). 
## CLI

```
cargo run --release -- dally-eval cost program.ir
cargo run --release -- dally-eval run  program.ir fixture.bin
cargo run --release -- dally-eval bench program.ir 1024 10
```

## Measured (this host, Ryzen 9 9950X, release build)

Python reference: ~2.8 s per 1,024 instances of the 73k-op program.

| bench | time |
| - | - |
| parse 73k ops | 4.4 ms |
| static cost fold | 392 us |
| eval 1,024 instances, serial | 77.1 ms |
| eval 1,024 instances, Rayon | 4.9 ms (~208k inst/s) |

That is ~570x the Python engine on the same workload shape, with
byte-identical outputs.

GPU (CubeRunner: CubeCL `#[cube]` kernel in pure Rust, compiled to the
active backend at runtime; static buffer reuse - op stream and address
tables stay resident, per batch only the instance matrix uploads and
outputs/traps read back. AMD RX 6900 XT / RADV, same 73,293-op program,
bit-exact vs CPU on every batch):

| batch | per batch | instances/s |
| - | - | - |
| 1,000 | 29.6 ms | 33,772 |
| 10,000 | 117.7 ms | 84,954 |
| 50,000 | 788.1 ms | 63,443 |
| 100,000 | 1.596 s | 62,674 |

Scored mode (grading on device, one flag word per instance instead of
the full output matrix) measures within noise of raw mode at every
batch size - readback is *not* the bottleneck for the global-memory
kernel.

LDS kernel (`LdsRunner`): per-instance cells live in workgroup shared
memory (tiling parameterized from adapter limits: lanes =
prev_pow2(max_shared_memory_bytes / instance_cell_bytes); on this
adapter 32 lanes x 451 words = 57.7KB of the 64KB budget). All
dependent cell reads/writes execute at LDS latency instead of global
memory latency:

| batch | global kernel | LDS kernel | |
| - | - | - | - |
| 1,000 | 33.4k inst/s | 68.4k inst/s | 2.1x |
| 10,000 | 84.2k | 160.4k | 1.9x |
| 50,000 | 64.2k | 157.6k | 2.5x |
| 100,000 | 63.0k | 162.5k | 2.6x |

An occupancy sweep (lanes 32/16/8/4, forcing 1-8 workgroups per CU)
measures flat within noise: the LDS kernel is not occupancy-bound but
per-op dispatch-bound (the 17-way opcode else-if chain costs ~400+
cycles/op on this backend regardless of resident waves). The next
throughput lever is dispatch restructuring (op-stream pre-bucketing by
kind or jump-table dispatch), not tiling. The 16-thread CPU runner
(208k inst/s) still leads at these batch sizes. All kernels are
portable Rust - one source targeting WGSL, SPIR-V, MSL, and CUDA via
CubeCL's runtime compilation.

## Parallelism shape

One instance is a sequential dataflow chain; instances and candidate
schedules are embarrassingly parallel. All backends consume the same
flat SoA buffers, so `CpuRunner` (Rayon) and `WgpuRunner` share one
interface and one buffer format; a WASM/component shell is a future
packaging step on the same buffers.

## Regenerating the golden fixtures

From `sutro-problems/sparse-parity`, export the current 20/40% record
program and 32 dev instances with the Python engine (see git history of
this README for the exact snippet); costs asserted in `tests/golden.rs`
must match the reference evaluator's output.

## Inner-loop search acceleration (measured)

`examples/search-sweep.rs` runs an automated candidate-layout search
(200 address permutations of the real 73k-op benchmark program, each
evaluated over 1,024 instances):

- Rust CPU (Rayon): 155 candidates/s end-to-end; the same sweep in the
  Python engine would take ~9.3 minutes (~460x slower).
- GPU LDS engine: 70 candidates/s at this batch size (dispatch-bound
  kernel; see the LDS section). At small batches the CPU engine is the
  right scorer; the GPU path exists for scale.

Sub-millisecond per-candidate evaluation is what makes autonomous
hardware search practical: a 10,000-candidate sweep finishes in ~65
seconds instead of ~8 hours.

## Systolic GEMM in the Dally IR (examples/systolic-gemm.rs)

The 2D systolic mesh's data-movement principle - stage each operand
once into a cell near its consumers, then re-read it at near-zero
distance - expressed as a straight-line Dally IR schedule, verified
bit-exact against a reference GEMM:

| size | input placement | naive cost | systolic cost | saved |
| - | - | - | - | - |
| 4x4 | addr 1+ (cheapest cells) | 2,072 | 2,846 | -37% (staging loses when inputs are already near) |
| 4x4 | addr 2001+ | 7,634 | 3,684 | 51.7% |
| 8x8 | addr 5001+ | 104,104 | 46,147 | 55.7% |
| 16x16 | addr 20001+ | 2,023,849 | 783,912 | 61.3% |

The negative control matters: when operands already live at the
cheapest addresses there is nothing to stage and the copies only add
cost. The win grows with operand distance and reuse count, which is
exactly the trade the matmul competition asks schedules to navigate.

## For AI agents working in this repo

Cold start: read this README, then `src/ir.rs` (the semantics contract
lives in `parse_op` and `Op::read_cost`), then `tests/golden.rs` (the
Python-parity contract). The golden fixtures are generated from
cybertronai/sutro-problems (sparse-parity tier, exported 2026-09-01);
`tests/golden.rs` pins their exact cost and outputs - any semantics
change that breaks them is wrong unless the reference evaluator changed
first. 
## Fixture provenance

`tests/fixtures/siswalk1_cap2.ir` is the sparse-parity benchmark's
2026-08-30 20/40%-band record program (layout-optimized static-IS walk)
and `tests/fixtures/parity32.bin` holds 32 deterministic dev-suite
instances with expected outputs from the Python reference evaluator,
both exported from cybertronai/sutro-problems on 2026-09-01.
