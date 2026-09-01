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
| 1,000 | 29.8 ms | 33,506 |
| 10,000 | 119.8 ms | 83,478 |
| 50,000 | 777.6 ms | 64,297 |

Honest reading: the GPU path no longer rebuilds its kernel or op
buffers per call (the CubeCL port removed that overhead; 1k batches
improved from 21.7k to 33.5k inst/s), but readback still dominates and
the 16-thread CPU runner remains ahead at these batch sizes (208k
inst/s). The kernel itself is now portable Rust - the same source
targets WGSL, SPIR-V, MSL, and CUDA via CubeCL's runtime compilation.

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
