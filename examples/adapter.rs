//! Builds a CubeCL client to prove a GPU adapter is reachable.
fn main() {
    let device = cubecl_wgpu::WgpuDevice::default();
    let _client = <cubecl_wgpu::WgpuRuntime as cubecl::Runtime>::client(&device);
    println!("cubecl client initialized on default device");
}
