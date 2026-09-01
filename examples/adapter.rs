fn main() {
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        match instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await {
            Some(a) => {
                let info = a.get_info();
                println!("adapter: {:?} ({:?})", info.name, info.backend);
            }
            None => println!("no adapter"),
        }
    });
}
