use gpu_datastore::engine::prelude::KernelLauncher;

use gpu_datastore::common::prelude::*;
use rustacuda::memory::DeviceBuffer;
use std::ffi::CString;

fn main() -> Result<()> {
    let mut launcher = KernelLauncher::try_new()?;
    let result = launcher.launch_add_kernel()?;

    println!("result 1: {:?}", result);

    // File needs to exist at compile time.
    // Relative path starts from this file's directory
    let module_data = CString::new(include_str!("../resources/add.ptx"))?;
    launcher.load_file(&module_data)?;
    launcher.allocate_buffer(DeviceBuffer::from_slice(&[1.0f64; 10])?); // in 1
    launcher.allocate_buffer(DeviceBuffer::from_slice(&[2.0f64; 10])?); // in 2
    launcher.allocate_buffer(DeviceBuffer::from_slice(&[0.0f64; 10])?); // out 1
    launcher.allocate_buffer(DeviceBuffer::from_slice(&[0.0f64; 10])?); // out 2
    let result = launcher.launch()?;

    println!("result 2: {:?}", result);

    Ok(())
}
