#![feature(array_methods)]

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
    launcher.launch()?;

    let mut out_host = [0.0f64; 10];
    launcher.collect_result(2, out_host.as_mut_slice(), 10)?;
    for x in out_host.iter() {
        assert_eq!(3_f64, *x)
    }
    launcher.collect_result(3, out_host.as_mut_slice(), 10)?;
    for x in out_host.iter() {
        assert_eq!(3_f64, *x)
    }

    println!("result 2: {:?}", out_host);

    Ok(())
}
