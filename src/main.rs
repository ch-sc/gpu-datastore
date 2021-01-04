#![feature(array_methods)]
#![feature(abi_ptx)]
use arrow::array::{Array, PrimitiveBuilder};
use arrow::datatypes::Float64Type;
use gpu_datastore::common::prelude::*;
use gpu_datastore::engine::prelude::{KernelConfiguration, KernelRunner};
use rustacuda::memory::DeviceBuffer;
use std::ffi::CString;
use std::mem;

const DATA_SIZE: usize = 1 * GB as usize;

fn generate_data() -> Result<Box<[f64]>> {
    const VALUES_COUNT: usize = DATA_SIZE / mem::size_of::<f64>();

    // create an arrow primitive array for demonstration purposes
    let mut builder = PrimitiveBuilder::<Float64Type>::new(VALUES_COUNT);
    let buffer = (0..VALUES_COUNT)
        .map(|x| (x % 128) as f64)
        .collect::<Vec<f64>>()
        .into_boxed_slice();
    let validity_vector = &[true; VALUES_COUNT];
    // builder.append_value((i % 128) as f64)
    builder.append_values(&buffer, validity_vector)?;
    let array = builder.finish();

    // arrow array -> native array
    let buf_ref = &array.data_ref().buffers()[0];
    let data = unsafe { buf_ref.typed_data::<f64>().to_vec().into_boxed_slice() };
    Ok(data)
}

fn test_run_4() -> Result<()> {
    let mut runner = KernelRunner::try_new()?;
    // load different module
    let module_data = CString::new(include_str!("../resources/binary_arithmetics.ptx"))?;
    runner.load_module(&module_data)?;

    const VALUES: usize = DATA_SIZE / mem::size_of::<f64>();

    // allocate input and output buffers
    let vec_a = generate_data()?;
    runner.allocate_buffer(DeviceBuffer::from_slice(&vec_a)?);
    let vec_b = generate_data()?;
    runner.allocate_buffer(DeviceBuffer::from_slice(&vec_b)?);
    let vec_scalar_2 = &mut vec![2_f64; VALUES].into_boxed_slice();
    runner.allocate_buffer(DeviceBuffer::from_slice(&vec_scalar_2)?);
    let vec_scalar_17 = &mut vec![17_f64; VALUES].into_boxed_slice();
    runner.allocate_buffer(DeviceBuffer::from_slice(&vec_scalar_17)?);
    // output buffer
    let vec_out = &mut vec![0_f64; VALUES].into_boxed_slice();
    runner.allocate_buffer(DeviceBuffer::from_slice(vec_out)?);

    let mut config = KernelConfiguration::new("add", 512, 1024, VALUES as u32, 0);

    runner.launch_binary_arithmetics_kernel(config.clone(), 0, 1, 4)?;

    config.kernel_name = "division".to_string();
    runner.launch_binary_arithmetics_kernel(config.clone(), 4, 2, 4)?;

    config.kernel_name = "sub".to_string();
    runner.launch_binary_arithmetics_kernel(config.clone(), 4, 3, 4)?;

    runner.collect_result(2, vec_out, 128)?;

    dbg!(vec_out);

    Ok(())
}

fn main() -> Result<()> {
    test_run_4()
}
