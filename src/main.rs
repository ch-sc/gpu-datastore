#![feature(array_methods)]
#![feature(abi_ptx)]
use gpu_datastore::common::prelude::*;
use gpu_datastore::engine::prelude::{KernelConfiguration, KernelRunner};

use arrow::array::{Array, PrimitiveBuilder};

use arrow::datatypes::Float64Type;
use rustacuda::memory::DeviceBuffer;
use std::ffi::CString;
use std::mem;

const DATA_SIZE: usize = 1 * GB as usize;

fn test_run_1(runner: &mut KernelRunner) -> Result<()> {
    let result = runner.launch_test_kernel()?;
    println!("result 1: {:?}", result);
    Ok(())
}

fn test_run_2(runner: &mut KernelRunner) -> Result<()> {
    // File needs to exist at compile time.
    // Relative path starts from this file's directory
    let module_data = CString::new(include_str!("../resources/add.ptx"))?;
    runner.load_module(&module_data)?;
    runner.allocate_buffer(DeviceBuffer::from_slice(&[1.0f64; 10])?); // in 1
    runner.allocate_buffer(DeviceBuffer::from_slice(&[2.0f64; 10])?); // in 2
    runner.allocate_buffer(DeviceBuffer::from_slice(&[0.0f64; 10])?); // out 1
    runner.allocate_buffer(DeviceBuffer::from_slice(&[0.0f64; 10])?); // out 2
    runner.launch_test_kernel_2()?;

    let mut out_host = [0.0f64; 10];
    runner.collect_result(2, out_host.as_mut_slice(), 10)?;
    runner.collect_result(3, out_host.as_mut_slice(), 10)?;

    println!("result 2: {:?}", out_host);
    Ok(())
}

fn generate_data() -> Result<Box<[f64]>> {
    const VALUES_COUNT: usize = DATA_SIZE / mem::size_of::<f64>();

    // create arrow primitive array
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

fn test_run_3(runner: &mut KernelRunner) -> Result<()> {
    // load different module
    let module_data = CString::new(include_str!("../resources/binary_arithmetics.ptx"))?;
    runner.load_module(&module_data)?;

    const VALUES: usize = DATA_SIZE / mem::size_of::<f64>();

    let vec_a = generate_data()?;
    let vec_b = generate_data()?;
    let vec_out = &mut vec![0_f64; VALUES].into_boxed_slice();
    // let vec_out = &mut [0_f64; VALUES]

    let config = KernelConfiguration::new("add", 512, 1024, VALUES as u32, 0);
    runner.allocate_buffer(DeviceBuffer::from_slice(&vec_a)?);
    runner.allocate_buffer(DeviceBuffer::from_slice(&vec_b)?);
    runner.allocate_buffer(DeviceBuffer::from_slice(vec_out)?);
    runner.launch_binary_arithmetics_kernel(config)?;
    runner.collect_result(2, vec_out, VALUES)?;
    let min = vec_out
        .iter()
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .expect("expected min");
    let max = vec_out
        .iter()
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .expect("expected max");
    dbg!(min);
    dbg!(max);
    dbg!(vec_out.iter().filter(|x| x.eq(&min)).count());
    dbg!(vec_out.iter().filter(|x| x.eq(&max)).count());
    Ok(())
}

fn main() -> Result<()> {
    let mut runner = KernelRunner::try_new()?;

    // test_run_1(&mut runner)?;
    // test_run_2(&mut runner)?;
    test_run_3(&mut runner)?;

    Ok(())
}
