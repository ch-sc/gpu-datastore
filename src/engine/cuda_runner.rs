use rustacuda::prelude::*;

use crate::common::errors::*;
use rustacuda::memory::DeviceBox;

use crate::engine::kernel_config::KernelConfiguration;
use std::ffi::CString;

pub struct KernelRunner {
    buffers: Vec<DeviceBuffer<f64>>,
    module: Option<Module>,
    _context: Context,
    _device: Device,
}

impl KernelRunner {
    pub fn try_new() -> Result<Self> {
        // Initialize the CUDA API
        rustacuda::init(CudaFlags::empty())?;
        // Get the first device
        let device = Device::get_device(0)?;
        // Create a context associated to this device
        let context =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

        Ok(Self {
            _device: device,
            _context: context,
            module: None,
            buffers: Vec::new(),
        })
    }

    pub fn load_file(&mut self, module_data: &CString) -> Result<()> {
        // let module_data = CString::new(include_str!(file_path))?;
        let module = Module::load_from_string(module_data)?;
        self.module = Some(module);
        Ok(())
    }

    /// Create buffers for input and output data
    pub fn allocate_buffer(&mut self, dev_buffer: DeviceBuffer<f64>) {
        self.buffers.push(dev_buffer);
    }

    pub fn launch_expr_binary_kernel(&mut self, config: KernelConfiguration) -> Result<()> {
        let kernel_fun_name = config.kernel_name.as_str();
        let total_work_items = config.work_items;
        let work_group_size = config.work_group_size;
        let grid_size = total_work_items / work_group_size;

        let batch = config.batch_size;
        let stride = total_work_items / batch;

        let module: &Module = self.module.as_ref().unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let function_name = CString::new(kernel_fun_name)?;
        let fun = module.get_function(&function_name)?;

        let buffers = &mut self.buffers;
        unsafe {
            let result = launch!(fun<<<grid_size, work_group_size, 0, stream>>>(
                buffers[0].as_device_ptr(),
                buffers[1].as_device_ptr(),
                buffers[2].as_device_ptr(),
                config.batch_size,
                stride
            ));
            result?;
        }

        Ok(())
    }

    pub fn launch_test_kernel_2(&mut self) -> Result<()> {
        let buffers = &mut self.buffers;
        let module: &Module = self.module.as_ref().unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        // This kernel adds each element in `in_x` and `in_y` and writes the result into `out`.
        unsafe {
            // Launch the kernel with one block of one thread, no dynamic shared memory on `stream`.
            let result = launch!(module.sum<<<1, 1, 0, stream>>>(
                buffers[0].as_device_ptr(),
                buffers[1].as_device_ptr(),
                buffers[2].as_device_ptr(),
                buffers[2].len()
            ));
            result?;

            // Launch the kernel again using the `function` form:
            let function_name = CString::new("sum")?;
            let sum = module.get_function(&function_name)?;
            // Launch with 1x1x1 (1) blocks of 10x1x1 (10) threads, to show that you can use tuples to
            // configure grid and block size.
            let result = launch!(sum<<<(1, 1, 1), (10, 1, 1), 0, stream>>>(
                buffers[0].as_device_ptr(),
                buffers[1].as_device_ptr(),
                buffers[3].as_device_ptr(),
                buffers[3].len()
            ));
            result?;
        }
        // Kernel launches are asynchronous, so we wait for the kernels to finish executing.
        stream.synchronize()?;
        Ok(())
    }

    pub fn collect_result(
        &mut self,
        buffer_index: usize,
        result_buffer: &mut [f64],
        length: usize,
    ) -> Result<()> {
        let buffers = &mut self.buffers;
        let out_buffer = &mut buffers[buffer_index];
        out_buffer.copy_to(&mut result_buffer[0..length])?;
        Ok(())
    }

    pub fn launch_test_kernel(&self) -> Result<f64> {
        let module_data = CString::new(include_str!("../../resources/add.ptx"))?;
        let module = Module::load_from_string(&module_data)?;

        // Create a stream to submit work to
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        // Allocate space on the device and copy numbers to it.
        let mut x = DeviceBox::new(&10.0f64)?;
        let mut y = DeviceBox::new(&20.0f64)?;
        let mut result = DeviceBox::new(&0.0f64)?;

        // Launching kernels is unsafe since Rust can't enforce safety - think of kernel launches
        // as a foreign-function call. In this case, it is - this kernel is written in CUDA C.
        unsafe {
            // Launch the `sum` function with one block containing one thread on the given stream.
            launch!(module.sum<<<1, 1, 0, stream>>>(
                x.as_device_ptr(),
                y.as_device_ptr(),
                result.as_device_ptr(),
                1 // Length
            ))?;
        }

        // The kernel launch is asynchronous, so we wait for the kernel to finish executing
        stream.synchronize()?;

        // Copy the result back to the host
        let mut result_host = 0.0f64;
        result.copy_to(&mut result_host)?;

        println!("Sum is {}", result_host);

        Ok(result_host)
    }
}
