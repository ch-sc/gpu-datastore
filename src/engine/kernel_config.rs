#[derive(Debug, Clone)]
pub struct KernelConfiguration {
    pub kernel_name: String,
    // work load per thread
    pub batch_size: u32,
    // threads in a work group
    pub work_group_size: u32,
    // total amount of threads
    pub work_items: u32,
    pub shared_memory: u64,
}

impl KernelConfiguration {
    pub fn new(
        kernel_name: &str,
        batch_size: u32,
        work_group_size: u32,
        work_items: u32,
        shared_memory: u64,
    ) -> Self {
        KernelConfiguration {
            kernel_name: kernel_name.to_string(),
            batch_size,
            work_group_size,
            work_items,
            shared_memory,
        }
    }
}
