extern "C" __constant__ int my_constant = 314;

extern "C" __global__ void add(const float* x, const float* y, float* out, int count) {
    // local_id             = threadIdx.x
    // work_group_id        = blockIdx.x
    // work_group_size      = blockDim.x
    // global_size          = blockDim.x * gridDim.x
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] + y[i];
    }
}