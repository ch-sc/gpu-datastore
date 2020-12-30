// compile command:
// nvcc binary_arithmetics.ptx --gpu-architecture=compute_32


extern "C" __global__ void add(const float* vec_a,
                               const float* vec_b,
                               float* out,
                               const uint BATCH,
                               const uint STRIDE) {
    uint local_thread_id = threadIdx.x;
    uint work_group_id = blockIdx.x;
    uint work_group_size = blockDim.x;
    uint global_thread_id = work_group_size * work_group_id + local_thread_id;
    // uint global_size = gridDim.x * work_group_size;

    uint end = BATCH * STRIDE;
    for (uint i = 0; i < end; i += STRIDE) {
        uint idx = global_thread_id + i;
        out[idx] = vec_a[idx] + vec_b[idx];
        i += STRIDE;
    }
}

extern "C" __global__ void sub(const float* x, const float* y, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] - y[i];
    }
}

extern "C" __global__ void mul(const float* x, const float* y, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] * y[i];
    }
}

