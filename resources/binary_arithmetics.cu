// compile command:
// https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
// nvcc binary_arithmetics.cu --ptx -o binary_arithmetics.ptx --gpu-architecture=compute_70 --gpu-code=sm_70,compute_70

#define ADD +
#define SUB -
#define MUL *
#define DIV /
#define MOD %

#define BINARY_EXPRESSION(_operation)                                           \
    uint local_thread_id = threadIdx.x;                                         \
    uint work_group_id = blockIdx.x;                                            \
    uint work_group_size = blockDim.x;                                          \
    uint global_thread_id = work_group_size * work_group_id + local_thread_id;  \
    uint end = BATCH * STRIDE;                                                  \
    for (uint i = 0; i < end; i += STRIDE) {                                    \
        uint idx = global_thread_id + i;                                        \
        out[idx] = vec_a[idx] _operation vec_b[idx];                            \
        i += STRIDE;                                                            \
    }


extern "C" __global__ void add(double* vec_a,
                               double* vec_b,
                               double* out,
                               const uint BATCH,
                               const uint STRIDE) {
    BINARY_EXPRESSION(ADD)
}

extern "C" __global__ void sub(double* vec_a,
                               double* vec_b,
                               double* out,
                               const uint BATCH,
                               const uint STRIDE) {
    BINARY_EXPRESSION(SUB)
}

extern "C" __global__ void mul(double* vec_a,
                               double* vec_b,
                               double* out,
                               const uint BATCH,
                               const uint STRIDE) {
    BINARY_EXPRESSION(MUL)
}

// 'div' is already reserved
extern "C" __global__ void division(double* vec_a,
                                    double* vec_b,
                                    double* out,
                                    const uint BATCH,
                                    const uint STRIDE) {
    BINARY_EXPRESSION(DIV)
}
