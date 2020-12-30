#[cfg(target_os = "cuda")]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn sum(
    vec_a: *const f64,
    vec_b: *const f64,
    out: *mut f64,
    BATCH: *const u32,
    STRIDE: *const u32,
) {
    use nvptx_builtins::*;
    let local_thread_id = thread_idx_x();
    let work_group_id = block_idx_x();
    let work_group_size = block_dim_x();
    let global_thread_id = work_group_size * work_group_id + local_thread_id;
    // global_size          = grid_dim_x() * work_group_size;

    let end = BATCH * STRIDE;
    let mut i = 0_u32;
    while i < end {
        let idx = global_thread_id + i;
        out[idx] = vec_a[idx] + vec_b[idx];
        i += STRIDE;
    }
}
