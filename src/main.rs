use gpu_datastore::engine::prelude::KernelLauncher;

use gpu_datastore::common::prelude::*;

fn main() -> Result<()> {
    let launcher = KernelLauncher::try_new()?;
    let result = launcher.launch_add_kernel()?;

    println!("result: {:?}", result);
    Ok(())
}
