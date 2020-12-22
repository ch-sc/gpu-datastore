use rustacuda::error::CudaError;
use std::ffi::NulError;
use std::result;

pub type Result<T> = result::Result<T, DataStoreErrors>;

#[derive(Debug)]
pub enum DataStoreErrors {
    CudaParseError(String),
    CudaExecutionError(String),
    DataStoreError(String),
}

impl From<NulError> for DataStoreErrors {
    fn from(nul_error: NulError) -> Self {
        DataStoreErrors::CudaExecutionError(format!("{:?}", nul_error))
    }
}

impl From<CudaError> for DataStoreErrors {
    fn from(cuda_error: CudaError) -> Self {
        DataStoreErrors::CudaExecutionError(format!("{:?}", cuda_error))
    }
}
