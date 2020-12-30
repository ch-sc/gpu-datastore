use arrow::error::ArrowError;
use rustacuda::error::CudaError;
use std::ffi::NulError;
use std::result;

pub type Result<T> = result::Result<T, DataStoreError>;

#[derive(Debug)]
pub enum DataStoreError {
    CudaParseError(String),
    CudaExecutionError(String),
    ArrowError(String),
    InternalError(String),
}

impl From<NulError> for DataStoreError {
    fn from(nul_error: NulError) -> Self {
        DataStoreError::CudaExecutionError(format!("{:?}", nul_error))
    }
}

impl From<CudaError> for DataStoreError {
    fn from(cuda_error: CudaError) -> Self {
        DataStoreError::CudaExecutionError(format!("{:?}", cuda_error))
    }
}

impl<T> From<(CudaError, T)> for DataStoreError {
    fn from(t: (CudaError, T)) -> Self {
        DataStoreError::from(t.0)
    }
}

impl From<ArrowError> for DataStoreError {
    fn from(error: ArrowError) -> Self {
        DataStoreError::ArrowError(format!("{:?}", error))
    }
}
