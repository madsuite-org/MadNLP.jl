# ordering
@enum ORDERING begin
    DEFAULT_ORDERING = 0
    METIS_ORDERING = 1
    AMD_ORDERING = 2
    USER_ORDERING = 3
    SYMAMD_ORDERING = 4
    COLAMD_ORDERING = 5
end

# Local function to move data on the device.
gpu_transfer!(x::AbstractArray, y::AbstractArray) = copyto!(x, y)

# GPU sparse CSC to dense matrix transfer (uses _csc_to_dense_kernel! from kernels_sparse.jl).
# Works for any GPU sparse CSC type with .colPtr, .rowVal, .nzVal fields.
# Called from per-backend gpu_transfer! stubs that dispatch on concrete sparse CSC types.
function _gpu_sparse_csc_to_dense!(y::AbstractGPUMatrix{T}, x) where {T}
    n = size(y, 2)
    fill!(y, zero(T))
    backend = get_backend(y)
    _csc_to_dense_kernel!(backend)(y, x.colPtr, x.rowVal, x.nzVal, ndrange = n)
    return
end

