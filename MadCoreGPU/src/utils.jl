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

# Workaround for broken norm() on views of GPU arrays.
# See: https://github.com/JuliaGPU/CUDA.jl/issues/2811
#      https://github.com/JuliaGPU/AMDGPU.jl/issues/607
if VERSION > v"1.11"
    _my1norm(x) = mapreduce(abs, +, x)
    function MadCore.get_sd(l::AbstractGPUVector{T}, zl_r, zu_r, s_max) where {T}
        return max(
            s_max,
            (_my1norm(l) + _my1norm(zl_r) + _my1norm(zu_r)) / max(1, (length(l) + length(zl_r) + length(zu_r))),
        ) / s_max
    end
    function MadCore.get_sc(zl_r::SubArray{T, 1, VT}, zu_r, s_max) where {T, VT <: AbstractGPUVector{T}}
        return max(
            s_max,
            (_my1norm(zl_r) + _my1norm(zu_r)) / max(1, length(zl_r) + length(zu_r)),
        ) / s_max
    end
end
