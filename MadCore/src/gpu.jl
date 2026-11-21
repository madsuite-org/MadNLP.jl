# Definitions shared by the GPU extensions. They need no GPU package, so that each
# backend extension depends only on MadCore and its own trigger packages:
#
# * MadCoreGPUExt (KernelAbstractions, GPUArraysCore, Adapt): the backend-agnostic
#   kernels and KKT systems on GPU arrays;
# * MadCoreCUDAExt, MadCoreCUDSSExt, MadCoreAMDGPUExt, MadCoreOneAPIExt,
#   MadCoreMetalExt: the vendor libraries;
# * MadCoreSDSExt: SparseDirectSolver.jl;
# * MadCoreAMDExt, MadCoreMetisExt: the fill-reducing orderings of cuDSS.

"""
    CUDSSSolver

Select the sparse direct solver cuDSS on NVIDIA GPUs. Implemented in the extension
loaded by CUDACore, cuBLAS, cuSOLVER, cuSPARSE and CUDSS.
"""
struct CUDSSSolver end
backend_package(::Type{CUDSSSolver}) = "CUDA and CUDSS"

"""
    LapackCUDASolver

Select the dense factorizations of cuSOLVER on NVIDIA GPUs. Implemented in the
extension loaded by CUDACore, cuBLAS, cuSOLVER and cuSPARSE.
"""
struct LapackCUDASolver end
backend_package(::Type{LapackCUDASolver}) = "CUDA"

"""
    LapackROCmSolver

Select the dense factorizations of rocSOLVER on AMD GPUs. Implemented in the
extension loaded by AMDGPU.
"""
struct LapackROCmSolver end
backend_package(::Type{LapackROCmSolver}) = "AMDGPU"

"""
    LapackOneMKLSolver

Select the dense factorizations of oneMKL on Intel GPUs. Implemented in the
extension loaded by oneAPI.
"""
struct LapackOneMKLSolver end
backend_package(::Type{LapackOneMKLSolver}) = "oneAPI"

"""
    SDSSolver

Select the sparse direct solver SparseDirectSolver.jl on any GPU, or on the CPU for a host
KKT matrix (`SparseMatrixCSC`). Implemented in the extension loaded by SparseDirectSolver.
"""
struct SDSSolver end
backend_package(::Type{SDSSolver}) = "SparseDirectSolver"

"""
    ORDERING

Fill-reducing ordering passed to cuDSS (`cudss_ordering`). `DEFAULT_ORDERING` keeps the
ordering of cuDSS; `METIS_ORDERING` needs Metis loaded, `AMD_ORDERING`,
`SYMAMD_ORDERING` and `COLAMD_ORDERING` need AMD loaded; `USER_ORDERING` takes the
permutation `cudss_perm`.
"""
@enum ORDERING begin
    DEFAULT_ORDERING = 0
    METIS_ORDERING = 1
    AMD_ORDERING = 2
    USER_ORDERING = 3
    SYMAMD_ORDERING = 4
    COLAMD_ORDERING = 5
end

"""
    ordering_permutation(::Val{ordering}, A::SparseMatrixCSC)

Return the fill-reducing permutation `ordering` of the lower triangle `A` of a symmetric
matrix. The methods are in the extensions loaded by Metis (`METIS_ORDERING`) and AMD
(`AMD_ORDERING`, `SYMAMD_ORDERING`, `COLAMD_ORDERING`).
"""
function ordering_permutation(::Val{ordering}, A) where {ordering}
    pkg = ordering == METIS_ORDERING ? "Metis" :
        ordering in (AMD_ORDERING, SYMAMD_ORDERING, COLAMD_ORDERING) ? "AMD" : nothing
    pkg === nothing && throw(ArgumentError("The ordering $ordering has no permutation to compute."))
    return error("Load $pkg to use the ordering $ordering.")
end

# Move data between host and device. The GPU extensions add methods for the
# device array types.
gpu_transfer!(x::AbstractArray, y::AbstractArray) = copyto!(x, y)

# Implemented in MadCoreGPUExt for GPU arrays.
function _gpu_sparse_csc_to_dense! end
function _transfer_coo_to_csc! end

export CUDSSSolver, LapackCUDASolver, LapackROCmSolver, LapackOneMKLSolver, SDSSolver
