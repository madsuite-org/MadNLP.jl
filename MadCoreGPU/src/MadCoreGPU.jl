module MadCoreGPU

import LinearAlgebra
import SparseArrays: SparseMatrixCSC, nonzeros, nnz
import LinearAlgebra: Symmetric, mul!, norm, dot

# GPU abstractions
import Adapt: adapt
import GPUArraysCore: AbstractGPUVector, AbstractGPUMatrix, AbstractGPUArray, @allowscalar
# Kernels
import KernelAbstractions: @kernel, @index, synchronize, @Const, get_backend

import MadCore: NLPModels
import MadCore
import MadCore:
    @kwdef, MadNLPLogger, @debug, @warn, @error,
    AbstractOptions, AbstractLinearSolver, AbstractNLPModel, set_options!,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia, tril_to_full!,
    LapackOptions, input_type, is_supported, default_options,
    setup_cholesky!, setup_lu!, setup_qr!, setup_evd!, setup_bunchkaufman!,
    factorize_cholesky!, factorize_lu!, factorize_qr!, factorize_evd!, factorize_bunchkaufman!,
    solve_cholesky!, solve_lu!, solve_qr!, solve_evd!, solve_bunchkaufman!
import MadCore:
    _get_varphi, get_varphi, get_inf_du, get_inf_compl, get_min_complementarity,
    get_varphi_d, get_alpha_max, get_alpha_z, get_rel_search_norm, SubVector, count_lu_bounds
# AMD and Metis
import AMD, Metis

const AbstractGPUVectorOrSubVector{T,VT<:AbstractGPUVector{T}} = Union{AbstractGPUVector{T}, SubVector{T, VT}}

include("KKT/kernels_dense.jl")
include("KKT/kernels_sparse.jl")
include("KKT/kernels_qn.jl")
include("utils.jl")
include("timer.jl")
include("KKT/gpu_dense.jl")
include("KKT/gpu_sparse.jl")
include("KKT/gpu_qn.jl")
include("IPM/utils.jl")
include("IPM/kernels.jl")

# Linear solvers implemented in the GPU extensions, selected by marker types.
"""
    CUDSSSolver

Select the sparse direct solver cuDSS on NVIDIA GPUs. Implemented in the extension
loaded by CUDACore, cuBLAS, cuSOLVER, cuSPARSE and CUDSS.
"""
struct CUDSSSolver end
MadCore.backend_package(::Type{CUDSSSolver}) = "CUDA and CUDSS"

"""
    LapackCUDASolver

Select the dense factorizations of cuSOLVER on NVIDIA GPUs. Implemented in the
extension loaded by CUDACore, cuBLAS, cuSOLVER and cuSPARSE.
"""
struct LapackCUDASolver end
MadCore.backend_package(::Type{LapackCUDASolver}) = "CUDA"

"""
    LapackROCmSolver

Select the dense factorizations of rocSOLVER on AMD GPUs. Implemented in the
extension loaded by AMDGPU.
"""
struct LapackROCmSolver end
MadCore.backend_package(::Type{LapackROCmSolver}) = "AMDGPU"

export LapackCUDASolver, CUDSSSolver, LapackROCmSolver

# Fallback that fires when a GPU array reaches sparse-matrix construction but no
# backend extension is active. The real overrides (CuSparseMatrixCSC /
# ROCSparseMatrixCSC) are strictly more specific and win once their extension
# loads; without them the call would otherwise hit the generic CPU
# `SparseMatrixCSC` constructor and fail with a cryptic MethodError.
function MadCore._get_sparse_csc(dims, colptr::AbstractGPUVector, rowval, nzval)
    error(
        "MadCoreGPU: cannot build a GPU sparse KKT system because the GPU backend " *
        "extension is not loaded. Load the GPU package (CUDA or AMDGPU) before " *
        "solving a model with GPU arrays.",
    )
end

end # module
