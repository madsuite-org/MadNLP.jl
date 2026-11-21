module MadCoreGPUExt

import LinearAlgebra
import SparseArrays: SparseMatrixCSC, nonzeros, nnz, getcolptr, rowvals
import LinearAlgebra: Symmetric, mul!, norm, dot

# GPU abstractions
import Adapt: adapt
import GPUArraysCore: AbstractGPUVector, AbstractGPUMatrix, AbstractGPUArray, @allowscalar
import GPUArrays: AbstractGPUSparseMatrixCSC
# Kernels
import KernelAbstractions: @kernel, @index, synchronize, @Const, get_backend, @localmem, @synchronize

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
    get_inf_du, get_inf_compl, get_min_complementarity, get_average_complementarity,
    get_alpha_max, get_alpha_z, SubVector, count_lu_bounds

const AbstractGPUVectorOrSubVector{T,VT<:AbstractGPUVector{T}} = Union{AbstractGPUVector{T}, SubVector{T, VT}}

include("KKT/kernels_dense.jl")
include("KKT/kernels_sparse.jl")
include("KKT/kernels_qn.jl")
include("utils.jl")
include("KKT/gpu_dense.jl")
include("KKT/gpu_sparse.jl")
include("KKT/gpu_qn.jl")
include("IPM/utils.jl")
include("IPM/kernels.jl")
include("scaling.jl")

# Fallback that fires when a GPU array reaches sparse-matrix construction but no
# backend extension is active. The real overrides (CuSparseMatrixCSC /
# ROCSparseMatrixCSC) are strictly more specific and win once their extension
# loads; without them the call would otherwise hit the generic CPU
# `SparseMatrixCSC` constructor and fail with a cryptic MethodError.
function MadCore._get_sparse_csc(dims, colptr::AbstractGPUVector, rowval, nzval)
    error(
        "MadCore: cannot build a GPU sparse KKT system because the GPU backend " *
            "extension is not loaded. Load the GPU package (CUDA, AMDGPU, oneAPI or Metal) before " *
        "solving a model with GPU arrays.",
    )
end

end # module
