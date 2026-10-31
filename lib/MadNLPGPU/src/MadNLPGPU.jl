module MadNLPGPU

import LinearAlgebra
import SparseArrays: SparseMatrixCSC, nonzeros, nnz
import LinearAlgebra: Symmetric, mul!, norm, dot

# GPU abstractions
import Adapt: adapt
import GPUArraysCore: AbstractGPUVector, AbstractGPUMatrix, AbstractGPUArray, @allowscalar
# Kernels
import KernelAbstractions: @kernel, @index, synchronize, @Const, get_backend

import MadNLP: NLPModels
import MadNLP
import MadNLP:
    @kwdef, MadNLPLogger, @debug, @warn, @error,
    AbstractOptions, AbstractLinearSolver, AbstractNLPModel, set_options!,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia, tril_to_full!,
    LapackOptions, input_type, is_supported, default_options,
    setup_cholesky!, setup_lu!, setup_qr!, setup_evd!, setup_bunchkaufman!,
    factorize_cholesky!, factorize_lu!, factorize_qr!, factorize_evd!, factorize_bunchkaufman!,
    solve_cholesky!, solve_lu!, solve_qr!, solve_evd!, solve_bunchkaufman!
import MadNLP:
    _get_varphi, get_varphi, get_inf_du, get_inf_compl, get_min_complementarity,
    get_varphi_d, get_alpha_max, get_alpha_z, get_obj_val_R, get_theta_R, get_inf_pr_R,
    get_inf_du_R, get_inf_compl_R, get_alpha_max_R, get_alpha_z_R, get_varphi_R, get_F,
    get_varphi_d_R, get_rel_search_norm, populate_RR_nn!, SubVector, count_lu_bounds
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

global LapackCUDASolver::Any = nothing
global CUDSSSolver::Any = nothing
global LapackROCmSolver::Any = nothing
export LapackCUDASolver, CUDSSSolver, LapackROCmSolver

# Fallback that fires when a GPU array reaches sparse-matrix construction but no
# backend extension is active. The real overrides (CuSparseMatrixCSC /
# ROCSparseMatrixCSC) are strictly more specific and win once their extension
# loads; without them the call would otherwise hit the generic CPU
# `SparseMatrixCSC` constructor and fail with a cryptic MethodError. For CUDA the
# extension only activates once CUDSS is loaded too.
function MadNLP._get_sparse_csc(dims, colptr::AbstractGPUVector, rowval, nzval)
    error(
        "MadNLPGPU: cannot build a GPU sparse KKT system because the GPU backend " *
        "extension is not loaded. For CUDA, the extension activates only once both " *
        "the CUDA backend and CUDSS are loaded — add `using CUDSS` (in addition to " *
        "your CUDA package) before solving a model with GPU arrays.",
    )
end

# re-export MadNLP, including deprecated names
for name in names(MadNLP, all=true)
    if Base.isexported(MadNLP, name)
        @eval using MadNLP: $(name)
        @eval export $(name)
    end
end

end # module
