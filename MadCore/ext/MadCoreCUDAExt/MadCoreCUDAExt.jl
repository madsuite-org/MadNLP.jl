module MadCoreCUDAExt

import LinearAlgebra
import SparseArrays: nonzeros, nnz
import LinearAlgebra: Symmetric, mul!

import MadCore
import MadCore: NLPModels
import MadCore:
    @kwdef, MadNLPLogger, @debug, @warn, @error,
    AbstractOptions, AbstractLinearSolver, AbstractNLPModel, set_options!,
    SymbolicException, FactorizationException, SolveException, InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia, tril_to_full!,
    LapackOptions, input_type, is_supported, default_options,
    setup_cholesky!, setup_lu!, setup_qr!, setup_evd!, setup_bunchkaufman!,
    factorize_cholesky!, factorize_lu!, factorize_qr!, factorize_evd!, factorize_bunchkaufman!,
    solve_cholesky!, solve_lu!, solve_qr!, solve_evd!, solve_bunchkaufman!

import KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend

using CUDACore
using cuSPARSE, cuBLAS, cuSOLVER


import .cuSOLVER: cusolverStatus_t, CuPtr, cudaDataType, cublasFillMode_t, cusolverDnHandle_t,
    dense_handle, CuSolverParameters, CUSOLVER_EIG_MODE_VECTOR
import .cuBLAS: handle, CUBLAS_DIAG_NON_UNIT,
    CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, CUBLAS_OP_T

include("cuda_sparse.jl")
include("lapackgpu.jl")
include("cusolver.jl")
include("cuda.jl")
include("timer.jl")

MadCore.solver_type(::Type{MadCore.LapackCUDASolver}) = LapackCUDAState

# Default linear solvers of models on NVIDIA GPUs (cuDSS needs CUDSS loaded).
MadCore.default_sparse_solver(::MadCore.AbstractNLPModel{T, <:CuVector{T}}) where {T} = MadCore.CUDSSSolver
MadCore.default_dense_solver(::MadCore.AbstractNLPModel{T, <:CuVector{T}}) where {T} = MadCore.LapackCUDASolver

end
