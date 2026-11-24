module MadCoreOneAPIExt

import LinearAlgebra
import SparseArrays: SparseMatrixCSC
import LinearAlgebra: Symmetric

import MadCore

using oneAPI
using oneAPI.oneMKL, oneAPI.Support

include("oneapi_sparse.jl")
include("onemkl.jl")
include("oneapi.jl")

MadCore.solver_type(::Type{MadCore.LapackOneMKLSolver}) = LapackOneMKLState

# Default linear solvers of models on Intel GPUs (SDSSolver needs SparseDirectSolver loaded).
MadCore.default_sparse_solver(::MadCore.AbstractNLPModel{T, <:oneVector{T}}) where {T} = MadCore.SDSSolver
MadCore.default_dense_solver(::MadCore.AbstractNLPModel{T, <:oneVector{T}}) where {T} = MadCore.LapackOneMKLSolver

end
