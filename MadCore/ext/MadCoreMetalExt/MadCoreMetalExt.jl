module MadCoreMetalExt

import LinearAlgebra
import LinearAlgebra: Symmetric, Adjoint, Transpose
import SparseArrays
import SparseArrays: SparseMatrixCSC

import KernelAbstractions: @kernel, @index, @Const
import GPUArrays

import MadCore

using Metal

include("sparse.jl")
include("metal_sparse.jl")
include("metal_dense.jl")

# Default linear solvers of models on Apple GPUs (SDSSolver needs SparseDirectSolver loaded).
# Metal has no dense factorizations: LapackCPUSolver factorizes the KKT matrix on the host,
# with the LAPACK that libblastrampoline forwards to (Accelerate after `using AppleAccelerate`).
MadCore.default_sparse_solver(::MadCore.AbstractNLPModel{T, <:MtlVector{T}}) where {T} = MadCore.SDSSolver
MadCore.default_dense_solver(::MadCore.AbstractNLPModel{T, <:MtlVector{T}}) where {T} = MadCore.LapackCPUSolver

end
