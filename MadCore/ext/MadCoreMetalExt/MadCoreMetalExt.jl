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

end
