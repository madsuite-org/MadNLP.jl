module MadCoreGPUMetalExt

import LinearAlgebra
import LinearAlgebra: Symmetric, Adjoint, Transpose
import SparseArrays
import SparseArrays: SparseMatrixCSC

import KernelAbstractions: @kernel, @index, @Const
import GPUArrays

import MadCore
import MadCoreGPU

using Metal

include("sparse.jl")
include("metal_sparse.jl")

end
