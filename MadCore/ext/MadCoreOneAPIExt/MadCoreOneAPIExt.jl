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

end
