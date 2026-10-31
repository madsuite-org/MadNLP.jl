module MadCore

import Printf: @sprintf
import LinearAlgebra: BLAS, LAPACK, Adjoint, Symmetric, Diagonal, mul!, ldiv!, rdiv!, lmul!, rmul!, norm, dot, diagind, normInf, transpose!, issuccess
import LinearAlgebra: BlasReal, bunchkaufman, cholesky, qr, lu, bunchkaufman!, cholesky!, axpy!, LowerTriangular
import LinearAlgebra.BLAS: libblastrampoline, BlasInt, @blasfunc
import SparseArrays: SparseArrays, AbstractSparseMatrix, SparseMatrixCSC, sparse, getcolptr, rowvals, nnz, nonzeros
import Base: string, show, print, size, getindex, copyto!, @kwdef
import NLPModels
import NLPModels: finalize, AbstractNLPModel, obj, grad!, cons!, jac_coord!, hess_coord!, hess_structure!, jac_structure!, hess_dense!, jac_dense!, NLPModelMeta, get_nvar, get_ncon, get_minimize, get_x0, get_y0, get_nnzj, get_nnzh, get_lvar, get_uvar, get_lcon, get_ucon
import SolverCore: getStatus, AbstractOptimizationSolver, AbstractExecutionStats, solve!

export LapackCPUSolver, MumpsSolver, LDLSolver, UmfpackSolver, CHOLMODSolver, DummySolver
export Ma27Solver, Ma57Solver, Ma77Solver, Ma86Solver, Ma97Solver, PardisoMKLSolver, PanuaPardisoSolver
export KrylovIterator
export MadNLPExecutionStats, SchurComplementKKTSystem, solve!, madsuite

"""
    AbstractMadNLPSolver{T}

Supertype of the interior-point solvers built on MadCore. The helpers in MadCore
(statistics, logging, KKT assembly, step computations) access the solver through
its fields, so a subtype is expected to carry the same fields as `MadNLPSolver`.
"""
abstract type AbstractMadNLPSolver{T} end

"""
    madsuite(::Val{name}, args...; kwargs...)

Entry point shared by the solvers of the MadSuite. Each solver adds a method for
its own `Val` tag, e.g. `madsuite(Val(:madnlp), nlp)`.
"""
function madsuite end

include("enums.jl")
include("utils.jl")
include("matrixtools.jl")
include(joinpath("Callbacks", "nlpmodels.jl"))
include(joinpath("Callbacks", "wrappers.jl"))
include("quasi_newton.jl")
include(joinpath("KKT", "KKTsystem.jl"))
include(joinpath("LinearSolvers", "linearsolvers.jl"))
include(joinpath("IPM", "types.jl"))
include(joinpath("IPM", "options.jl"))
include(joinpath("IPM", "utils.jl"))
include(joinpath("IPM", "api.jl"))
include(joinpath("IPM", "kernels.jl"))
include(joinpath("IPM", "callbacks.jl"))
include(joinpath("IPM", "factorization.jl"))

"""
    MadCore.internal_names(mod::Module = MadCore)

Return the names defined in `mod`, excluding compiler-generated names. The packages
built on MadCore import these names to use MadCore's internals without qualification:

```julia
for name in MadCore.internal_names()
    @eval import MadCore: \$name
end
```
"""
function internal_names(mod::Module = MadCore)
    return filter(names(mod; all = true, imported = true)) do name
        s = string(name)
        !startswith(s, '#') &&
            name ∉ (:eval, :include, :__init__, nameof(mod)) && isdefined(mod, name)
    end
end

function __init__()
    Base.Experimental.register_error_hint(_missing_extension_hint, MethodError)
end

end # module
