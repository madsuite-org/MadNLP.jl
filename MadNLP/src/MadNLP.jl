module MadNLP

import MadCore, MadNLPCore
import NLPModels
import HSL, LDLFactorizations, MUMPS_seq_jll, OpenBLAS32_jll, SuiteSparse
import Krylov   # GMRES refinement (KrylovIterator), which the default preset uses

# MadNLP bundles MadNLPCore with the CPU linear solvers. For backward compatibility,
# every name of MadCore and MadNLPCore also resolves as `MadNLP.name`.
for name in MadCore.defined_names(MadCore)
    @eval import MadCore: $name
end
for name in MadCore.defined_names(MadNLPCore)
    name === :MadCore && continue
    @eval import MadNLPCore: $name
end

export MadNLPSolver, MadNLPOptions, LDLSolver, LapackCPUSolver, MumpsSolver, MadNLPExecutionStats, madnlp, solve!, madsuite, SchurComplementKKTSystem
export AbstractPreset, ShiftedPreset, AugmentedPreset, CondensedPreset
Base.USE_GPL_LIBS && export UmfpackSolver, CHOLMODSolver
export Ma27Solver, Ma57Solver, Ma77Solver, Ma86Solver, Ma97Solver, PardisoMKLSolver, PanuaPardisoSolver, KrylovIterator
export AbstractScaler, NoScaler, JacobiScaler, MaxNormScaler, RuizScaler
export AbstractOrdering, DefaultOrdering, MetisOrdering, AMDOrdering

# The public names of MadCore and MadNLPCore are public in MadNLP too, so that
# `MadNLP.name` raises no warning in the REPL. The keyword `public` does not parse on Julia 1.10.
@static if VERSION >= v"1.11"
    let pub = union(MadCore.public_names(), MadNLPCore.NLPCORE_PUBLIC_NAMES,
                      filter(n -> Base.isexported(MadCore, n), names(MadCore)),
                      filter(n -> Base.isexported(MadNLPCore, n), names(MadNLPCore)))
        eval(Expr(:public, filter(n -> n ∉ (:MadCore, :MadNLPCore) && !Base.isexported(@__MODULE__, n), pub)...))
    end
end

# MUMPS is the default sparse linear solver of MadNLP. MadCore and MadNLPCore have none.
MadCore.default_sparse_solver(::NLPModels.AbstractNLPModel) = MumpsSolver

# ShiftedPreset factorizes by CHOLMOD's LDLᵀ, with small pivots bounded as cuDSS's are on a GPU,
# and CondensedPreset by CHOLMOD's Cholesky; without Julia's GPL libraries, by MUMPS.
MadNLPCore.shifted_sparse_solver(::NLPModels.AbstractNLPModel) = Base.USE_GPL_LIBS ? MadCore.CHOLMODSolver : MumpsSolver
if Base.USE_GPL_LIBS
    # CHOLMOD under ShiftedPreset on the augmented system: LDLᵀ with small pivots bounded
    function MadCore.default_options(
            ::NLPModels.AbstractNLPModel{T}, ::Type{MadNLPCore.ShiftedPreset}, ::Type{MadCore.SparseKKTSystem},
            ::Type{MadCore.CHOLMODSolver},
        ) where {T}
        opt = MadCore.default_options(MadCore.CHOLMODSolver)
        opt.cholmod_algorithm = MadCore.LDL
        opt.cholmod_dbound = Float64(MadNLPCore.preset_perturbation(T))
        return opt
    end
    # CHOLMOD on the condensed system, under any preset: Cholesky of the positive definite matrix
    function MadCore.default_options(
            ::NLPModels.AbstractNLPModel{T}, ::Type{MadCore.SparseCondensedKKTSystem}, ::Type{MadCore.CHOLMODSolver},
        ) where {T}
        opt = MadCore.default_options(MadCore.CHOLMODSolver)
        opt.cholmod_algorithm = MadCore.CHOLESKY
        return opt
    end
end

# The other LDLᵀ solvers under ShiftedPreset on the augmented system, with small pivots bounded as
# CHOLMOD's and cuDSS's are: a pivot smaller than `preset_perturbation(T)` in magnitude becomes one of
# that magnitude. LDLFactorizations chooses the sign by block, positive in the primal block and
# negative in the dual block (the KKT system declares the blocks), as SparseDirectSolver.jl does: the
# sign of the pivots in the inertia the interior-point method wants.
function MadCore.default_options(
        ::NLPModels.AbstractNLPModel{T}, ::Type{MadNLPCore.ShiftedPreset}, ::Type{MadCore.SparseKKTSystem},
        ::Type{MadCore.LDLSolver},
    ) where {T}
    opt = MadCore.default_options(MadCore.LDLSolver)
    δ = Float64(MadNLPCore.preset_perturbation(T))
    opt.ldl_r1, opt.ldl_r2, opt.ldl_tol = δ, -δ, δ
    return opt
end
# MA86 under ShiftedPreset: static pivoting, no threshold pivoting
function MadCore.default_options(
        ::NLPModels.AbstractNLPModel{T}, ::Type{MadNLPCore.ShiftedPreset}, ::Type{MadCore.SparseKKTSystem},
        ::Type{MadCore.Ma86Solver},
    ) where {T}
    opt = MadCore.default_options(MadCore.Ma86Solver)
    opt.ma86_static = Float64(MadNLPCore.preset_perturbation(T))
    opt.ma86_u = 0.0
    return opt
end
# MA57 under ShiftedPreset: no threshold pivoting, and small pivots bounded by its static pivoting
function MadCore.default_options(
        ::NLPModels.AbstractNLPModel{T}, ::Type{MadNLPCore.ShiftedPreset}, ::Type{MadCore.SparseKKTSystem},
        ::Type{MadCore.Ma57Solver},
    ) where {T}
    opt = MadCore.default_options(MadCore.Ma57Solver)
    opt.ma57_pivot_bound = Float64(MadNLPCore.preset_perturbation(T))
    return opt
end

using PrecompileTools: @setup_workload, @compile_workload
include("precompile.jl")

global Optimizer

end # end module
