module MadNLP

import MadCore, MadNLPCore
import NLPModels
import HSL, LDLFactorizations, MUMPS_seq_jll, OpenBLAS32_jll, SuiteSparse
import Krylov   # GMRES refinement (KrylovIterator), which the default preset uses

# MadNLP bundles MadNLPCore with the CPU linear solvers. For backward compatibility,
# every name of MadCore and MadNLPCore also resolves as `MadNLP.name`.
for name in MadCore.internal_names()
    @eval import MadCore: $name
end
for name in MadCore.internal_names(MadNLPCore)
    name === :MadCore && continue
    @eval import MadNLPCore: $name
end

export MadNLPSolver, MadNLPOptions, LDLSolver, LapackCPUSolver, MumpsSolver, MadNLPExecutionStats, madnlp, solve!, madsuite, SchurComplementKKTSystem
export AbstractPreset, ShiftedPreset, AugmentedPreset, CondensedPreset
Base.USE_GPL_LIBS && export UmfpackSolver, CHOLMODSolver
export Ma27Solver, Ma57Solver, Ma77Solver, Ma86Solver, Ma97Solver, PardisoMKLSolver, PanuaPardisoSolver, KrylovIterator
export AbstractScaler, NoScaler, JacobiScaler, MaxNormScaler, RuizScaler

# MUMPS is the default sparse linear solver of MadNLP. MadCore and MadNLPCore have none.
MadCore.default_sparse_solver(::NLPModels.AbstractNLPModel) = MumpsSolver

# ShiftedPreset factorizes by CHOLMOD's LDLᵀ, with small pivots bounded as cuDSS's are on a GPU,
# and CondensedPreset by CHOLMOD's Cholesky; without Julia's GPL libraries, by MUMPS.
MadNLPCore.shifted_sparse_solver(::NLPModels.AbstractNLPModel) = Base.USE_GPL_LIBS ? MadCore.CHOLMODSolver : MumpsSolver
if Base.USE_GPL_LIBS
    # CHOLMOD under ShiftedPreset: LDLᵀ with small pivots bounded
    function MadCore.default_options(
            ::NLPModels.AbstractNLPModel{T}, ::Type{<:MadNLPCore.PivotBoundedPreset}, ::Type{<:MadNLPCore.SparsePresetKKTSystem},
            ::Type{MadCore.CHOLMODSolver},
        ) where {T}
        opt = MadCore.default_options(MadCore.CHOLMODSolver)
        opt.cholmod_algorithm = MadCore.LDL
        opt.cholmod_dbound = Float64(MadNLPCore.preset_perturbation(T))
        return opt
    end
    # CHOLMOD under CondensedPreset: Cholesky of the positive definite condensed matrix
    function MadCore.default_options(
            ::NLPModels.AbstractNLPModel{T}, ::Type{MadNLPCore.CondensedPreset}, ::Type{MadCore.SparseCondensedKKTSystem},
            ::Type{MadCore.CHOLMODSolver},
        ) where {T}
        opt = MadCore.default_options(MadCore.CHOLMODSolver)
        opt.cholmod_algorithm = MadCore.CHOLESKY
        return opt
    end
end


using PrecompileTools: @setup_workload, @compile_workload
include("precompile.jl")

global Optimizer

end # end module
