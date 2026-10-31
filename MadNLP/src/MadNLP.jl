module MadNLP

import MadCore, MadNLPCore
import NLPModels
import HSL, LDLFactorizations, MUMPS_seq_jll, OpenBLAS32_jll, SuiteSparse

# MadNLP bundles MadNLPCore with the CPU linear solvers. For backward compatibility,
# every name of MadCore and MadNLPCore also resolves as `MadNLP.name`.
for name in MadCore.internal_names()
    @eval import MadCore: $name
end
for name in MadCore.internal_names(MadNLPCore)
    name in (:MadCore, :version) && continue
    @eval import MadNLPCore: $name
end

export MadNLPSolver, MadNLPOptions, LDLSolver, LapackCPUSolver, MumpsSolver, MadNLPExecutionStats, madnlp, solve!, madsuite, SchurComplementKKTSystem
Base.USE_GPL_LIBS && export UmfpackSolver, CHOLMODSolver
export Ma27Solver, Ma57Solver, Ma77Solver, Ma86Solver, Ma97Solver, PardisoMKLSolver, PanuaPardisoSolver, KrylovIterator

version() = string(pkgversion(@__MODULE__))

# MUMPS is the default sparse linear solver of MadNLP. MadCore and MadNLPCore have none.
MadCore.default_sparse_solver(::NLPModels.AbstractNLPModel) = MumpsSolver


using PrecompileTools: @setup_workload, @compile_workload
include("precompile.jl")

global Optimizer

end # end module
