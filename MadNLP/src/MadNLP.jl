module MadNLP

import MadCore, MadNLPCore
import NLPModels
import HSL, LDLFactorizations, MUMPS_seq_jll, OpenBLAS32_jll, SuiteSparse

# MadNLP bundles MadNLPCore with the CPU linear solvers. For backward compatibility,
# the public names of MadCore and
# every name of MadNLPCore also resolve as `MadNLP.name`.
for name in MadCore.public_names()
    @eval import MadCore: $name
end
for name in MadCore.defined_names(MadNLPCore)
    name === :MadCore && continue
    @eval import MadNLPCore: $name
end

export MadNLPSolver, MadNLPOptions, LDLSolver, LapackCPUSolver, MumpsSolver, MadNLPExecutionStats, madnlp, solve!, madsuite, SchurComplementKKTSystem
Base.USE_GPL_LIBS && export UmfpackSolver, CHOLMODSolver
export Ma27Solver, Ma57Solver, Ma77Solver, Ma86Solver, Ma97Solver, PardisoMKLSolver, PanuaPardisoSolver, KrylovIterator

# MUMPS is the default sparse linear solver of MadNLP. MadCore and MadNLPCore have none.
MadCore.default_sparse_solver(::NLPModels.AbstractNLPModel) = MumpsSolver


using PrecompileTools: @setup_workload, @compile_workload
include("precompile.jl")

global Optimizer

end # end module
