"""
    MadNLPApp

MadNLP on the CPU, compiled ahead of time, with two interfaces:

  - `libmadnlp`, a shared library with a C entry point, `madnlp_solve`
    (`include/madnlp.h`): a problem given as sizes, bounds and evaluation callbacks, solved
    with MadNLP. The same function is exported as `cnlp_solve`, so the library is also a
    cnlp solver (`cnlp solve ... --solver madnlp`).
  - `madnlp`, an executable that solves AMPL `.nl` files (`madnlp model.nl -AMPL`).

It is MadNLPCore with MUMPS as its linear solver; the HSL solvers are not included (their
license does not allow it). The GPU solver is cuMadNLPApp.
"""
module MadNLPApp

import MadCore, MadNLPCore
import MUMPS_seq_jll, OpenBLAS32_jll   # MadCore's MUMPS extension: the default linear solver
import Krylov                          # GMRES refinement (KrylovIterator) of the default preset
import NLPModels
using NLPModels: AbstractNLPModel
import CNLPLib

# MUMPS is the sparse linear solver, under every preset (as MadNLP sets it, without HSL)
MadCore.default_sparse_solver(::AbstractNLPModel) = MadCore.MumpsSolver
MadNLPCore.shifted_sparse_solver(::AbstractNLPModel) = MadCore.MumpsSolver

include("capi.jl")
include("ampl.jl")
include("warmup.jl")

function __init__()
    TRACE.on = -1     # an image keeps the build's setting
    empty!(BLAS)      # recorded at the first call (an image keeps the build's)
    return nothing
end

end # module
