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
import CNLPLib, MadNLPAppBase

include("capi.jl")
include("ampl.jl")
include("warmup.jl")

function __init__()
    TRACE.on = -1     # an image keeps the build's setting
    # the BLAS as this library starts (MUMPS's LP64 OpenBLAS included), recorded before a
    # model library's runtime in the same process can forward libblastrampoline elsewhere
    append!(empty!(BLAS), CNLPLib.blas_state())
    return nothing
end

end # module
