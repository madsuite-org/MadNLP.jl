"""
    MadNLPApp

MadNLP compiled ahead of time, with two interfaces:

  - `libmadnlp`, a shared library with a C entry point, `madnlp_solve`
    (`include/madnlp.h`): a problem given as sizes, bounds and evaluation callbacks, solved
    with MadNLP. The same function is exported as `cnlp_solve`, so the library is also a
    cnlp solver (`cnlp solve ... --solver madnlp`).
  - `madnlp`, an executable that solves AMPL `.nl` files (`madnlp model.nl -AMPL`).

Both run on the CPU, with MadNLP's default linear solver (MUMPS).
"""
module MadNLPApp

using MadNLP: MadNLP, madnlp
import MadCore, MadNLPCore
import NLPModels
using NLPModels: AbstractNLPModel, NLPModelMeta, Counters, increment!

include("capi.jl")
include("ampl.jl")
include("warmup.jl")

__init__() = (TRACE[] = -1; nothing)

end # module
