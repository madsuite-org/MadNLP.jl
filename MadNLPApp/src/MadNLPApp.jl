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
import LinearAlgebra
using NLPModels: AbstractNLPModel

# The BLAS this library forwards libblastrampoline to. Another Julia runtime in the same
# process (a cnlp model library) may open the same libblastrampoline by name when it starts
# and forward it elsewhere, or nowhere; every solve puts this library's back first. Recorded
# at the first call into the library, when all of it (MadNLP's MUMPS extension forwards the
# LP64 OpenBLAS) has started, and before a model library loaded after it can have run.
const BLAS_LIBS = String[]
function blas_libs()
    isempty(BLAS_LIBS) && append!(BLAS_LIBS, [l.libname for l in LinearAlgebra.BLAS.lbt_get_config().loaded_libs])
    return BLAS_LIBS
end
function restore_blas!()
    libs = blas_libs()
    cur = [l.libname for l in LinearAlgebra.BLAS.lbt_get_config().loaded_libs]
    cur == libs && return nothing
    for (k, l) in enumerate(libs)
        LinearAlgebra.BLAS.lbt_forward(l; clear = k == 1)
    end
    return nothing
end

include("capi.jl")
include("ampl.jl")
include("warmup.jl")

function __init__()
    TRACE[] = -1
    empty!(BLAS_LIBS)   # recorded at the first call (an image keeps the build's)
    return nothing
end

end # module
