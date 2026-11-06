module MadNLPCore

import MadCore
for name in MadCore.public_names()
    @eval import MadCore: $name
end
import Base: @kwdef, copyto!, finalize, string
import LinearAlgebra: axpy!, dot, norm
import MadCore: SubVector, _get_primary_options, _set_aug_diagonal!
import NLPModels
import NLPModels: AbstractNLPModel, get_x0, obj
import Printf: @sprintf
import SolverCore: solve!

export MadNLPSolver, MadNLPOptions, madnlp
# MadNLPCore also exports what MadCore exports.
for name in names(MadCore)
    (name === :MadCore || !Base.isexported(MadCore, name)) && continue
    @eval export $name
end

# Version info
version() = string(pkgversion(@__MODULE__))
introduce() = "\033[34mMad\033[31mN\033[32mL\033[35mP\033[0m version v$(version())"

include(joinpath("IPM", "IPM.jl"))
include("precompile_models.jl")

madsuite(::Val{:madnlp}, args...; kwargs...) = madnlp(args...; kwargs...)

end # module
