module MadNLPCore

import MadCore
for name in MadCore.internal_names()
    @eval import MadCore: $name
end

export MadNLPSolver, MadNLPOptions, madnlp
# MadNLPCore also exports what MadCore exports.
for name in names(MadCore)
    name === :MadCore && continue
    @eval export $name
end

# Version info
version() = string(pkgversion(@__MODULE__))
introduce() = "\033[34mMad\033[31mN\033[32mL\033[35mP\033[0m version v$(version())"

include(joinpath("IPM", "IPM.jl"))
include("precompile_models.jl")

madsuite(::Val{:madnlp}, args...; kwargs...) = madnlp(args...; kwargs...)

end # module
