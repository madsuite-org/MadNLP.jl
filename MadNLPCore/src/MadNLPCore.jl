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
# The banner reports the MadNLP release when MadNLP is loaded, and MadNLPCore's otherwise.
const MADNLP_PKGID = Base.PkgId(Base.UUID("2621e9c9-9eb4-46b1-8089-e8c72242dfb6"), "MadNLP")
function introduce()
    madnlp = get(Base.loaded_modules, MADNLP_PKGID, nothing)
    name, v = madnlp === nothing ? ("Core", version()) : ("", string(pkgversion(madnlp)))
    return "\033[34mMad\033[31mN\033[32mL\033[35mP\033[0m$name version v$v"
end

include(joinpath("IPM", "IPM.jl"))

madsuite(::Val{:madnlp}, args...; kwargs...) = madnlp(args...; kwargs...)

end # module
