# HSL, brought by the user. The library carries HSL.jl's wrappers and MadCore's HSL solvers
# but no HSL: its license does not allow passing it on. A solve that asks for an HSL solver
# (`linear_solver=Ma57Solver`, ...) uses the user's own `libhsl`, found, in order:
#
#   1. the option `hsl_library=DIR`;
#   2. `JULIA_HSL_LIBRARY_PATH` (as HSL.jl reads it);
#   3. `artifacts.hsl` in $CNLPLIB_HOME/config.toml (`cnlp config set artifacts.hsl DIR`);
#   4. $CNLPLIB_HOME/solvers/madnlp/lib/hsl, where `cnlp hsl add` puts a user's HSL_jll.
#
# DIR holds libhsl.so (and libhsl_subset.so, libhsl_subset_64.so).

const HSL_SOLVERS = (MadCore.Ma27Solver, MadCore.Ma57Solver, MadCore.Ma77Solver, MadCore.Ma86Solver, MadCore.Ma97Solver)
const HSL_IN_USE = Ref("")   # the libhsl the wrappers call

_cnlplib_home() = get(ENV, "CNLPLIB_HOME", joinpath(homedir(), ".cnlplib"))
function _configured_hsl()
    file = joinpath(_cnlplib_home(), "config.toml")
    isfile(file) || return nothing
    cfg = TOML.tryparsefile(file)
    cfg isa AbstractDict || return nothing
    arts = get(cfg, "artifacts", nothing)
    dir = arts isa AbstractDict ? get(arts, "hsl", nothing) : nothing
    return dir isa String && !isempty(dir) ? dir : nothing
end

"The directory of the user's HSL, or `nothing` (see the top of src/hsl.jl)."
function hsl_dir(option::Union{Nothing, AbstractString} = nothing)
    option === nothing || return String(option)
    d = get(ENV, "JULIA_HSL_LIBRARY_PATH", ""); isempty(d) || return d
    d = _configured_hsl(); d === nothing || return d
    d = joinpath(_cnlplib_home(), "solvers", "madnlp", "lib", "hsl")
    return isfile(joinpath(d, "libhsl.$(Libdl.dlext)")) ? d : nothing
end

"""
    use_hsl!(dir)

Point HSL.jl's wrappers at the libhsl in `dir`, and check it is a working HSL (HSL_jll's
placeholder is not).
"""
function use_hsl!(dir::AbstractString)
    lib = joinpath(dir, "libhsl.$(Libdl.dlext)")
    lib == HSL_IN_USE[] && return nothing
    isfile(lib) || throw(ArgumentError("no libhsl.$(Libdl.dlext) in $dir"))
    HSL.libhsl = lib
    HSL.libhsl_subset = joinpath(dir, "libhsl_subset.$(Libdl.dlext)")
    HSL.libhsl_subset_64 = joinpath(dir, "libhsl_subset_64.$(Libdl.dlext)")
    HSL.LIBHSL_isfunctional() || throw(ArgumentError(
        "$lib is not a working HSL (HSL_jll's placeholder?): use the HSL_jll you obtained from STFC"))
    HSL_IN_USE[] = lib
    return nothing
end

# Options: `hsl_library` is ours, not MadNLP's; an HSL solver needs the user's HSL in place.
function hsl_options!(kw::Vector{Pair{Symbol, Any}})
    k = findfirst(p -> first(p) === :hsl_library, kw)
    option = k === nothing ? nothing : string(last(kw[k]))
    k === nothing || deleteat!(kw, k)
    ls = (i = findfirst(p -> first(p) === :linear_solver, kw)) === nothing ? nothing : last(kw[i])
    if ls in HSL_SOLVERS
        dir = hsl_dir(option)
        # an HSL already in use (an earlier hsl_library=) serves the solves that name none
        dir === nothing && !isempty(HSL_IN_USE[]) && return kw
        dir === nothing && throw(ArgumentError(
            "$(nameof(ls)) needs HSL, which this library does not include (its license does not allow " *
            "it): add your own with `cnlp hsl add <your HSL_jll directory or tarball>`, or set " *
            "JULIA_HSL_LIBRARY_PATH, or pass hsl_library=DIR"))
        use_hsl!(dir)
    end
    return kw
end
