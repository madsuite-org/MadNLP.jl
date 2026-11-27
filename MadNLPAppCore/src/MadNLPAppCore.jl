"""
    MadNLPAppCore

What MadNLP's two cnlp solver libraries share, MadNLPApp (CPU) and cuMadNLPApp (GPU): the
solver setup ([`App`](@ref): options from C strings, defaults, tracing, the BLAS it restores),
the solve, MadNLP's status in cnlp's terms, and the preparation of a library image
([`precompile_image`](@ref)). Their builds share `build/build_library.jl`.
"""
module MadNLPAppCore

import CNLPLib, MadCore, MadNLPCore
using NLPModels: AbstractNLPModel

"""
    parse_value(key::Symbol, v::String, modules) -> value

An option value as a user wrote it, as MadNLP takes it. `print_level` takes MadNLP's level
names (INFO, ERROR, ...) or 1-6, and 0 for quiet; a value naming a type of one of `modules`
(`linear_solver=LapackCPUSolver`) is that type; otherwise integers, numbers and booleans are
read as such.
"""
function parse_value(key::Symbol, v::String, modules)
    if key === :print_level || key === :file_print_level
        n = tryparse(Int, v)
        n === nothing && return getfield(MadCore, Symbol(uppercase(v)))::MadCore.LogLevels
        return n == 0 ? MadCore.ERROR : MadCore.LogLevels(n)
    end
    for mod in modules
        isdefined(mod, Symbol(v)) && (t = getfield(mod, Symbol(v)); t isa Type) && return t
    end
    v == "true" && return true
    v == "false" && return false
    n = tryparse(Int, v); n === nothing || return n
    x = tryparse(Float64, v); x === nothing || return x
    return v
end

"""
    options(opts, modules) -> Vector{Pair{Symbol, Any}}

`key => value` string pairs (`CNLPLib.solver_options`) as MadNLP options. `sb` (Ipopt's
banner switch) is accepted and ignored.
"""
options(opts, modules) =
    Pair{Symbol, Any}[Symbol(k) => parse_value(Symbol(k), String(v), modules) for (k, v) in opts if k != "sb"]

"MadNLP's termination status as a cnlp status (`CNLPLib.CNLP_*`)."
function status_of(s)
    s == MadCore.SOLVE_SUCCEEDED && return CNLPLib.CNLP_SOLVED
    s == MadCore.SOLVED_TO_ACCEPTABLE_LEVEL && return CNLPLib.CNLP_ACCEPTABLE
    s == MadCore.INFEASIBLE_PROBLEM_DETECTED && return CNLPLib.CNLP_INFEASIBLE
    s == MadCore.DIVERGING_ITERATES && return CNLPLib.CNLP_UNBOUNDED
    s == MadCore.MAXIMUM_ITERATIONS_EXCEEDED && return CNLPLib.CNLP_ITERATION_LIMIT
    s == MadCore.MAXIMUM_WALLTIME_EXCEEDED && return CNLPLib.CNLP_TIME_LIMIT
    return CNLPLib.CNLP_ERROR
end

"""
    run_madnlp(model, options::Vector{Pair{Symbol, Any}})

Solve as `madnlp(model; options...)` does, with the options passed as a `Dict` instead of
keywords: a keyword call is compiled for every set of option names and value types, this is
compiled once.
"""
function run_madnlp(m::AbstractNLPModel, kw::Vector{Pair{Symbol, Any}})
    nlp = MadCore.ModelBarrier(m)
    solver = MadNLPCore.build_solver(nlp, MadNLPCore.load_options(nlp, Dict{Symbol, Any}(kw)))
    return MadNLPCore.solve!(solver)
end

"""
    outcome(model, options) -> (x, y, zl, zu, objective, status, iterations, message)

A MadNLP solve, as `CNLPLib.serve` takes it.
"""
function outcome(m::AbstractNLPModel, kw::Vector{Pair{Symbol, Any}})
    s = run_madnlp(m, kw)
    return (s.solution, s.multipliers, s.multipliers_L, s.multipliers_U, s.objective,
            status_of(s.status), s.iter, "MadNLP: " * string(s.status))
end

"""
    App(name, tracevar, modules; defaults = [])

A cnlp solver library: its name (as `cnlp_solver_name` reports it), the environment
variable that turns on its compile trace, the modules whose types an option value may name,
and option defaults the caller has not set (`:linear_solver => MadCore.MumpsSolver`).
"""
struct App
    name::String
    modules::Vector{Module}
    defaults::Vector{Pair{Symbol, Any}}
    trace::CNLPLib.CompileTrace
    blas::Vector{String}
end
App(name::AbstractString, tracevar::AbstractString, modules; defaults = Pair{Symbol, Any}[]) =
    App(String(name), collect(Module, modules), collect(Pair{Symbol, Any}, defaults),
        CNLPLib.CompileTrace(tracevar), String[])

"The app's options: `opts` (`key => value` strings) as MadNLP options, with its defaults."
function options(app::App, opts)
    kw = options(opts, app.modules)
    for d in app.defaults
        any(p -> first(p) === first(d), kw) || push!(kw, d)
    end
    return kw
end

"""
    serve(app, model, nopt, keys, vals, result) -> Int64

A solve for a C caller (`CNLPLib.serve`): `model()` builds the model, options come as C
strings, the outcome goes to `result`; the app's BLAS is restored first, and its compile
trace reported when on.
"""
serve(app::App, model, nopt::Int64, keys::Ptr{Ptr{UInt8}}, vals::Ptr{Ptr{UInt8}}, result::Ptr{CNLPLib.CNLPResult}) =
    CNLPLib.serve(outcome, model, nopt, keys, vals, result;
                  options = o -> options(app, o), blas = app.blas, trace = app.trace, name = app.name)

"""
    init!(app)

From the app's `__init__`: the trace setting is read again (an image keeps the build's), and
the BLAS the library starts with is recorded, before a model library's runtime in the same
process can forward libblastrampoline elsewhere.
"""
function init!(app::App)
    app.trace.on = -1
    append!(empty!(app.blas), CNLPLib.blas_state())
    return nothing
end

"""
    precompile_image(statements)

From a library image's build script: the traced statements of runs of a built library
(`statements`, a file of `precompile(...)` lines, when it exists), and what an image misses
that a package session has from the sysimage.
"""
function precompile_image(statements::AbstractString)
    isfile(statements) && CNLPLib.eval_statements(Module(:PrecompileStatements), statements)
    # Base.merge_names builds a Symbol[...] when it merges keyword-argument names not known
    # at compile time (the solver's options)
    for n in 4:16
        precompile(Tuple{typeof(Base.getindex), Type{Symbol}, ntuple(_ -> Symbol, n)...})
    end
    precompile(Tuple{typeof(Base.getindex), Type{Symbol}, Symbol, Symbol, Symbol, Vararg{Symbol}})
    # stdout and stderr may be a file, a pipe or a terminal
    for io in (Base.PipeEndpoint, Base.TTY, IOStream), f in (println, print)
        precompile(f, (io, String))
        precompile(flush, (io,))
    end
    return nothing
end

end # module
