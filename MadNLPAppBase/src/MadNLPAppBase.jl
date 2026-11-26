"""
    MadNLPAppBase

What MadNLP's two cnlp solver libraries share, MadNLPApp (CPU) and cuMadNLPApp (GPU): solver
options from C strings, the solve itself, and MadNLP's status in cnlp's terms.
"""
module MadNLPAppBase

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

end # module
