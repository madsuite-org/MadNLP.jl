# The solver, for both entry points: options from C strings, MadNLP's status in cnlp's terms,
# and the solve. Option values may name MadNLP's and cuMadNLP's types.

const OPTION_MODULES = Module[cuMadNLP, MadCore, MadNLPCore]

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
    options(opts; device = false)

`opts` (`key => value` strings) as MadNLP options. On the CPU, MUMPS is the sparse linear
solver unless the caller names another (MadNLP's own default would be HSL's, which this
library does not carry); on the GPU, cuMadNLP's defaults apply.
"""
function options(opts; device::Bool = false)
    kw = options(opts, OPTION_MODULES)
    device || any(p -> first(p) === :linear_solver, kw) || push!(kw, :linear_solver => MadCore.MumpsSolver)
    return kw
end

# MADNLPAPP_TRACE=1: every method compiled at run time on stderr, and the total after each
# solve (what the library still lacks; a build is checked to report 0); a device solve also
# reports the model's first GPU evaluations and the GPU kernels compiled at run time.
const TRACE = CNLPLib.CompileTrace("MADNLPAPP_TRACE")
# The BLAS this library forwards libblastrampoline to, recorded by `__init__` and restored
# before every solve: another Julia runtime in the process (a model library) may open the
# same libblastrampoline by name when it starts, and forward it elsewhere.
const BLAS = String[]

"""
    c_solve(model, nopt, keys, vals, result; device = false)

Solve for a C caller: `model()` builds the model, options come as C strings, the outcome
goes to `result`. Returns 0 when the solver ran, 1 for an invalid option, 2 for a failure.
"""
c_solve(model, nopt::Int64, keys::Ptr{Ptr{UInt8}}, vals::Ptr{Ptr{UInt8}}, result::Ptr{CNLPResult}; device::Bool = false) =
    CNLPLib.serve(outcome, model, nopt, keys, vals, result;
                  options = o -> options(o; device), blas = BLAS, trace = TRACE, name = "madnlp")
c_solve(problem::Ptr{CNLPProblem}, nopt::Int64, keys::Ptr{Ptr{UInt8}}, vals::Ptr{Ptr{UInt8}}, result::Ptr{CNLPResult}) =
    c_solve(() -> ProblemModel(problem), nopt, keys, vals, result)
