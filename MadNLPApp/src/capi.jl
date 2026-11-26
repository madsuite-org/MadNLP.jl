# The C entry point: `madnlp_solve(problem, noptions, keys, values, result)`, with the
# structs of include/madnlp.h, which are cnlplib's cnlp_problem and cnlp_result
# (`cnlp_solver.h`). CNLPLib carries the binding and what a solver library does around its
# solve (`CNLPLib.serve`); what is MadNLP's is here.

using CNLPLib: CNLPProblem, CNLPResult, ProblemModel

# ── options ─────────────────────────────────────────────────────────────────────────
# key=value strings, as a user wrote them, to MadNLP's options. print_level takes
# MadNLP's level names (INFO, ERROR, ...) or 1-6, and 0 for quiet; a value naming one of
# MadNLP's types (linear_solver=LapackCPUSolver) is that type; otherwise integers, numbers
# and booleans are read as such. `sb` (Ipopt's banner switch) is accepted and ignored.
const OPTION_MODULES = Module[MadCore, MadNLPCore]

function parse_value(key::Symbol, v::String)
    if key === :print_level || key === :file_print_level
        n = tryparse(Int, v)
        n === nothing && return getfield(MadCore, Symbol(uppercase(v)))::MadCore.LogLevels
        return n == 0 ? MadCore.ERROR : MadCore.LogLevels(n)
    end
    for mod in OPTION_MODULES
        isdefined(mod, Symbol(v)) && (t = getfield(mod, Symbol(v)); t isa Type) && return t
    end
    v == "true" && return true
    v == "false" && return false
    n = tryparse(Int, v); n === nothing || return n
    x = tryparse(Float64, v); x === nothing || return x
    return v
end

function options(opts)
    kw = Pair{Symbol, Any}[]
    for (k, v) in opts
        key = Symbol(k)
        key === :sb && continue
        push!(kw, key => parse_value(key, String(v)))
    end
    return kw
end

# ── solve ───────────────────────────────────────────────────────────────────────────

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

function madnlp_outcome(m, kw)
    s = run_madnlp(m, kw)
    return (s.solution, s.multipliers, s.multipliers_L, s.multipliers_U, s.objective,
            status_of(s.status), s.iter, "MadNLP: " * string(s.status))
end

# MADNLPAPP_TRACE=1: every method compiled at run time on stderr, and the total after each
# solve (what the library still lacks; a build is checked to report 0)
const TRACE = CNLPLib.CompileTrace("MADNLPAPP_TRACE")
# The BLAS this library forwards libblastrampoline to, recorded at the first call into the
# library (when all of it has started, MUMPS's LP64 OpenBLAS included) and restored before
# every solve: another Julia runtime in the process (a model library) may open the same
# libblastrampoline by name when it starts, and forward it elsewhere.
const BLAS = String[]
blas() = (isempty(BLAS) && append!(BLAS, CNLPLib.blas_state()); BLAS)

"""
    c_solve(model, nopt, keys, vals, result)

Solve for a C caller: `model()` builds the model, options come as C strings, the outcome
goes to `result`. Returns 0 when the solver ran, 1 for an invalid option, 2 for a failure.
"""
c_solve(model, nopt::Int64, keys::Ptr{Ptr{UInt8}}, vals::Ptr{Ptr{UInt8}}, result::Ptr{CNLPResult}) =
    CNLPLib.serve(madnlp_outcome, model, nopt, keys, vals, result; options, blas = blas(), trace = TRACE, name = "madnlp")
c_solve(problem::Ptr{CNLPProblem}, nopt::Int64, keys::Ptr{Ptr{UInt8}}, vals::Ptr{Ptr{UInt8}}, result::Ptr{CNLPResult}) =
    c_solve(() -> ProblemModel(problem), nopt, keys, vals, result)

# ── the library's exports (`juliac --output-lib`) ─────────────────────────────────────

const VERSION_STRING = "0.1.0"

"The C entry point: solve the problem with MadNLP (include/madnlp.h)."
Base.@ccallable function madnlp_solve(
        problem::Ptr{CNLPProblem}, nopt::Int64, keys::Ptr{Ptr{UInt8}},
        vals::Ptr{Ptr{UInt8}}, result::Ptr{CNLPResult}
    )::Int64
    return c_solve(problem, nopt, keys, vals, result)
end

Base.@ccallable function madnlp_version(buf::Ptr{UInt8}, cap::Int64)::Int64
    blas()
    return CNLPLib.copyout(VERSION_STRING, buf, cap)
end

# the cnlp solver interface (cnlp_solver.h): the same entry point, under cnlp's names
Base.@ccallable function cnlp_solve(
        problem::Ptr{CNLPProblem}, nopt::Int64, keys::Ptr{Ptr{UInt8}},
        vals::Ptr{Ptr{UInt8}}, result::Ptr{CNLPResult}
    )::Int64
    return c_solve(problem, nopt, keys, vals, result)
end

Base.@ccallable function cnlp_solver_version()::Int64
    blas()
    return Int64(CNLPLib.CNLP_SOLVER_VERSION)
end

Base.@ccallable function cnlp_solver_name(buf::Ptr{UInt8}, cap::Int64)::Int64
    blas()
    return CNLPLib.copyout("madnlp", buf, cap)
end
