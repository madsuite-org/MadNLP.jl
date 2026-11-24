# The C entry point: `madnlp_solve(problem, noptions, keys, values, result)`, with the
# structs of include/madnlp.h, which are cnlplib's cnlp_problem and cnlp_result
# (`cnlp_solver.h`; CNLPLib carries the binding). The problem is solved with MadNLP.

using CNLPLib: CNLPLib, CNLPProblem, CNLPResult, ProblemModel

# ── options ─────────────────────────────────────────────────────────────────────────
# key=value strings, as a user wrote them, to MadNLP's options. print_level takes
# MadNLP's level names (INFO, ERROR, ...) or 1-6, and 0 for quiet; a value naming one of
# MadNLP's types (linear_solver=LapackCPUSolver) is that type; otherwise integers, numbers
# and booleans are read as such. `sb` (Ipopt's banner switch) is accepted and ignored.
function parse_value(key::Symbol, v::String)
    if key === :print_level || key === :file_print_level
        n = tryparse(Int, v)
        n === nothing && return getfield(MadCore, Symbol(uppercase(v)))::MadCore.LogLevels
        return n == 0 ? MadCore.ERROR : MadCore.LogLevels(n)
    end
    isdefined(MadNLP, Symbol(v)) && (t = getfield(MadNLP, Symbol(v)); t isa Type) && return t
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

# MADNLPAPP_TRACE=1: list every method compiled at run time on stderr, as a
# `precompile(...)` statement with its compile time, and report the total after each solve
# (what the library still lacks; a build is checked to report 0).
# (-1: not yet read from the environment; reset when the library loads, since the
# build-time workload leaves its own setting in the image)
const TRACE = Ref(-1)
function trace_on()
    if TRACE[] < 0
        TRACE[] = get(ENV, "MADNLPAPP_TRACE", "0") == "1" ? 1 : 0
        if TRACE[] == 1
            ccall(:jl_force_trace_compile_timing_enable, Cvoid, ())
            Base.cumulative_compile_timing(true)
        end
    end
    return TRACE[] == 1
end
compile_ns() = Base.cumulative_compile_time_ns()[1]
# The caller is C: flush before returning, or what Julia buffered is lost when it exits.
function report_compiled(ns)
    println(stderr, string("madnlp: compiled at run time: ", round(Int, ns / 1.0e6), " ms of Julia methods"))
    flush(stdout); flush(stderr)
    return nothing
end

function c_solve(
        problem::Ptr{CNLPProblem}, nopt::Int64, keys::Ptr{Ptr{UInt8}}, vals::Ptr{Ptr{UInt8}},
        result::Ptr{CNLPResult}
    )::Int64
    trace = trace_on()
    ct0 = trace ? compile_ns() : UInt64(0)
    rc = _c_solve(problem, nopt, keys, vals, result)
    trace ? report_compiled(compile_ns() - ct0) : (flush(stdout); flush(stderr))
    return rc
end

function _c_solve(
        problem::Ptr{CNLPProblem}, nopt::Int64, keys::Ptr{Ptr{UInt8}}, vals::Ptr{Ptr{UInt8}},
        result::Ptr{CNLPResult}
    )::Int64
    kw = try
        options(CNLPLib.solver_options(nopt, keys, vals))
    catch e
        CNLPLib.set_failure!(result, "invalid option: " * sprint(showerror, e))
        return 1
    end
    try
        m = ProblemModel(problem)
        s = run_madnlp(m, kw)
        CNLPLib.set_result!(
            result, m, s.solution, s.multipliers, s.multipliers_L, s.multipliers_U,
            s.objective, status_of(s.status), s.iter, "MadNLP: " * string(s.status)
        )
        return 0
    catch e
        CNLPLib.set_failure!(result, sprint(showerror, e))
        return 2
    end
end

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
    return copyout(VERSION_STRING, buf, cap)
end

# the cnlp solver interface (cnlp_solver.h): the same entry point, under cnlp's names
Base.@ccallable function cnlp_solve(
        problem::Ptr{CNLPProblem}, nopt::Int64, keys::Ptr{Ptr{UInt8}},
        vals::Ptr{Ptr{UInt8}}, result::Ptr{CNLPResult}
    )::Int64
    return c_solve(problem, nopt, keys, vals, result)
end

Base.@ccallable function cnlp_solver_version()::Int64
    return Int64(CNLPLib.CNLP_SOLVER_VERSION)
end

Base.@ccallable function cnlp_solver_name(buf::Ptr{UInt8}, cap::Int64)::Int64
    return copyout("madnlp", buf, cap)
end

function copyout(s::String, buf::Ptr{UInt8}, cap::Int64)
    b = codeunits(s); n = length(b)
    buf == C_NULL || cap <= 0 || GC.@preserve b unsafe_copyto!(buf, pointer(b), min(n, Int(cap)))
    return Int64(n)
end
