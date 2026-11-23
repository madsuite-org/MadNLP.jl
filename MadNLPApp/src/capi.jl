# The C entry point: `madnlp_solve(problem, noptions, keys, values, result)`, with the
# structs of include/madnlp.h, which are laid out as cnlp_solver.h's cnlp_problem and
# cnlp_result. The problem's callbacks are wrapped as an NLPModel and solved with MadNLP.

struct CProblem
    ctx::Ptr{Cvoid}
    nvar::Int64
    ncon::Int64
    nnzj::Int64
    nnzh::Int64
    minimize::Int64
    x0::Ptr{Float64}
    lvar::Ptr{Float64}
    uvar::Ptr{Float64}
    lcon::Ptr{Float64}
    ucon::Ptr{Float64}
    y0::Ptr{Float64}
    zl0::Ptr{Float64}
    zu0::Ptr{Float64}
    obj::Ptr{Cvoid}
    grad::Ptr{Cvoid}
    cons::Ptr{Cvoid}
    jac_structure::Ptr{Cvoid}
    jac::Ptr{Cvoid}
    hess_structure::Ptr{Cvoid}
    hess::Ptr{Cvoid}
end

struct CResult
    x::Ptr{Float64}
    y::Ptr{Float64}
    zl::Ptr{Float64}
    zu::Ptr{Float64}
    objective::Float64
    status::Int64
    iterations::Int64
    solver_status::NTuple{128, UInt8}
end

# solver-neutral statuses (madnlp.h, cnlp_solver.h)
const SOLVED, ACCEPTABLE, INFEASIBLE, UNBOUNDED, ITERATION_LIMIT, TIME_LIMIT, ERROR = 0, 1, 2, 3, 4, 5, 6

struct CallbackError <: Exception
    name::Symbol
    status::Int64
end
Base.showerror(io::IO, e::CallbackError) = print(io, "the problem's ", e.name, " callback returned ", e.status)
check(st::Int64, name::Symbol) = st == 0 || throw(CallbackError(name, st))

"""
    CProblemModel

An `AbstractNLPModel` over a problem given as C callbacks.
"""
struct CProblemModel <: AbstractNLPModel{Float64, Vector{Float64}}
    meta::NLPModelMeta{Float64, Vector{Float64}}
    counters::Counters
    p::CProblem
    # the Jacobian in coordinates, for the products a quasi-Newton solve asks for (filled
    # on first use)
    jrows::Vector{Int64}
    jcols::Vector{Int64}
    jvals::Vector{Float64}
end

function CProblemModel(p::CProblem)
    n, m = Int(p.nvar), Int(p.ncon)
    vec(ptr, k) = k == 0 ? Float64[] : copy(unsafe_wrap(Array, ptr, k))
    y0 = p.y0 == C_NULL ? zeros(m) : vec(p.y0, m)
    meta = NLPModelMeta(
        n; ncon = m, nnzj = Int(p.nnzj), nnzh = Int(p.nnzh),
        x0 = vec(p.x0, n), lvar = vec(p.lvar, n), uvar = vec(p.uvar, n),
        y0 = y0, lcon = vec(p.lcon, m), ucon = vec(p.ucon, m), minimize = p.minimize != 0,
    )
    return CProblemModel(meta, Counters(), p, Int64[], Int64[], Float64[])
end

function NLPModels.obj(m::CProblemModel, x::AbstractVector{Float64})
    increment!(m, :neval_obj)
    f = Ref(0.0)
    check(ccall(m.p.obj, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}), m.p.ctx, x, f), :obj)
    return f[]
end

function NLPModels.grad!(m::CProblemModel, x::AbstractVector{Float64}, g::AbstractVector{Float64})
    increment!(m, :neval_grad)
    check(ccall(m.p.grad, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}), m.p.ctx, x, g), :grad)
    return g
end

function NLPModels.cons!(m::CProblemModel, x::AbstractVector{Float64}, c::AbstractVector{Float64})
    increment!(m, :neval_cons)
    m.meta.ncon == 0 && return c
    check(ccall(m.p.cons, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}), m.p.ctx, x, c), :cons)
    return c
end

function _structure(m::CProblemModel, f::Ptr{Cvoid}, nnz::Int, rows, cols, name::Symbol)
    r = Vector{Int64}(undef, nnz); c = Vector{Int64}(undef, nnz)
    nnz == 0 || check(ccall(f, Int64, (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64}), m.p.ctx, r, c), name)
    copyto!(rows, r); copyto!(cols, c)
    return rows, cols
end
NLPModels.jac_structure!(m::CProblemModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}) =
    _structure(m, m.p.jac_structure, m.meta.nnzj, rows, cols, :jac_structure)
NLPModels.hess_structure!(m::CProblemModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}) =
    _structure(m, m.p.hess_structure, m.meta.nnzh, rows, cols, :hess_structure)

function NLPModels.jac_coord!(m::CProblemModel, x::AbstractVector{Float64}, vals::AbstractVector{Float64})
    increment!(m, :neval_jac)
    m.meta.nnzj == 0 && return vals
    check(ccall(m.p.jac, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}), m.p.ctx, x, vals), :jac)
    return vals
end

function NLPModels.hess_coord!(m::CProblemModel, x::AbstractVector{Float64}, y::AbstractVector{Float64},
                               vals::AbstractVector{Float64}; obj_weight::Real = 1.0)
    increment!(m, :neval_hess)
    m.meta.nnzh == 0 && return vals
    check(ccall(m.p.hess, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Float64, Ptr{Float64}),
                m.p.ctx, x, y, Float64(obj_weight), vals), :hess)
    return vals
end
# J v and J' v from the Jacobian's coordinates (the C interface has no products)
function _jacobian!(m::CProblemModel, x::AbstractVector{Float64})
    if length(m.jrows) != m.meta.nnzj
        resize!(m.jrows, m.meta.nnzj); resize!(m.jcols, m.meta.nnzj); resize!(m.jvals, m.meta.nnzj)
        NLPModels.jac_structure!(m, m.jrows, m.jcols)
    end
    NLPModels.jac_coord!(m, x, m.jvals)
    return m
end
function NLPModels.jprod_nln!(m::CProblemModel, x::AbstractVector{Float64}, v::AbstractVector{Float64},
                              Jv::AbstractVector{Float64})
    _jacobian!(m, x)
    fill!(Jv, 0.0)
    @inbounds for k in eachindex(m.jvals)
        Jv[m.jrows[k]] += m.jvals[k] * v[m.jcols[k]]
    end
    return Jv
end
function NLPModels.jtprod_nln!(m::CProblemModel, x::AbstractVector{Float64}, v::AbstractVector{Float64},
                               Jtv::AbstractVector{Float64})
    _jacobian!(m, x)
    fill!(Jtv, 0.0)
    @inbounds for k in eachindex(m.jvals)
        Jtv[m.jcols[k]] += m.jvals[k] * v[m.jrows[k]]
    end
    return Jtv
end

NLPModels.hess_coord!(m::CProblemModel, x::AbstractVector{Float64}, vals::AbstractVector{Float64};
                      obj_weight::Real = 1.0) = NLPModels.hess_coord!(m, x, zeros(m.meta.ncon), vals; obj_weight)

# ── options ─────────────────────────────────────────────────────────────────────────
# key=value strings, as a user wrote them, to MadNLP's keyword options. print_level takes
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

function options(nopt, keys::Ptr{Ptr{UInt8}}, vals::Ptr{Ptr{UInt8}})
    kw = Pair{Symbol, Any}[]
    for k in 1:nopt
        key = Symbol(unsafe_string(unsafe_load(keys, k)))
        key === :sb && continue
        push!(kw, key => parse_value(key, unsafe_string(unsafe_load(vals, k))))
    end
    return kw
end

# ── solve ───────────────────────────────────────────────────────────────────────────

function status_of(s)
    s == MadCore.SOLVE_SUCCEEDED && return SOLVED
    s == MadCore.SOLVED_TO_ACCEPTABLE_LEVEL && return ACCEPTABLE
    s == MadCore.INFEASIBLE_PROBLEM_DETECTED && return INFEASIBLE
    s == MadCore.DIVERGING_ITERATES && return UNBOUNDED
    s == MadCore.MAXIMUM_ITERATIONS_EXCEEDED && return ITERATION_LIMIT
    s == MadCore.MAXIMUM_WALLTIME_EXCEEDED && return TIME_LIMIT
    return ERROR
end

text128(s::AbstractString) = (b = codeunits(s); ntuple(i -> i < 128 && i <= length(b) ? b[i] : 0x00, 128))

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
    solve(problem::CProblem, kwargs) -> (status code, CResult fields)

Solve with MadNLP and return the result in the convention of madnlp.h: multipliers with
`s grad f + J'y - zL + zU = 0` (s = -1 when maximizing), `zL, zU >= 0`.
"""
function solve(p::CProblem, kw)
    m = CProblemModel(p)
    stats = run_madnlp(m, kw)
    # MadNLP's (NLPModels') multipliers satisfy grad f + J'y - zL + zU = 0 in either sense;
    # the cnlp convention negates all three when maximizing.
    s = m.meta.minimize ? 1.0 : -1.0
    return stats, s
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
    println(stderr, string("madnlp: compiled at run time: ", round(Int, ns / 1e6), " ms of Julia methods"))
    flush(stdout); flush(stderr)
    return nothing
end

function c_solve(problem::Ptr{CProblem}, nopt::Int64, keys::Ptr{Ptr{UInt8}}, vals::Ptr{Ptr{UInt8}},
                 result::Ptr{CResult})::Int64
    trace = trace_on()
    ct0 = trace ? compile_ns() : UInt64(0)
    rc = _c_solve(problem, nopt, keys, vals, result)
    trace ? report_compiled(compile_ns() - ct0) : (flush(stdout); flush(stderr))
    return rc
end

function _c_solve(problem::Ptr{CProblem}, nopt::Int64, keys::Ptr{Ptr{UInt8}}, vals::Ptr{Ptr{UInt8}},
                  result::Ptr{CResult})::Int64
    r = unsafe_load(result)
    kw = try
        options(nopt, keys, vals)
    catch e
        unsafe_store!(result, CResult(r.x, r.y, r.zl, r.zu, NaN, ERROR, -1,
                                      text128("invalid option: " * sprint(showerror, e))))
        return 1
    end
    try
        p = unsafe_load(problem)
        stats, s = solve(p, kw)
        n, k = Int(p.nvar), Int(p.ncon)
        unsafe_copyto!(r.x, pointer(stats.solution), n)
        k > 0 && unsafe_copyto!(r.y, pointer(s .* stats.multipliers), k)
        unsafe_copyto!(r.zl, pointer(s .* stats.multipliers_L), n)
        unsafe_copyto!(r.zu, pointer(s .* stats.multipliers_U), n)
        unsafe_store!(result, CResult(r.x, r.y, r.zl, r.zu, stats.objective, status_of(stats.status),
                                      Int64(stats.iter), text128("MadNLP: " * string(stats.status))))
        return 0
    catch e
        unsafe_store!(result, CResult(r.x, r.y, r.zl, r.zu, NaN, ERROR, -1, text128(sprint(showerror, e))))
        return 2
    end
end

# ── the library's exports (`juliac --output-lib`) ─────────────────────────────────────

const VERSION_STRING = "0.1.0"

"The C entry point: solve the problem with MadNLP (include/madnlp.h)."
Base.@ccallable function madnlp_solve(problem::Ptr{CProblem}, nopt::Int64, keys::Ptr{Ptr{UInt8}},
                                      vals::Ptr{Ptr{UInt8}}, result::Ptr{CResult})::Int64
    return c_solve(problem, nopt, keys, vals, result)
end

Base.@ccallable function madnlp_version(buf::Ptr{UInt8}, cap::Int64)::Int64
    return copyout(VERSION_STRING, buf, cap)
end

# the cnlp solver interface (cnlp_solver.h): the same entry point, under cnlp's names
Base.@ccallable function cnlp_solve(problem::Ptr{CProblem}, nopt::Int64, keys::Ptr{Ptr{UInt8}},
                                    vals::Ptr{Ptr{UInt8}}, result::Ptr{CResult})::Int64
    return c_solve(problem, nopt, keys, vals, result)
end

Base.@ccallable function cnlp_solver_version()::Int64
    return Int64(1)   # cnlp solver interface 0.1
end

Base.@ccallable function cnlp_solver_name(buf::Ptr{UInt8}, cap::Int64)::Int64
    return copyout("madnlp", buf, cap)
end

function copyout(s::String, buf::Ptr{UInt8}, cap::Int64)
    b = codeunits(s); n = length(b)
    buf == C_NULL || cap <= 0 || GC.@preserve b unsafe_copyto!(buf, pointer(b), min(n, Int(cap)))
    return Int64(n)
end
