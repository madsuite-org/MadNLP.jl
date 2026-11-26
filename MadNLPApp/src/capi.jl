# The C entry point: `madnlp_solve(problem, noptions, keys, values, result)`, with the
# structs of include/madnlp.h, which are cnlplib's cnlp_problem and cnlp_result
# (`cnlp_solver.h`). CNLPLib carries the binding and what a solver library does around its
# solve (`CNLPLib.serve`); what is MadNLP's is here.

using CNLPLib: CNLPProblem, CNLPResult, ProblemModel

# ── options and the solve (MadNLPAppBase, shared with cuMadNLPApp) ────────────────────
# key=value strings, as a user wrote them, to MadNLP's options (MadNLPAppBase.parse_value).
# MUMPS is the sparse linear solver unless the caller names another: MadNLP's own default
# would be HSL's, which this library does not carry.
const OPTION_MODULES = Module[MadCore, MadNLPCore]
function options(opts)
    kw = MadNLPAppBase.options(opts, OPTION_MODULES)
    any(p -> first(p) === :linear_solver, kw) || push!(kw, :linear_solver => MadCore.MumpsSolver)
    return kw
end
const madnlp_outcome = MadNLPAppBase.outcome

# MADNLPAPP_TRACE=1: every method compiled at run time on stderr, and the total after each
# solve (what the library still lacks; a build is checked to report 0)
const TRACE = CNLPLib.CompileTrace("MADNLPAPP_TRACE")
# The BLAS this library forwards libblastrampoline to, recorded by `__init__` and restored
# before every solve: another Julia runtime in the process (a model library) may open the
# same libblastrampoline by name when it starts, and forward it elsewhere.
const BLAS = String[]

"""
    c_solve(model, nopt, keys, vals, result)

Solve for a C caller: `model()` builds the model, options come as C strings, the outcome
goes to `result`. Returns 0 when the solver ran, 1 for an invalid option, 2 for a failure.
"""
c_solve(model, nopt::Int64, keys::Ptr{Ptr{UInt8}}, vals::Ptr{Ptr{UInt8}}, result::Ptr{CNLPResult}) =
    CNLPLib.serve(madnlp_outcome, model, nopt, keys, vals, result; options, blas = BLAS, trace = TRACE, name = "madnlp")
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
    return Int64(CNLPLib.CNLP_SOLVER_VERSION)
end

Base.@ccallable function cnlp_solver_name(buf::Ptr{UInt8}, cap::Int64)::Int64
    return CNLPLib.copyout("madnlp", buf, cap)
end
