# The C entry point: `madnlp_solve(problem, noptions, keys, values, result)`, with the
# structs of include/madnlp.h, which are cnlplib's cnlp_problem and cnlp_result
# (`cnlp_solver.h`). CNLPLib carries the binding and what a solver library does around its
# solve (`CNLPLib.serve`); what is MadNLP's is here.

using CNLPLib: CNLPProblem, CNLPResult, ProblemModel

# ── the solver (MadNLPAppCore, shared with cuMadNLPApp) ───────────────────────────────
# Option values may name MadNLP's types. MUMPS is the sparse linear solver unless the caller
# names another: MadNLP's own default would be HSL's, which this library does not carry.
# MADNLPAPP_TRACE=1: every method compiled at run time on stderr, and the total after each
# solve (what the library still lacks; a build is checked to report 0).
const APP = MadNLPAppCore.App("madnlp", "MADNLPAPP_TRACE", (MadCore, MadNLPCore);
                              defaults = [:linear_solver => MadCore.MumpsSolver])
options(opts) = MadNLPAppCore.options(APP, opts)

"""
    c_solve(model, nopt, keys, vals, result)

Solve for a C caller: `model()` builds the model, options come as C strings, the outcome
goes to `result`. Returns 0 when the solver ran, 1 for an invalid option, 2 for a failure.
"""
c_solve(model, nopt::Int64, keys::Ptr{Ptr{UInt8}}, vals::Ptr{Ptr{UInt8}}, result::Ptr{CNLPResult}) =
    MadNLPAppCore.serve(APP, model, nopt, keys, vals, result)
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
