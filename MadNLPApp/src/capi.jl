# The C entry point: `madnlp_solve(problem, noptions, keys, values, result)`, with the
# structs of include/madnlp.h, which are cnlplib's cnlp_problem and cnlp_result
# (`cnlp_solver.h`). CNLPLib carries the binding and what a solver library does around its
# solve (`CNLPLib.serve`); what is MadNLP's is here.


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

"The cnlp solver interface's device entry point (`cnlp_solver.h` 0.2): CUDA."
Base.@ccallable function cnlp_solve_device(
        p::Ptr{CNLPProblem}, d::Ptr{CNLPDeviceProblem}, nopt::Int64,
        keys::Ptr{Ptr{UInt8}}, vals::Ptr{Ptr{UInt8}}, result::Ptr{CNLPResult}
    )::Int64
    dev = unsafe_load(d).device
    if dev != CNLPLib.CNLP_DEVICE_CUDA
        CNLPLib.set_failure!(result, "madnlp: device $dev is not supported (CUDA is $(CNLPLib.CNLP_DEVICE_CUDA))")
        return Int64(1)
    end
    if !CUDACore.functional()
        CNLPLib.set_failure!(result, "madnlp: no usable CUDA GPU")
        return Int64(2)
    end
    return c_solve_device(p, d, nopt, keys, vals, result)
end

Base.@ccallable function cnlp_solver_version()::Int64
    return Int64(CNLPLib.CNLP_SOLVER_VERSION)
end

Base.@ccallable function cnlp_solver_name(buf::Ptr{UInt8}, cap::Int64)::Int64
    return CNLPLib.copyout("madnlp", buf, cap)
end
