# Solve small problems with each HSL solver during precompilation, so their first use
# does not compile the solver. Runs only when a functional HSL library is installed.
PrecompileTools.@compile_workload begin
    if HSL.LIBHSL_isfunctional()
        for solver in (Ma27Solver, Ma57Solver, Ma77Solver, Ma86Solver, Ma97Solver)
            MadNLP._precompile_linear_solver(Float64[], solver)
        end
    end
end
