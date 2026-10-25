# Solve small problems with each HSL solver during precompilation, so their first use
# does not compile the solver. Runs only when a functional HSL library is installed.
PrecompileTools.@compile_workload begin
    if HSL.LIBHSL_isfunctional()
        for solver in (Ma27Solver, Ma57Solver, Ma77Solver, Ma86Solver, Ma97Solver)
            # one size: the two sizes of the default workload matter only for GPU kernels
            MadNLP._precompile_solves(Float64[]; sizes = (100,), linear_solver = solver)
        end
    end
end
