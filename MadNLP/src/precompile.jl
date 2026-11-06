@setup_workload begin
    @compile_workload begin
        # Built here, not in the setup block: HS15Model belongs to MadNLPCore, and only
        # code compiled inside `@compile_workload` is kept in this package image.
        nlp = HS15Model()
        madnlp(nlp; print_level=MadNLP.ERROR)
        _precompile_solves(Float64[])
        # a solver without inertia uses another inertia corrector, so another IPM type
        _precompile_solves(Float64[]; sizes = (100,), linear_solver = UmfpackSolver)
        for solver in (CHOLMODSolver, LapackCPUSolver)
            _precompile_linear_solver(Float64[], solver)
        end
        # LDLFactorizations throws `SQDException` on the fixed-variable problem
        _precompile_linear_solver(Float64[], LDLSolver; fixed = (false,))
        # the HSL solvers, when a functional HSL library is installed
        if HSL.LIBHSL_isfunctional()
            for solver in (Ma27Solver, Ma57Solver, Ma77Solver, Ma86Solver, Ma97Solver)
                _precompile_linear_solver(Float64[], solver)
            end
        end
    end
end
