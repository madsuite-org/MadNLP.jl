# Regions are timed with device events, so the counters are finite and
# collected by the end of the solve even though the solver never syncs.
function test_event_timer(Arr, opts; sparse=false)
    nlp = if sparse
        MadNLP.SparseWrapperModel(Arr, MadNLPTests.DenseDummyQP(zeros(Float64, 50); m=10))
    else
        MadNLPTests.DenseDummyQP(Arr(zeros(Float64, 50)); m=10)
    end
    solver = MadNLPSolver(nlp; print_level=MadNLP.ERROR, opts...)
    stats = MadNLP.solve!(solver)
    cnt = solver.cnt
    @test stats.status == MadNLP.SOLVE_SUCCEEDED
    @test cnt.timer isa MadNLPGPU.EventTimer
    @test isempty(cnt.timer.pending)
    @test isfinite(cnt.linear_solver_time) && cnt.linear_solver_time > 0
    @test isfinite(cnt.eval_function_time) && cnt.eval_function_time > 0
    @test cnt.linear_solver_time + cnt.eval_function_time <= cnt.total_time
end

@testset "EventTimer" begin
    if HAS_CUDA && CUDA.functional()
        @testset "CUDSS" test_event_timer(CuArray, (;
            callback=MadNLP.SparseCallback,
            kkt_system=MadNLP.SparseKKTSystem,
            linear_solver=MadNLPGPU.CUDSSSolver,
        ); sparse=true)
        @testset "LapackCUDA" test_event_timer(CuArray, (;
            callback=MadNLP.DenseCallback,
            kkt_system=MadNLP.DenseKKTSystem,
            linear_solver=MadNLPGPU.LapackCUDASolver,
        ))
    end
    if HAS_AMDGPU && AMDGPU.functional()
        @testset "LapackROCm" test_event_timer(ROCArray, (;
            callback=MadNLP.DenseCallback,
            kkt_system=MadNLP.DenseKKTSystem,
            linear_solver=MadNLPGPU.LapackROCmSolver,
        ))
    end
end
