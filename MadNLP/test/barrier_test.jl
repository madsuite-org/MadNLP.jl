# lootsma under ShiftedPreset with a dense LU factorization. With the barrier parameter
# floored at sigma_min times the average complementarity (sigma_min = 0.1), the floor raises
# mu when the average complementarity is far above it, and the solve stalls.

const MT = MadNLPTests  # JuMP's Model and macros, as imported by MadNLPTests

function lootsma_model()
    m = MT.Model()
    MT.@variable(m, par == 6.0)
    MT.@variable(m, 0 <= x[i = 1:3] <= 5, start = 0.0)
    MT.@constraint(m, -sqrt(x[1]) - sqrt(x[2]) + sqrt(x[3]) >= 0.0)
    MT.@constraint(m, sqrt(x[1]) + sqrt(x[2]) + sqrt(x[3]) >= 4.0)
    MT.@objective(m, Min, x[1]^3 + 11.0 * x[1] - par * sqrt(x[1]) + x[3])
    return MadNLP.SparseWrapperModel(Array, MT.NLPModelsJuMP.MathOptNLPModel(m))
end

@testset "Barrier: ShiftedPreset with a dense LU factorization" begin
    opts = (;
        preset = MadNLP.ShiftedPreset, linear_solver = MadNLP.LapackCPUSolver,
        lapack_algorithm = MadNLP.LU, print_level = MadNLP.ERROR,
    )
    # mu decreases at most tenfold per iteration and is never raised by the bound
    res = MadNLP.madnlp(lootsma_model(); opts...)
    @test res.status == MadNLP.SOLVE_SUCCEEDED
    @test res.solution[2:4] ≈ [0.07415998565403112, 2.9848713863700236, 4.0000304145340415] atol = 1.0e-4
end

# The adaptive barrier rules leave the monotone mode when the progress test passes. The test
# uses the line-search filter, which is reset to its single entry (theta_max, -Inf) whenever mu
# changes; until the line search adds an entry, any point with theta < theta_max passes it.
@testset "Adaptive barrier: leave the monotone mode only after the line search has added a filter entry" begin
    for barrier in (MadNLP.QualityFunctionUpdate{Float64}(), MadNLP.LOQOUpdate{Float64}())
        solver = MadNLP.MadNLPSolver(MadNLPTests.HS15Model(); barrier, print_level = MadNLP.ERROR)
        MadNLP.initialize!(solver)
        b = solver.opt.barrier
        b.free_mode = false
        mu0 = solver.mu
        @test length(solver.filter) == 1
        @test MadNLP._check_progress(b, solver)
        MadNLP.update_barrier!(b, solver, 1.0)
        # still monotone: mu kept or decreased, not set by the adaptive rule
        @test !b.free_mode
        @test solver.mu <= mu0
        @test length(solver.filter) == 1
        # an entry that the current point improves on: back to the free mode, mu from the adaptive rule
        push!(solver.filter, (MadNLP.get_theta(solver) + 1, MadNLP.get_varphi(solver) + 1))
        @test MadNLP._check_progress(b, solver)
        n_update = b isa MadNLP.QualityFunctionUpdate ? b.n_update : 0
        MadNLP.update_barrier!(b, solver, 1.0)
        @test b.free_mode
        if b isa MadNLP.QualityFunctionUpdate
            @test b.n_update == n_update + 1
        else
            @test solver.mu == MadNLP.get_adaptive_mu(solver, b)
        end
    end
end
