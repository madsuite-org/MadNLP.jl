# lootsma under ShiftedPreset with a dense LU factorization. With the barrier parameter
# floored at sigma_min times the average complementarity (sigma_min = 0.1), the floor raises
# mu when the average complementarity is far above it, and the solve stalls; the soft
# restoration phase then reached a point where the constraints are not finite (sqrt of a
# negative number) and kept it, so the solve ran on with NaN until the iteration limit.

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
    opts = (; preset = MadNLP.ShiftedPreset, linear_solver = MadNLP.LapackCPUSolver,
        lapack_algorithm = MadNLP.LU, print_level = MadNLP.ERROR)
    # mu decreases at most tenfold per iteration and is never raised by the bound
    res = MadNLP.madnlp(lootsma_model(); opts...)
    @test res.status == MadNLP.SOLVE_SUCCEEDED
    @test res.solution[2:4] ≈ [0.07415998565403112, 2.9848713863700236, 4.0000304145340415] atol = 1.0e-4

    # the floor sigma_min = 0.1 stalls; a non-finite trial point of the soft restoration phase
    # now falls back to the robust phase instead of being kept, so the solve stops early
    qf = MadNLP.QualityFunctionUpdate{Float64}(sigma_max = 2.0, sigma_min = 0.1, mu_min = 1.0e-11)
    res = MadNLP.madnlp(lootsma_model(); opts..., barrier = qf)
    @test res.status != MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
    @test res.iter < 100
end
