# The soft restoration phase takes full steps and falls back to the robust phase when the
# primal-dual error does not decrease enough. On lootsma under ShiftedPreset with a dense LU
# factorization and the barrier floor sigma_min = 0.1, it reaches a trial point where the
# constraints are not finite (sqrt of a slightly negative x). That trial point must fall back
# too; it was kept, and the solve ran on with NaN until the iteration limit.

const MT = MadNLPTests  # JuMP's Model and macros, as imported by MadNLPTests

function lootsma_resto_model()
    m = MT.Model()
    MT.@variable(m, par == 6.0)
    MT.@variable(m, 0 <= x[i = 1:3] <= 5, start = 0.0)
    MT.@constraint(m, -sqrt(x[1]) - sqrt(x[2]) + sqrt(x[3]) >= 0.0)
    MT.@constraint(m, sqrt(x[1]) + sqrt(x[2]) + sqrt(x[3]) >= 4.0)
    MT.@objective(m, Min, x[1]^3 + 11.0 * x[1] - par * sqrt(x[1]) + x[3])
    return MadNLP.SparseWrapperModel(Array, MT.NLPModelsJuMP.MathOptNLPModel(m))
end

@testset "Soft restoration: a non-finite trial point falls back" begin
    qf = MadNLP.QualityFunctionUpdate{Float64}(sigma_max = 2.0, sigma_min = 0.1, mu_min = 1.0e-11)
    res = MadNLP.madnlp(
        lootsma_resto_model(); preset = MadNLP.ShiftedPreset, linear_solver = MadNLP.LapackCPUSolver,
        lapack_algorithm = MadNLP.LU, barrier = qf, print_level = MadNLP.ERROR,
    )
    @test res.status != MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
    @test res.iter < 100
    @test all(isfinite, res.solution)
end
