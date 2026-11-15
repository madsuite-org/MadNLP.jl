# Gradient-based NLP scaling and the unscaled termination test, as in Ipopt.

# min (x - 2)^2  s.t.  a (x^2 - 1) = 0,  lvar <= x.
struct ScaledCircleModel{T} <: NLPModels.AbstractNLPModel{T, Vector{T}}
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    counters::NLPModels.Counters
    a::T
end
function ScaledCircleModel(; a = 1e6, x0 = [3.0], lvar = [-Inf])
    meta = NLPModels.NLPModelMeta(
        1; ncon = 1, nnzj = 1, nnzh = 1, x0 = x0, lvar = lvar, uvar = [Inf],
        lcon = [0.0], ucon = [0.0],
    )
    return ScaledCircleModel(meta, NLPModels.Counters(), a)
end
NLPModels.obj(::ScaledCircleModel, x::AbstractVector) = (x[1] - 2)^2
NLPModels.grad!(::ScaledCircleModel, x::AbstractVector, g::AbstractVector) = (g[1] = 2 * (x[1] - 2); g)
NLPModels.cons!(m::ScaledCircleModel, x::AbstractVector, c::AbstractVector) = (c[1] = m.a * (x[1]^2 - 1); c)
NLPModels.jac_structure!(::ScaledCircleModel, I::AbstractVector{T}, J::AbstractVector{T}) where T = (I .= 1; J .= 1; (I, J))
NLPModels.jac_coord!(m::ScaledCircleModel, x::AbstractVector, J::AbstractVector) = (J[1] = 2 * m.a * x[1]; J)
NLPModels.hess_structure!(::ScaledCircleModel, I::AbstractVector{T}, J::AbstractVector{T}) where T = (I .= 1; J .= 1; (I, J))
function NLPModels.hess_coord!(m::ScaledCircleModel, x::AbstractVector, y::AbstractVector, H::AbstractVector; obj_weight = 1.0)
    H[1] = 2 * obj_weight + 2 * m.a * y[1]
    return H
end

# min x - log(x)  s.t.  x >= 0. The gradient is -Inf at the bound.
struct LogBarrierModel{T} <: NLPModels.AbstractNLPModel{T, Vector{T}}
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    counters::NLPModels.Counters
end
LogBarrierModel(; x0 = [0.0]) =
    LogBarrierModel(NLPModels.NLPModelMeta(1; nnzh = 1, x0 = x0, lvar = [0.0], uvar = [Inf]), NLPModels.Counters())
NLPModels.obj(::LogBarrierModel, x::AbstractVector) = x[1] - log(x[1])
NLPModels.grad!(::LogBarrierModel, x::AbstractVector, g::AbstractVector) = (g[1] = 1 - 1 / x[1]; g)
NLPModels.cons!(::LogBarrierModel, x::AbstractVector, c::AbstractVector) = c
NLPModels.jac_structure!(::LogBarrierModel, I::AbstractVector{T}, J::AbstractVector{T}) where T = (I, J)
NLPModels.jac_coord!(::LogBarrierModel, x::AbstractVector, J::AbstractVector) = J
NLPModels.hess_structure!(::LogBarrierModel, I::AbstractVector{T}, J::AbstractVector{T}) where T = (I .= 1; J .= 1; (I, J))
function NLPModels.hess_coord!(::LogBarrierModel, x::AbstractVector, y::AbstractVector, H::AbstractVector; obj_weight = 1.0)
    H[1] = obj_weight / x[1]^2
    return H
end

# Scaling factors computed from the model at x, as gradient-based scaling defines them.
function expected_scaling(nlp, x)
    obj = min(1, 100 / maximum(abs, NLPModels.grad(nlp, x)))
    J = NLPModels.jac(nlp, x)
    con = [min(1, 100 / max(1, maximum(abs, J[i, :]))) for i in 1:size(J, 1)]
    return obj, con
end

@testset "Scaling computed at the user starting point ($callback)" for (callback, nlp, kkt_system, linear_solver) in (
    # x0 lies on the lower bound x >= 0.5, so the bound push moves it.
    (MadNLP.SparseCallback, ScaledCircleModel(; x0 = [0.5], lvar = [0.5]), MadNLP.SparseKKTSystem, MadNLP.MumpsSolver),
    # x0 lies on the upper bound x1 <= 0.5 of HS15.
    (MadNLP.DenseCallback, MadNLPTests.HS15Model(; x0 = [0.5, 1.0]), MadNLP.DenseKKTSystem, MadNLP.LapackCPUSolver),
)
    x0 = copy(NLPModels.get_x0(nlp))
    solver = MadNLPSolver(nlp; callback, kkt_system, linear_solver, print_level = MadNLP.ERROR)
    MadNLP.initialize!(solver)
    xp = copy(MadNLP.variable(solver.x))
    @test xp != x0
    obj, con = expected_scaling(nlp, x0)
    obj_p, con_p = expected_scaling(nlp, xp)
    @test solver.cb.obj_scale[] ≈ obj
    @test solver.cb.con_scale ≈ con
    # The test discriminates: the pushed point gives different factors.
    @test (obj, con) != (obj_p, con_p)
end

@testset "Scaling falls back to the pushed point when the user point is not finite" begin
    nlp = LogBarrierModel()
    solver = MadNLPSolver(nlp; print_level = MadNLP.ERROR)
    stats = MadNLP.solve!(solver)
    @test isfinite(solver.cb.obj_scale[]) && solver.cb.obj_scale[] > 0
    @test stats.status == MadNLP.SOLVE_SUCCEEDED
    @test stats.solution[1] ≈ 1 atol = 1e-6
end

@testset "Termination checks the unscaled problem" begin
    # The constraint row is scaled by 100 / 6e6, so a loose scaled tolerance stops the solve
    # with an unscaled violation far above constr_viol_tol unless the unscaled test holds it.
    nlp = ScaledCircleModel()
    viol(stats) = abs(NLPModels.cons(nlp, stats.solution)[1])

    stats_scaled_only = madnlp(nlp; tol = 1e-1, constr_viol_tol = Inf, dual_inf_tol = Inf,
        compl_inf_tol = Inf, print_level = MadNLP.ERROR)
    @test stats_scaled_only.status == MadNLP.SOLVE_SUCCEEDED
    @test viol(stats_scaled_only) > 1e-4

    stats = madnlp(nlp; tol = 1e-1, print_level = MadNLP.ERROR)
    @test stats.status == MadNLP.SOLVE_SUCCEEDED
    @test viol(stats) <= 1e-4
    @test stats.iter > stats_scaled_only.iter

    # The unscaled measures match the original model at the returned point.
    solver = MadNLPSolver(nlp; print_level = MadNLP.ERROR)
    MadNLP.solve!(solver)
    @test MadNLP.get_constr_viol_unscaled(solver) ≈ abs(NLPModels.cons(nlp, MadNLP.variable(solver.x))[1])
    @test MadNLP.get_inf_du_unscaled(solver) <= 1.0
    @test MadNLP.get_inf_compl_unscaled(solver) <= 1e-4
end
