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

# Ruiz NLP scaling: variables, constraints and objective scaled at the starting point.

# min 1e6 (x1 - 1e-3)^2 + (x2 / 1e3 - 2)^2 + (x3 - 7)^2 / 4  s.t.  1e4 x1 + x2 / 1e3 + x3^2 >= 100 (active),
# x1 >= 0, x3 fixed or free.
struct BadUnitsModel{T} <: NLPModels.AbstractNLPModel{T, Vector{T}}
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    counters::NLPModels.Counters
end
function BadUnitsModel(; fix_x3 = false)
    lvar = [0.0, -Inf, fix_x3 ? 7.0 : -Inf]
    uvar = [Inf, Inf, fix_x3 ? 7.0 : Inf]
    meta = NLPModels.NLPModelMeta(3; ncon = 1, nnzj = 3, nnzh = 3, x0 = [0.0, 0.0, 1.0],
        lvar, uvar, lcon = [100.0], ucon = [Inf])
    return BadUnitsModel(meta, NLPModels.Counters())
end
NLPModels.obj(::BadUnitsModel, x::AbstractVector) = 1e6 * (x[1] - 1e-3)^2 + (x[2] / 1e3 - 2)^2 + (x[3] - 7)^2 / 4
function NLPModels.grad!(::BadUnitsModel, x::AbstractVector, g::AbstractVector)
    g[1] = 2e6 * (x[1] - 1e-3); g[2] = 2 * (x[2] / 1e3 - 2) / 1e3; g[3] = (x[3] - 7) / 2
    return g
end
NLPModels.cons!(::BadUnitsModel, x::AbstractVector, c::AbstractVector) = (c[1] = 1e4 * x[1] + x[2] / 1e3 + x[3]^2; c)
NLPModels.jac_structure!(::BadUnitsModel, I::AbstractVector{T}, J::AbstractVector{T}) where T = (I .= 1; J .= 1:3; (I, J))
NLPModels.jac_coord!(::BadUnitsModel, x::AbstractVector, J::AbstractVector) = (J .= (1e4, 1e-3, 2 * x[3]); J)
NLPModels.jtprod!(m::BadUnitsModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector) = (Jtv .= NLPModels.jac_coord(m, x) .* v[1]; Jtv)
NLPModels.hess_structure!(::BadUnitsModel, I::AbstractVector{T}, J::AbstractVector{T}) where T = (I .= 1:3; J .= 1:3; (I, J))
function NLPModels.hess_coord!(::BadUnitsModel, x::AbstractVector, y::AbstractVector, H::AbstractVector; obj_weight = 1.0)
    H .= (2e6 * obj_weight, 2e-6 * obj_weight, obj_weight / 2 + 2 * y[1])
    return H
end

# ∇f + Jᵀ y - zl + zu of the original model at the returned point (MadNLP's sign convention)
function kkt_residual(nlp, stats)
    x = stats.solution
    r = NLPModels.grad(nlp, x) .+ NLPModels.jac(nlp, x)' * stats.multipliers .- stats.multipliers_L .+ stats.multipliers_U
    return maximum(abs, r)
end

# HS15 has two local minima, and the scaling can change which one the solve finds; there only the
# optimality conditions are checked.
@testset "Ruiz NLP scaling ($name)" for (name, nlp, same_solution) in (
    ("HS15", MadNLPTests.HS15Model(), false),
    ("HS15 x0 outside bounds", MadNLPTests.HS15Model(; x0 = [1.0, 1.0]), false),
    ("scaled circle", ScaledCircleModel(), true),
    ("bad units", BadUnitsModel(), true),
    ("bad units, fixed variable", BadUnitsModel(; fix_x3 = true), true),
)
    sparse = (; callback = MadNLP.SparseCallback, kkt_system = MadNLP.SparseKKTSystem, linear_solver = MadNLP.MumpsSolver)
    ref = madnlp(nlp; sparse..., print_level = MadNLP.ERROR)
    solver = MadNLPSolver(nlp; sparse..., nlp_scaling_method = :ruiz, print_level = MadNLP.ERROR)
    stats = MadNLP.solve!(solver)
    cb = solver.cb
    @test stats.status == MadNLP.SOLVE_SUCCEEDED
    @test cb.var_scaled[]
    # the factors are clamped powers of 2
    for v in (cb.var_scale, cb.con_scale, [cb.obj_scale[]])
        @test all(d -> 1e-4 <= d <= 1e4 && exp2(round(log2(d))) == d, v)
    end
    if same_solution
        @test stats.solution ≈ ref.solution rtol = 1e-5 atol = 1e-8
        @test stats.objective ≈ ref.objective rtol = 1e-6 atol = 1e-10
        @test stats.multipliers ≈ ref.multipliers rtol = 1e-4 atol = 1e-6
    end
    @test MadNLP.get_constr_viol_unscaled(solver) <= 1e-6
    @test kkt_residual(nlp, stats) <= 1e-5 * max(1, maximum(abs, NLPModels.grad(nlp, stats.solution)))
    @test stats.constraints ≈ NLPModels.cons(nlp, stats.solution)
end

@testset "Ruiz NLP scaling scales the variables" begin
    nlp = BadUnitsModel()
    solver = MadNLPSolver(nlp; nlp_scaling_method = :ruiz, print_level = MadNLP.ERROR)
    MadNLP.initialize!(solver)
    # x1 enters with 1e4 in the Jacobian and 2e6 in the Hessian, x2 with 1e-3 and 2e-6
    @test solver.cb.var_scale[1] < 1 < solver.cb.var_scale[2]
    # the solver's x is the model's x divided by the factors
    x = MadNLP.variable(solver.x)
    xm = similar(x)
    MadNLP.unpack_x!(xm, solver.cb, x)
    @test xm ≈ x .* solver.cb.var_scale
    # the scaled Jacobian is J .* con_scale .* var_scale
    J = zeros(1, 3)
    MadNLP._eval_jac_wrapper!(solver.cb, x, J)
    @test vec(J) ≈ NLPModels.jac_coord(nlp, xm) .* solver.cb.con_scale[1] .* solver.cb.var_scale
end

@testset "Ruiz NLP scaling needs a SparseCallback" begin
    @test_throws ErrorException MadNLP.solve!(MadNLPSolver(MadNLPTests.HS15Model();
        callback = MadNLP.DenseCallback, kkt_system = MadNLP.DenseKKTSystem,
        linear_solver = MadNLP.LapackCPUSolver, nlp_scaling_method = :ruiz,
        print_level = MadNLP.ERROR, rethrow_error = true))
end

@testset "Ruiz NLP scaling bounds" begin
    nlp = BadUnitsModel()
    sparse = (; callback = MadNLP.SparseCallback, kkt_system = MadNLP.SparseKKTSystem, linear_solver = MadNLP.MumpsSolver)
    wide = MadNLPSolver(nlp; sparse..., nlp_scaling_method = :ruiz, print_level = MadNLP.ERROR)
    MadNLP.initialize!(wide)
    # without the tighter bounds, some factor is outside [1e-2, 1e2]
    @test any(d -> !(1e-2 <= d <= 1e2), vcat(wide.cb.var_scale, wide.cb.con_scale, wide.cb.obj_scale[]))
    solver = MadNLPSolver(nlp; sparse..., nlp_scaling_method = :ruiz, nlp_scaling_ruiz_min = 1e-2,
        nlp_scaling_ruiz_max = 1e2, print_level = MadNLP.ERROR)
    stats = MadNLP.solve!(solver)
    @test stats.status == MadNLP.SOLVE_SUCCEEDED
    @test all(d -> 1e-2 <= d <= 1e2, vcat(solver.cb.var_scale, solver.cb.con_scale, solver.cb.obj_scale[]))
end

@testset "Ruiz NLP scaling, scaling down only" begin
    # nlp_scaling_ruiz_max = 1 only scales down, as gradient-based scaling does
    nlp = BadUnitsModel()
    sparse = (; callback = MadNLP.SparseCallback, kkt_system = MadNLP.SparseKKTSystem, linear_solver = MadNLP.MumpsSolver)
    solver = MadNLPSolver(nlp; sparse..., nlp_scaling_method = :ruiz, nlp_scaling_ruiz_max = 1.0, print_level = MadNLP.ERROR)
    stats = MadNLP.solve!(solver)
    @test stats.status == MadNLP.SOLVE_SUCCEEDED
    @test all(d -> 1e-4 <= d <= 1, vcat(solver.cb.var_scale, solver.cb.con_scale, solver.cb.obj_scale[]))
    # the objective, with Hessian entry 2e6, is still scaled down
    @test solver.cb.obj_scale[] < 1
    @test any(<(1), solver.cb.var_scale)
end
