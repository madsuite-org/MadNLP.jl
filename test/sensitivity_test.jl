using Test
using NLPModels
using NLPModelsExtensions
using LinearAlgebra
using MadNLPTests
using Random

# Convex QP with constant Hessian P and Jacobian A
function _solve_QP(; n = 10, m = 5, fixed_variables = Int[], equality_cons = [1, 3], kwargs...)
    nlp = MadNLPTests.DenseDummyQP(zeros(n); m = m, fixed_variables = fixed_variables, equality_cons = equality_cons)
    solver = MadNLPSolver(nlp; print_level = MadNLP.ERROR, tol = 1.0e-8, kwargs...)
    stats = MadNLP.solve!(solver)
    return nlp, solver, stats
end

# Residual [rx; ry; rzl; rzu] := M d - p, with the Hessian P and Jacobian A of the QP
# and the variable bounds rows equal to zl, zu, x - l, u - x
function _kkt_residuals(solver, d, p)
    nlp, cb, kkt = MadNLP.get_nlp(solver), MadNLP.get_cb(solver), MadNLP.get_kkt(solver)
    n, m = nlp.meta.nvar, nlp.meta.ncon
    px, py, pzl, pzu = p[1:n, :], p[n .+ (1:m), :], p[n + m .+ (1:n), :], p[2n + m .+ (1:n), :]
    rx = nlp.P * d.dx .+ nlp.A' * d.dy .- d.dzl .+ d.dzu .- px
    ry = nlp.A * d.dx .- py
    tomodel = cb isa MadNLP.SparseCallback && cb.fixed_handler isa MadNLP.MakeParameter ? cb.fixed_handler.free : 1:n
    il = findall(<=(length(tomodel)), kkt.ind_lb); vl = tomodel[kkt.ind_lb[il]]
    iu = findall(<=(length(tomodel)), kkt.ind_ub); vu = tomodel[kkt.ind_ub[iu]]
    rzl = kkt.l_lower[il] ./ cb.obj_scale[] .* d.dx[vl, :] .- kkt.l_diag[il] .* d.dzl[vl, :] .- pzl[vl, :]
    rzu = .-kkt.u_lower[iu] ./ cb.obj_scale[] .* d.dx[vu, :] .- kkt.u_diag[iu] .* d.dzu[vu, :] .- pzu[vu, :]
    return rx, ry, rzl, rzu
end

# Dense KKT matrix M at the final point of the QP with equality constraints only, rows x, y, zl, zu,
# with the bounds relaxed by MadNLP and the bound rows zl dx + (x - l) dzl and -zu dx + (u - x) dzu
function _dense_kkt(solver, stats)
    nlp = MadNLP.get_nlp(solver)
    n, m = nlp.meta.nvar, nlp.meta.ncon
    x, zl, zu = stats.solution, stats.multipliers_L, stats.multipliers_U
    l, u = MadNLP.full(solver.xl)[1:n], MadNLP.full(solver.xu)[1:n]
    L, U = findall(isfinite, nlp.meta.lvar), findall(isfinite, nlp.meta.uvar)
    EL, EU = Matrix(1.0I, n, n)[:, L], Matrix(1.0I, n, n)[:, U]
    nl, nu = length(L), length(U)
    M = [
        nlp.P                  nlp.A'        -EL                    EU
        nlp.A                  zeros(m, m)   zeros(m, nl)           zeros(m, nu)
        Diagonal(zl[L]) * EL'  zeros(nl, m)  Diagonal(x[L] - l[L])  zeros(nl, nu)
        -Diagonal(zu[U]) * EU' zeros(nu, m)  zeros(nu, nl)          Diagonal(u[U] - x[U])
    ]
    return M, L, U
end

# Test model implementing the ParametricNLPModels interface with constant Hxθ and Jθ
struct ParametricModel{T, M} <: NLPModels.AbstractNLPModel{T, Vector{T}}
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    counters::NLPModels.Counters
    inner::M
    θ::Vector{T}
    Hxθ::Matrix{T}
    Jθ::Matrix{T}
    nnzj_lvar_par::Int
end
ParametricModel(inner, θ, Hxθ, Jθ; nnzj_lvar_par = 0) = ParametricModel(inner.meta, inner.counters, inner, θ, Hxθ, Jθ, nnzj_lvar_par)
NLPModels.obj(nlp::ParametricModel, x::AbstractVector) = NLPModels.obj(nlp.inner, x)
NLPModels.grad!(nlp::ParametricModel, x::AbstractVector, g::AbstractVector) = NLPModels.grad!(nlp.inner, x, g)
NLPModels.cons!(nlp::ParametricModel, x::AbstractVector, c::AbstractVector) = NLPModels.cons!(nlp.inner, x, c)
NLPModels.jtprod!(nlp::ParametricModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector) = NLPModels.jtprod!(nlp.inner, x, v, Jtv)
NLPModels.jac_structure!(nlp::ParametricModel, rows::AbstractVector, cols::AbstractVector) = NLPModels.jac_structure!(nlp.inner, rows, cols)
NLPModels.jac_coord!(nlp::ParametricModel, x::AbstractVector, vals::AbstractVector) = NLPModels.jac_coord!(nlp.inner, x, vals)
NLPModels.hess_structure!(nlp::ParametricModel, rows::AbstractVector, cols::AbstractVector) = NLPModels.hess_structure!(nlp.inner, rows, cols)
NLPModels.hess_coord!(nlp::ParametricModel, x::AbstractVector, y::AbstractVector, vals::AbstractVector; obj_weight = 1.0) =
    NLPModels.hess_coord!(nlp.inner, x, y, vals; obj_weight = obj_weight)
NLPModelsExtensions.get_par_meta(nlp::ParametricModel) =
    ParametricNLPModelMeta(; npar = length(nlp.θ), nnzj_par = length(nlp.Jθ), nnzh_par = length(nlp.Hxθ), nnzj_lvar_par = nlp.nnzj_lvar_par, grad_par_available = false)
NLPModelsExtensions.jac_par_structure!(nlp::ParametricModel, rows, cols) = _dense_structure!(nlp.Jθ, rows, cols)
NLPModelsExtensions.jac_par_coord!(nlp::ParametricModel, x, vals) = copyto!(vals, nlp.Jθ)
NLPModelsExtensions.hess_par_structure!(nlp::ParametricModel, rows, cols) = _dense_structure!(nlp.Hxθ, rows, cols)
NLPModelsExtensions.hess_par_coord!(nlp::ParametricModel, x, y, vals; obj_weight = 1.0) = copyto!(vals, nlp.Hxθ)

function _dense_structure!(A, rows, cols)
    for (k, I) in enumerate(CartesianIndices(A))
        rows[k], cols[k] = Tuple(I)
    end
    return rows, cols
end

# Test model maximizing -f of another model, same solution with obj_sign = -1
struct MaximizeModel{T, M} <: NLPModels.AbstractNLPModel{T, Vector{T}}
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    counters::NLPModels.Counters
    inner::M
end
MaximizeModel(inner) = MaximizeModel(NLPModels.NLPModelMeta(inner.meta; minimize = false), NLPModels.Counters(), inner)
NLPModels.obj(nlp::MaximizeModel, x::AbstractVector) = -NLPModels.obj(nlp.inner, x)
function NLPModels.grad!(nlp::MaximizeModel, x::AbstractVector, g::AbstractVector)
    NLPModels.grad!(nlp.inner, x, g)
    return g .*= -1
end
NLPModels.cons!(nlp::MaximizeModel, x::AbstractVector, c::AbstractVector) = NLPModels.cons!(nlp.inner, x, c)
NLPModels.jtprod!(nlp::MaximizeModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector) = NLPModels.jtprod!(nlp.inner, x, v, Jtv)
NLPModels.jac_structure!(nlp::MaximizeModel, rows::AbstractVector, cols::AbstractVector) = NLPModels.jac_structure!(nlp.inner, rows, cols)
NLPModels.jac_coord!(nlp::MaximizeModel, x::AbstractVector, vals::AbstractVector) = NLPModels.jac_coord!(nlp.inner, x, vals)
NLPModels.hess_structure!(nlp::MaximizeModel, rows::AbstractVector, cols::AbstractVector) = NLPModels.hess_structure!(nlp.inner, rows, cols)
NLPModels.hess_coord!(nlp::MaximizeModel, x::AbstractVector, y::AbstractVector, vals::AbstractVector; obj_weight = 1.0) =
    NLPModels.hess_coord!(nlp.inner, x, y, vals; obj_weight = -obj_weight)

# Pirnay et al. (2012) sIPOPT example with x1⁴ in the objective and c3 = x1² + x2² ≤ 1
struct PirnayModel{T} <: NLPModels.AbstractNLPModel{T, Vector{T}}
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    counters::NLPModels.Counters
    θ::Vector{T}
end
PirnayModel(θ) = PirnayModel(
    NLPModels.NLPModelMeta(
        3; ncon = 3, x0 = fill(0.5, 3), lvar = zeros(3), uvar = fill(Inf, 3),
        lcon = [0.0, 0.0, -Inf], ucon = [0.0, 0.0, 1.0], nnzj = 9, nnzh = 3
    ), NLPModels.Counters(), θ
)
_pirnay_jac(nlp, x) = [6 3 2; nlp.θ[2] 1 -1; 2x[1] 2x[2] 0]
NLPModels.obj(nlp::PirnayModel, x::AbstractVector) = sum(abs2, x) + x[1]^4
NLPModels.grad!(nlp::PirnayModel, x::AbstractVector, g::AbstractVector) = copyto!(g, [2x[1] + 4x[1]^3, 2x[2], 2x[3]])
NLPModels.cons!(nlp::PirnayModel, x::AbstractVector, c::AbstractVector) =
    copyto!(c, [6x[1] + 3x[2] + 2x[3] - nlp.θ[1], nlp.θ[2] * x[1] + x[2] - x[3] - 1, x[1]^2 + x[2]^2])
NLPModels.jac_structure!(nlp::PirnayModel, rows::AbstractVector, cols::AbstractVector) = _dense_structure!(zeros(3, 3), rows, cols)
NLPModels.jac_coord!(nlp::PirnayModel, x::AbstractVector, vals::AbstractVector) = copyto!(vals, _pirnay_jac(nlp, x))
NLPModels.jtprod!(nlp::PirnayModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector) = copyto!(Jtv, _pirnay_jac(nlp, x)' * v)
NLPModels.hess_structure!(nlp::PirnayModel, rows::AbstractVector, cols::AbstractVector) = (copyto!(rows, 1:3), copyto!(cols, 1:3))
NLPModels.hess_coord!(nlp::PirnayModel, x::AbstractVector, y::AbstractVector, vals::AbstractVector; obj_weight = 1.0) =
    copyto!(vals, obj_weight .* [2 + 12x[1]^2, 2, 2] .+ y[3] .* [2, 2, 0])
NLPModelsExtensions.get_par_meta(nlp::PirnayModel) = ParametricNLPModelMeta(; npar = 2, nnzj_par = 2, nnzh_par = 1, grad_par_available = false)
NLPModelsExtensions.jac_par_structure!(nlp::PirnayModel, rows, cols) = (copyto!(rows, [1, 2]), copyto!(cols, [1, 2]))
NLPModelsExtensions.jac_par_coord!(nlp::PirnayModel, x, vals) = copyto!(vals, [-1, x[1]])
NLPModelsExtensions.hess_par_structure!(nlp::PirnayModel, rows, cols) = (copyto!(rows, [1]), copyto!(cols, [2]))
NLPModelsExtensions.hess_par_coord!(nlp::PirnayModel, x, y, vals; obj_weight = 1.0) = copyto!(vals, [y[2]])

sparse_options = Dict{Symbol, Any}(
    :callback => MadNLP.SparseCallback,
    :kkt_system => MadNLP.SparseKKTSystem,
)
dense_options = Dict{Symbol, Any}(
    :callback => MadNLP.DenseCallback,
    :kkt_system => MadNLP.DenseKKTSystem,
    :linear_solver => MadNLP.LapackCPUSolver,
)
relax_bound = Dict{Symbol, Any}(:fixed_variable_treatment => MadNLP.RelaxBound)

@testset "Sensitivity: backsolve_kkt!" begin
    n, m, eq = 10, 5, [1, 3]
    Random.seed!(1)
    p = randn(3n + m, 2)

    @testset "$name" for (name, options, fixed) in [
            ("sparse", sparse_options, Int[]),
            ("sparse + scaling", merge(sparse_options, Dict{Symbol, Any}(:nlp_scaling_max_gradient => 0.1)), Int[]),
            ("sparse + fixed variables", merge(sparse_options, relax_bound), [9, 10]),
        ]
        nlp, solver, stats = _solve_QP(; n = n, m = m, fixed_variables = fixed, equality_cons = eq, options...)
        d = MadNLP.backsolve_kkt!(solver, p)
        dvec = MadNLP.backsolve_kkt!(solver, p[:, 1])
        @test dvec.dx == d.dx[:, 1] && dvec.dy == d.dy[:, 1] && dvec.dzl == d.dzl[:, 1] && dvec.dzu == d.dzu[:, 1]
        @test_throws ArgumentError MadNLP.backsolve_kkt!(solver, zeros(n + m, 2))
        free = setdiff(1:n, fixed)
        rx, ry, rzl, rzu = _kkt_residuals(solver, d, p)
        @test norm(rx[free, :], Inf) <= 1.0e-6
        @test norm(ry[eq, :], Inf) <= 1.0e-6
        @test size(rzl, 1) == size(rzu, 1) == n
        @test norm(rzl, Inf) <= 1.0e-6
        @test norm(rzu, Inf) <= 1.0e-6
    end
end

@testset "Sensitivity: sensitivity" begin
    n, m, k = 10, 5, 3
    Random.seed!(1)
    Hxθ, Jθ = randn(n, k), randn(m, k)
    nlp = ParametricModel(MadNLPTests.DenseDummyQP(zeros(n); m = m, equality_cons = [1, 3]), [0.5, -1.0, 2.0], Hxθ, Jθ)
    solver = MadNLPSolver(nlp; print_level = MadNLP.ERROR, tol = 1.0e-8, sparse_options...)
    MadNLP.solve!(solver)
    s = MadNLP.sensitivity!(solver, Hxθ, Jθ)
    @test s == MadNLP.backsolve_kkt!(solver, [-Hxθ; -Jθ; zeros(2n, k)])
    @test MadNLP.sensitivity!(solver) == s
    dθ = [0.1, -0.2, 0.3]
    sd = MadNLP.sensitivity_jvp!(solver, dθ)
    @test sd.dx ≈ s.dx * dθ
    @test sd.dy ≈ s.dy * dθ
    @test sd.dzl ≈ s.dzl * dθ
    @test sd.dzu ≈ s.dzu * dθ
    @test all(map(≈, MadNLP.sensitivity_jvp!(solver, Hxθ, Jθ, dθ), sd))
end

@testset "Sensitivity: refactorize_kkt" begin
    n, m, k = 10, 5, 3
    Random.seed!(1)
    Hxθ, Jθ = randn(n, k), randn(m, k)

    @testset "$kkt_system" for (kkt_system, callback, tol) in [
            (MadNLP.SparseKKTSystem, MadNLP.SparseCallback, 1.0e-10),
            (MadNLP.SparseCondensedKKTSystem, MadNLP.SparseCallback, 1.0e-4),
            (MadNLP.SparseUnreducedKKTSystem, MadNLP.SparseCallback, 1.0e-10),
            (MadNLP.ScaledSparseKKTSystem, MadNLP.SparseCallback, 1.0e-10),
            (MadNLP.DenseKKTSystem, MadNLP.DenseCallback, 1.0e-10),
            (MadNLP.DenseCondensedKKTSystem, MadNLP.DenseCallback, 1.0e-10),
        ]
        nlp, solver, stats = _solve_QP(; n = n, m = m, equality_cons = collect(1:m), kkt_system = kkt_system, callback = callback)
        s = MadNLP.sensitivity!(solver, Hxθ, Jθ)
        M, L, U = _dense_kkt(solver, stats)
        nl, nu = length(L), length(U)
        d = -M \ [Hxθ; Jθ; zeros(nl + nu, k)]
        @test isapprox(s.dx, d[1:n, :]; atol = tol, rtol = tol)
        @test isapprox(s.dy, d[n .+ (1:m), :]; atol = tol, rtol = tol)
        @test isapprox(s.dzl[L, :], d[n + m .+ (1:nl), :]; atol = tol, rtol = tol)
        @test isapprox(s.dzu[U, :], d[n + m + nl .+ (1:nu), :]; atol = tol, rtol = tol)
        @test MadNLP.sensitivity!(solver, Hxθ, Jθ; refactorize_kkt = false) == s
    end
end

@testset "Sensitivity: sensitivity_vjp!" begin
    n, m, k, eq = 10, 5, 3, [1, 3]
    Random.seed!(1)
    Hxθ, Jθ, dθ = randn(n, k), randn(m, k), randn(k)
    vx, vy, vzl, vzu = randn(n), randn(m), randn(n), randn(n)
    v_dot(s) = dot(vx, s.dx) + dot(vy, s.dy) + dot(vzl, s.dzl) + dot(vzu, s.dzu)
    qp(; fixed = Int[]) = MadNLPTests.DenseDummyQP(zeros(n); m = m, fixed_variables = fixed, equality_cons = eq)

    @testset "$name" for (name, nlp, options) in [
            ("sparse", qp(), sparse_options),
            ("sparse + scaling", qp(), merge(sparse_options, Dict{Symbol, Any}(:nlp_scaling_max_gradient => 0.1))),
            ("sparse + fixed variables", qp(fixed = [9, 10]), merge(sparse_options, relax_bound)),
            ("sparse + maximize", MaximizeModel(qp()), sparse_options),
            ("sparse + maximize + fixed variables", MaximizeModel(qp(fixed = [9, 10])), merge(sparse_options, relax_bound)),
            ("SparseCondensedKKTSystem", qp(), merge(sparse_options, Dict{Symbol, Any}(:kkt_system => MadNLP.SparseCondensedKKTSystem))),
            ("SparseUnreducedKKTSystem", qp(), merge(sparse_options, Dict{Symbol, Any}(:kkt_system => MadNLP.SparseUnreducedKKTSystem))),
            ("ScaledSparseKKTSystem", qp(), merge(sparse_options, Dict{Symbol, Any}(:kkt_system => MadNLP.ScaledSparseKKTSystem))),
            ("DenseCondensedKKTSystem", qp(), merge(dense_options, Dict{Symbol, Any}(:kkt_system => MadNLP.DenseCondensedKKTSystem))),
        ]
        solver = MadNLPSolver(nlp; print_level = MadNLP.ERROR, tol = 1.0e-8, options...)
        MadNLP.solve!(solver)
        jvp = MadNLP.sensitivity_jvp!(solver, Hxθ, Jθ, dθ)
        vjp = MadNLP.sensitivity_vjp!(solver, Hxθ, Jθ; vx, vy, vzl, vzu)
        @test dot(vjp, dθ) ≈ v_dot(jvp)
    end

    nlp = ParametricModel(qp(), zeros(k), Hxθ, Jθ)
    solver = MadNLPSolver(nlp; print_level = MadNLP.ERROR, tol = 1.0e-8, sparse_options...)
    MadNLP.solve!(solver)
    @test dot(MadNLP.sensitivity_vjp!(solver; vx, vy, vzl, vzu), dθ) ≈ v_dot(MadNLP.sensitivity_jvp!(solver, dθ))
    @test MadNLP.sensitivity_vjp!(solver; vx) ≈ MadNLP.sensitivity!(solver).dx' * vx
    @test_throws ArgumentError MadNLP.sensitivity_vjp!(solver)
    @test_throws DimensionMismatch MadNLP.sensitivity_vjp!(solver; vy = zeros(m + 1))
end

@testset "Sensitivity: bounds warning" begin
    n, m, k = 10, 5, 3
    Random.seed!(1)
    Hxθ, Jθ = randn(n, k), randn(m, k)
    captured_stdout(f) = mktemp() do path, io
        redirect_stdout(f, io)
        flush(io)
        return read(path, String)
    end

    @testset "nnzj_lvar_par = $nnzj_lvar_par" for nnzj_lvar_par in (0, 1)
        nlp = ParametricModel(MadNLPTests.DenseDummyQP(zeros(n); m = m, equality_cons = [1, 3]), zeros(k), Hxθ, Jθ; nnzj_lvar_par)
        solver = MadNLPSolver(nlp; print_level = MadNLP.WARN, tol = 1.0e-8, sparse_options...)
        MadNLP.solve!(solver)
        for f in (() -> MadNLP.sensitivity!(solver), () -> MadNLP.sensitivity_jvp!(solver, ones(k)), () -> MadNLP.sensitivity_vjp!(solver; vx = ones(n)))
            @test occursin("bounds that depend on parameters", captured_stdout(f)) == (nnzj_lvar_par > 0)
        end
    end
end

@testset "Sensitivity: errors" begin
    n, m = 10, 5
    p = zeros(3n + m)
    nlp, solver, stats = _solve_QP(; n = n, m = m, sparse_options...)
    @test_throws ArgumentError MadNLP.sensitivity!(solver)
    @test_throws DimensionMismatch MadNLP.sensitivity!(solver, zeros(n + 1, 2), zeros(m - 1, 2))
    nlp, solver, stats = _solve_QP(; n = n, m = m, fixed_variables = [9, 10], sparse_options...)
    @test_throws ArgumentError MadNLP.backsolve_kkt!(solver, p)
    nlp, solver, stats = _solve_QP(; n = n, m = m, fixed_variables = [9, 10], dense_options...)
    @test_throws ArgumentError MadNLP.backsolve_kkt!(solver, p)
    nlp, solver, stats = _solve_QP(; n = n, m = m, hessian_approximation = MadNLP.CompactLBFGS, sparse_options...)
    @test_throws ArgumentError MadNLP.backsolve_kkt!(solver, p)
end

@testset "Sensitivity: finite differences" begin
    θ, h = [4.5, 1.0], 1.0e-5
    function s(θ)
        stats = MadNLP.solve!(MadNLPSolver(PirnayModel(θ); print_level = MadNLP.ERROR, tol = 1.0e-11))
        return [stats.solution; stats.multipliers; stats.multipliers_L; stats.multipliers_U]
    end
    fd = reduce(hcat, (s(θ + h * e) - s(θ - h * e)) / 2h for e in eachcol(Matrix(1.0I, 2, 2)))
    solver = MadNLPSolver(PirnayModel(θ); print_level = MadNLP.ERROR)
    MadNLP.solve!(solver)
    S = MadNLP.sensitivity!(solver)
    @test [S.dx; S.dy; S.dzl; S.dzu] ≈ fd rtol = 1.0e-4
end
