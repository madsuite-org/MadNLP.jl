using Test
using NLPModels
using ParametricNLPModels
using LinearAlgebra
using MadNLPTests
using Random

# Convex QP with constant Hessian P and Jacobian A
function _solve_QP(; n=10, m=5, fixed_variables=Int[], equality_cons=[1, 3], kwargs...)
    nlp = MadNLPTests.DenseDummyQP(zeros(n); m=m, fixed_variables=fixed_variables, equality_cons=equality_cons)
    solver = MadNLPSolver(nlp; print_level=MadNLP.ERROR, tol=1e-8, kwargs...)
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

# Test model implementing the ParametricNLPModels interface with constant Hxθ and Jθ
struct ParametricModel{T, M} <: NLPModels.AbstractNLPModel{T, Vector{T}}
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    counters::NLPModels.Counters
    inner::M
    θ::Vector{T}
    Hxθ::Matrix{T}
    Jθ::Matrix{T}
end
ParametricModel(inner, θ, Hxθ, Jθ) = ParametricModel(inner.meta, inner.counters, inner, θ, Hxθ, Jθ)
NLPModels.obj(nlp::ParametricModel, x::AbstractVector) = NLPModels.obj(nlp.inner, x)
NLPModels.grad!(nlp::ParametricModel, x::AbstractVector, g::AbstractVector) = NLPModels.grad!(nlp.inner, x, g)
NLPModels.cons!(nlp::ParametricModel, x::AbstractVector, c::AbstractVector) = NLPModels.cons!(nlp.inner, x, c)
NLPModels.jtprod!(nlp::ParametricModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector) = NLPModels.jtprod!(nlp.inner, x, v, Jtv)
NLPModels.jac_structure!(nlp::ParametricModel, rows::AbstractVector, cols::AbstractVector) = NLPModels.jac_structure!(nlp.inner, rows, cols)
NLPModels.jac_coord!(nlp::ParametricModel, x::AbstractVector, vals::AbstractVector) = NLPModels.jac_coord!(nlp.inner, x, vals)
NLPModels.hess_structure!(nlp::ParametricModel, rows::AbstractVector, cols::AbstractVector) = NLPModels.hess_structure!(nlp.inner, rows, cols)
NLPModels.hess_coord!(nlp::ParametricModel, x::AbstractVector, y::AbstractVector, vals::AbstractVector; obj_weight=1.0) =
    NLPModels.hess_coord!(nlp.inner, x, y, vals; obj_weight=obj_weight)
ParametricNLPModels.get_par_meta(nlp::ParametricModel) =
    ParametricNLPModelMeta(; npar=length(nlp.θ), nnzj_par=length(nlp.Jθ), nnzh_par=length(nlp.Hxθ), grad_par_available=false)
ParametricNLPModels.jac_par_structure!(nlp::ParametricModel, rows, cols) = _dense_structure!(nlp.Jθ, rows, cols)
ParametricNLPModels.jac_par_coord!(nlp::ParametricModel, x, vals) = copyto!(vals, nlp.Jθ)
ParametricNLPModels.hess_par_structure!(nlp::ParametricModel, rows, cols) = _dense_structure!(nlp.Hxθ, rows, cols)
ParametricNLPModels.hess_par_coord!(nlp::ParametricModel, x, y, vals; obj_weight=1.0) = copyto!(vals, nlp.Hxθ)

function _dense_structure!(A, rows, cols)
    for (k, I) in enumerate(CartesianIndices(A))
        rows[k], cols[k] = Tuple(I)
    end
    return rows, cols
end

sparse_options = Dict{Symbol, Any}(
    :callback=>MadNLP.SparseCallback,
    :kkt_system=>MadNLP.SparseKKTSystem,
)
dense_options = Dict{Symbol, Any}(
    :callback=>MadNLP.DenseCallback,
    :kkt_system=>MadNLP.DenseKKTSystem,
    :linear_solver=>MadNLP.LapackCPUSolver,
)

@testset "Sensitivity: backsolve_kkt!" begin
    n, m, eq = 10, 5, [1, 3]
    Random.seed!(1)
    p = randn(3n + m, 2)

    @testset "$name" for (name, options, fixed) in [
        ("sparse", sparse_options, Int[]),
        ("sparse + scaling", merge(sparse_options, Dict{Symbol, Any}(:nlp_scaling_max_gradient=>0.1)), Int[]),
        ("sparse + fixed variables", sparse_options, [9, 10]),
        ("dense + fixed variables", dense_options, [9, 10]),
    ]
        nlp, solver, stats = _solve_QP(; n=n, m=m, fixed_variables=fixed, equality_cons=eq, options...)
        d = MadNLP.backsolve_kkt!(solver, p)
        dvec = MadNLP.backsolve_kkt!(solver, p[:, 1])
        @test dvec.dx == d.dx[:, 1] && dvec.dy == d.dy[:, 1] && dvec.dzl == d.dzl[:, 1] && dvec.dzu == d.dzu[:, 1]
        @test MadNLP.backsolve_kkt!(solver, p[1:n + m, :]) == MadNLP.backsolve_kkt!(solver, [p[1:n + m, :]; zeros(2n, 2)])
        @test_throws ArgumentError MadNLP.backsolve_kkt!(solver, zeros(2n + m, 2))
        free = setdiff(1:n, fixed)
        rx, ry, rzl, rzu = _kkt_residuals(solver, d, p)
        @test norm(rx[free, :], Inf) <= 1e-6
        @test norm(ry[eq, :], Inf) <= 1e-6
        @test size(rzl, 1) == size(rzu, 1) == length(free)
        @test norm(rzl, Inf) <= 1e-6
        @test norm(rzu, Inf) <= 1e-6
        @test iszero(d.dx[fixed, :]) && iszero(d.dzl[fixed, :]) && iszero(d.dzu[fixed, :])
    end
end

@testset "Sensitivity: sensitivity" begin
    n, m, k = 10, 5, 3
    Random.seed!(1)
    Hxθ, Jθ = randn(n, k), randn(m, k)
    nlp = ParametricModel(MadNLPTests.DenseDummyQP(zeros(n); m=m, equality_cons=[1, 3]), [0.5, -1.0, 2.0], Hxθ, Jθ)
    solver = MadNLPSolver(nlp; print_level=MadNLP.ERROR, tol=1e-8, sparse_options...)
    MadNLP.solve!(solver)
    s = MadNLP.sensitivity(solver, Hxθ, Jθ)
    @test s == MadNLP.backsolve_kkt!(solver, -[Hxθ; Jθ])
    @test MadNLP.sensitivity(solver) == s
    dθ = [0.1, -0.2, 0.3]
    sd = MadNLP.sensitivity(solver, dθ)
    @test sd.dx ≈ s.dx * dθ
    @test sd.dy ≈ s.dy * dθ
    @test sd.dzl ≈ s.dzl * dθ
    @test sd.dzu ≈ s.dzu * dθ
    @test all(map(≈, MadNLP.sensitivity(solver, Hxθ, Jθ, dθ), sd))
end
