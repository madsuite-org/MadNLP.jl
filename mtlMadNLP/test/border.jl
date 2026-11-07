using NLPModels

# Dense Jacobian rows kept out of the condensation as a border, factored by SDSSolver's LDLᵀ.
# The problem has n sparse rows x_i - x_{i+1} <= 1 (cyclic) and one dense row sum(x) <= n/2;
# every callback is a broadcast, so the model runs on the device.
struct BorderQPGPU{T, VT} <: NLPModels.AbstractNLPModel{T, VT}
    meta::NLPModels.NLPModelMeta{T, VT}
    counters::NLPModels.Counters
    n::Int
    idx::VT      # 1:n on the device
    jvals::VT    # the constant Jacobian values
end

function BorderQPGPU(x0::VT) where {T, VT <: AbstractVector{T}}
    n = length(x0)
    m = n + 1
    meta = NLPModels.NLPModelMeta(
        n; ncon = m, nnzj = 3n, nnzh = n,
        x0 = x0, lvar = fill!(similar(x0), T(-10)), uvar = fill!(similar(x0), T(10)),
        lcon = fill!(similar(x0, m), T(-Inf)), ucon = VT([ones(T, n); T(n) / 2]),
        minimize = true, name = "BorderQPGPU",
    )
    jvals = VT([repeat([one(T), -one(T)], n); ones(T, n)])
    return BorderQPGPU{T, VT}(meta, NLPModels.Counters(), n, VT(collect(T, 1:n)), jvals)
end

NLPModels.obj(nlp::BorderQPGPU, x::AbstractVector) = sum((x .- nlp.idx) .^ 2) / 2
function NLPModels.grad!(nlp::BorderQPGPU, x::AbstractVector, g::AbstractVector)
    g .= x .- nlp.idx
    return g
end
function NLPModels.cons!(nlp::BorderQPGPU, x::AbstractVector, c::AbstractVector)
    n = nlp.n
    c[1:(n - 1)] .= view(x, 1:(n - 1)) .- view(x, 2:n)
    c[n:n] .= view(x, n:n) .- view(x, 1:1)
    c[(n + 1):(n + 1)] .= sum(x)
    return c
end
function NLPModels.jac_structure!(nlp::BorderQPGPU, I::AbstractVector, J::AbstractVector)
    n = nlp.n
    rows = Int[]; cols = Int[]
    for i in 1:n
        push!(rows, i, i); push!(cols, i, mod1(i + 1, n))
    end
    append!(rows, fill(n + 1, n)); append!(cols, 1:n)
    copyto!(I, rows); copyto!(J, cols)
    return I, J
end
function NLPModels.jac_coord!(nlp::BorderQPGPU, x::AbstractVector, V::AbstractVector)
    copyto!(V, nlp.jvals)
    return V
end
function NLPModels.hess_structure!(nlp::BorderQPGPU, I::AbstractVector, J::AbstractVector)
    copyto!(I, collect(1:nlp.n)); copyto!(J, collect(1:nlp.n))
    return I, J
end
function NLPModels.hess_coord!(nlp::BorderQPGPU, x::AbstractVector, y::AbstractVector, V::AbstractVector; obj_weight = 1.0)
    fill!(V, obj_weight)
    return V
end

@testset "SparseCondensedKKTSystem with a border" begin
    n = 40
    tol = 1.0f-3
    border = Dict{Symbol, Any}(:dense_row_abs => 8, :dense_row_rel => 0.1, :dense_row_fill_min => 0.0)
    kkt_system = MadNLPCore.SparseCondensedKKTSystem

    # with the defaults the dense row is condensed
    solver0 = MadNLPSolver(BorderQPGPU(MtlVector(zeros(Float32, n))); print_level = MadNLPCore.ERROR, kkt_system, tol)
    @test isempty(MadCore.get_kkt(solver0).border.ind_D)
    results0 = MadNLPCore.solve!(solver0)
    @test results0.status == MadNLPCore.SOLVE_SUCCEEDED

    solver = MadNLPSolver(
        BorderQPGPU(MtlVector(zeros(Float32, n)));
        print_level = MadNLPCore.ERROR, kkt_system, kkt_options = border, tol,
    )
    kkt = MadCore.get_kkt(solver)
    @test length(kkt.border.ind_D) == 1
    @test size(kkt.aug_com, 1) == n + 1
    results = MadNLPCore.solve!(solver)
    @test results.status == MadNLPCore.SOLVE_SUCCEEDED
    @test Array(results.solution) ≈ Array(results0.solution) rtol = 1.0f-2

    @test_throws ArgumentError MadNLPSolver(
        BorderQPGPU(MtlVector(zeros(Float32, n)));
        print_level = MadNLPCore.ERROR, kkt_system, kkt_options = border, tol, sds_factorization = :cholesky,
    )
end
