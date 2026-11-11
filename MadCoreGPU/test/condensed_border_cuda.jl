using CUDA
using NLPModels
using SparseArrays

# Dense Jacobian rows kept out of the condensation as a border, factored by cuDSS's LDLᵀ.
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

# rows 1:n/2 x_{2i-1} - x_{2i} = 0 (equalities), rows n/2+1:3n/2-1 x_i - x_{i+1} <= 1/2,
# row 3n/2 sum(x) <= n/4 (dense, active at the solution)
struct EqBorderQPGPU{T, VT} <: NLPModels.AbstractNLPModel{T, VT}
    meta::NLPModels.NLPModelMeta{T, VT}
    counters::NLPModels.Counters
    n::Int
    target::VT
    jvals::VT
end

function EqBorderQPGPU(x0::VT) where {T, VT <: AbstractVector{T}}
    n = length(x0)
    me = n ÷ 2
    m = me + n
    meta = NLPModels.NLPModelMeta(
        n; ncon = m, nnzj = 2me + 2 * (n - 1) + n, nnzh = n,
        x0 = x0, lvar = fill!(similar(x0), T(-10)), uvar = fill!(similar(x0), T(10)),
        lcon = VT([zeros(T, me); fill(T(-Inf), n - 1); T(-Inf)]),
        ucon = VT([zeros(T, me); fill(T(1 // 2), n - 1); T(n) / 4]),
        minimize = true, name = "EqBorderQPGPU",
    )
    jvals = VT([repeat([one(T), -one(T)], me + n - 1); ones(T, n)])
    return EqBorderQPGPU{T, VT}(meta, NLPModels.Counters(), n, VT(collect(T, 1:n) ./ n), jvals)
end

NLPModels.obj(nlp::EqBorderQPGPU, x::AbstractVector) = sum((x .- nlp.target) .^ 2) / 2
function NLPModels.grad!(nlp::EqBorderQPGPU, x::AbstractVector, g::AbstractVector)
    g .= x .- nlp.target
    return g
end
function NLPModels.cons!(nlp::EqBorderQPGPU, x::AbstractVector, c::AbstractVector)
    n = nlp.n
    me = n ÷ 2
    view(c, 1:me) .= view(x, 1:2:n) .- view(x, 2:2:n)
    view(c, (me + 1):(me + n - 1)) .= view(x, 1:(n - 1)) .- view(x, 2:n)
    c[(me + n):(me + n)] .= sum(x)
    return c
end
function NLPModels.jac_structure!(nlp::EqBorderQPGPU, I::AbstractVector, J::AbstractVector)
    n = nlp.n
    me = n ÷ 2
    rows = Int[]; cols = Int[]
    for i in 1:me
        push!(rows, i, i); push!(cols, 2i - 1, 2i)
    end
    for i in 1:(n - 1)
        push!(rows, me + i, me + i); push!(cols, i, i + 1)
    end
    append!(rows, fill(me + n, n)); append!(cols, 1:n)
    copyto!(I, rows); copyto!(J, cols)
    return I, J
end
function NLPModels.jac_coord!(nlp::EqBorderQPGPU, x::AbstractVector, V::AbstractVector)
    copyto!(V, nlp.jvals)
    return V
end
function NLPModels.hess_structure!(nlp::EqBorderQPGPU, I::AbstractVector, J::AbstractVector)
    copyto!(I, collect(1:nlp.n)); copyto!(J, collect(1:nlp.n))
    return I, J
end
function NLPModels.hess_coord!(nlp::EqBorderQPGPU, x::AbstractVector, y::AbstractVector, V::AbstractVector; obj_weight = 1.0)
    fill!(V, obj_weight)
    return V
end

@testset "MadNLP -- CUDSSSolver: SparseCondensedKKTSystem with a border" begin
    n = 40
    border = Dict{Symbol, Any}(:dense_row_abs => 8, :dense_row_rel => 0.1, :dense_row_fill_min => 0.0)
    results_ref = MadNLP.madnlp(
        BorderQPGPU(zeros(Float64, n)); kkt_system = MadNLP.SparseKKTSystem,
        linear_solver = MadNLP.UmfpackSolver, print_level = MadNLP.ERROR, tol = 1.0e-8
    )
    @test results_ref.status == MadNLP.SOLVE_SUCCEEDED

    nlp = BorderQPGPU(CUDA.zeros(Float64, n))
    solver = MadNLPSolver(
        nlp;
        print_level = MadNLP.ERROR,
        preset = MadNLP.CondensedPreset,
        kkt_system = MadNLP.SparseCondensedKKTSystem,
        linear_solver = MadCoreGPU.CUDSSSolver,
        kkt_options = border,
        tol = 1.0e-8,
    )
    kkt = MadNLP.get_kkt(solver)
    @test length(kkt.border.ind_D) == 1
    @test size(kkt.aug_com, 1) == n + 1
    results = MadNLP.solve!(solver)
    @test results.status == MadNLP.SOLVE_SUCCEEDED
    @test results.objective ≈ results_ref.objective atol = 1.0e-6
    # the condensed system relaxes the bounds: the solution agrees with the augmented one to that accuracy
    @test Array(results.solution) ≈ Array(results_ref.solution) atol = 1.0e-4
    @test Array(results.multipliers)[n + 1] ≈ results_ref.multipliers[n + 1] atol = 1.0e-4

    # with a threshold above the row's nonzeros the dense row is condensed: the plain condensed system
    solver0 = MadNLPSolver(
        BorderQPGPU(CUDA.zeros(Float64, n)); print_level = MadNLP.ERROR, preset = MadNLP.CondensedPreset,
        kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = MadCoreGPU.CUDSSSolver, tol = 1.0e-8,
        kkt_options = Dict{Symbol, Any}(:condense_max_nnz => n + 1),
    )
    @test isempty(MadNLP.get_kkt(solver0).border.ind_D)
    results0 = MadNLP.solve!(solver0)
    @test results0.status == MadNLP.SOLVE_SUCCEEDED
    # the two condensed solves, with and without the border, agree to the tolerance
    @test Array(results0.solution) ≈ Array(results.solution) atol = 1.0e-6
end

# ShiftedPreset (the GPU default) on the condensed system: the equalities, and the inequality rows
# of 20 or more nonzeros, are kept as the border
@testset "MadNLP -- CUDSSSolver: ShiftedPreset on SparseCondensedKKTSystem with equalities" begin
    n = 40
    me = n ÷ 2
    ref = MadNLP.madnlp(EqBorderQPGPU(CUDA.zeros(Float64, n)); print_level = MadNLP.ERROR)
    @test ref.status == MadNLP.SOLVE_SUCCEEDED
    @test sum(Array(ref.solution)) ≈ n / 4 atol = 1.0e-6

    solver = MadNLPSolver(
        EqBorderQPGPU(CUDA.zeros(Float64, n)); kkt_system = MadNLP.SparseCondensedKKTSystem,
        print_level = MadNLP.ERROR,
    )
    kkt = MadNLP.get_kkt(solver)
    @test Array(kkt.ind_eq) == collect(1:me)
    @test Array(kkt.border.ind_D) == [1:me; me + n]
    @test size(kkt.aug_com, 1) == n + me + 1
    results = MadNLP.solve!(solver)
    @test results.status == MadNLP.SOLVE_SUCCEEDED
    @test Array(results.solution) ≈ Array(ref.solution) atol = 1.0e-6
    @test Array(results.multipliers) ≈ Array(ref.multipliers) atol = 1.0e-6
end

# The Ruiz sweeps on the augmented system [H + Σx  Jᵀ; J  Σd - Σs⁻¹] written out on host arrays
# (H lower triangular, Jt with a column per constraint): the scaling of the condensed block and of
# the rows kept in the border, as AugmentedRuizState must reproduce it.
function reference_augmented_ruiz(H, Jt, p, du, Σs, ind_ineq, ind_D, sweeps)
    n, m = size(Jt)
    a = copy(du); a[ind_ineq] .-= 1 ./ Σs
    d = ones(n + m)
    for _ in 1:sweeps
        r = zeros(n + m)
        for i in 1:n
            r[i] = abs(p[i]) * d[i]^2
        end
        for j in 1:n, k in H.colptr[j]:(H.colptr[j + 1] - 1)
            i = H.rowval[k]
            v = i == j ? abs(H.nzval[k] + p[i]) * d[i]^2 : abs(H.nzval[k]) * d[i] * d[j]
            r[i] = max(r[i], v); r[j] = max(r[j], v)
        end
        for c in 1:m
            r[n + c] = abs(a[c]) * d[n + c]^2
        end
        for c in 1:m, k in Jt.colptr[c]:(Jt.colptr[c + 1] - 1)
            i = Jt.rowval[k]
            v = abs(Jt.nzval[k]) * d[i] * d[n + c]
            r[i] = max(r[i], v); r[n + c] = max(r[n + c], v)
        end
        d .= ifelse.(r .> 0, d ./ sqrt.(r), d)
    end
    return [d[1:n]; d[n .+ ind_D]]
end

# the sweeps run on the device and must give the scaling of the sweeps written out on host copies
@testset "MadNLP -- CUDSSSolver: AugmentedRuizScaler on SparseCondensedKKTSystem" begin
    n = 40
    kkt_options = Dict{Symbol, Any}(:condense_max_nnz => 10)   # the dense row (n nonzeros) is kept
    solver = MadNLPSolver(
        BorderQPGPU(CUDA.zeros(Float64, n)); preset = MadNLP.CondensedPreset, kkt_scaler = MadNLP.AugmentedRuizScaler(),
        kkt_options, linear_solver = MadCoreGPU.CUDSSSolver, max_iter = 5, print_level = MadNLP.ERROR,
    )
    MadNLP.solve!(solver)
    kkt = MadNLP.get_kkt(solver)
    @test Array(kkt.border.ind_D) == [n + 1]
    @test kkt.aug_scaling isa MadNLP.MadCore.AugmentedRuizState
    @test kkt.aug_scaling.absnz isa CuVector
    MadNLP.build_kkt!(kkt)
    M = getfield(kkt.linear_solver, :wrapped)
    @test M.d isa CuVector
    ns = length(kkt.ind_ineq)
    pr = Array(kkt.pr_diag)
    d = reference_augmented_ruiz(
        SparseMatrixCSC(kkt.hess_com), SparseMatrixCSC(kkt.jt_csc), pr[1:n], Array(kkt.du_diag),
        pr[(n + 1):(n + ns)], Array(kkt.ind_ineq), Array(kkt.border.ind_D), 4,
    )
    @test Array(M.d) ≈ d rtol = 1.0e-12
    @test !all(==(1.0), Array(M.d))

    results = MadNLP.solve!(
        MadNLPSolver(
            BorderQPGPU(CUDA.zeros(Float64, n)); preset = MadNLP.CondensedPreset, kkt_options,
            linear_solver = MadCoreGPU.CUDSSSolver, print_level = MadNLP.ERROR,
        )
    )
    @test results.status == MadNLP.SOLVE_SUCCEEDED
end
