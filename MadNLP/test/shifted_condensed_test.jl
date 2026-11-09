# ShiftedPreset on SparseCondensedKKTSystem: the equalities are enforced and kept uncondensed,
# with the inequality rows of `condense_max_nnz` (20 by default) or more nonzeros, as the border
# of the condensed matrix. The problem below has n/2 sparse equalities, n - 1 sparse
# inequalities and one dense inequality over every variable, active at the solution.

using NLPModels, LinearAlgebra, Random

struct EqBorderQP{T} <: NLPModels.AbstractNLPModel{T, Vector{T}}
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    counters::NLPModels.Counters
    n::Int
end

# rows 1:n/2           x_{2i-1} - x_{2i} = 0      (equalities)
# rows n/2+1:3n/2-1    x_i - x_{i+1} <= 1/2       (sparse inequalities)
# row 3n/2             sum(x) <= n/4              (dense inequality)
function EqBorderQP(n::Int; T = Float64)
    @assert iseven(n)
    me = n ÷ 2
    m = me + n
    meta = NLPModels.NLPModelMeta(
        n; ncon = m, nnzj = 2me + 2 * (n - 1) + n, nnzh = n,
        x0 = zeros(T, n), lvar = fill(T(-10), n), uvar = fill(T(10), n),
        lcon = [zeros(T, me); fill(T(-Inf), n - 1); T(-Inf)],
        ucon = [zeros(T, me); fill(T(1 // 2), n - 1); T(n) / 4],
        minimize = true, name = "EqBorderQP",
    )
    return EqBorderQP{T}(meta, NLPModels.Counters(), n)
end

NLPModels.obj(nlp::EqBorderQP, x::AbstractVector) = sum((x[i] - i / nlp.n)^2 for i in 1:nlp.n) / 2
function NLPModels.grad!(nlp::EqBorderQP, x::AbstractVector, g::AbstractVector)
    g .= x .- (1:nlp.n) ./ nlp.n
    return g
end
function NLPModels.cons!(nlp::EqBorderQP, x::AbstractVector, c::AbstractVector)
    n = nlp.n
    me = n ÷ 2
    for i in 1:me
        c[i] = x[2i - 1] - x[2i]
    end
    for i in 1:(n - 1)
        c[me + i] = x[i] - x[i + 1]
    end
    c[me + n] = sum(x)
    return c
end
function NLPModels.jac_structure!(nlp::EqBorderQP, I::AbstractVector, J::AbstractVector)
    n = nlp.n
    me = n ÷ 2
    k = 0
    for i in 1:me
        I[k + 1] = i; J[k + 1] = 2i - 1
        I[k + 2] = i; J[k + 2] = 2i
        k += 2
    end
    for i in 1:(n - 1)
        I[k + 1] = me + i; J[k + 1] = i
        I[k + 2] = me + i; J[k + 2] = i + 1
        k += 2
    end
    for j in 1:n
        I[k + j] = me + n; J[k + j] = j
    end
    return I, J
end
function NLPModels.jac_coord!(nlp::EqBorderQP, x::AbstractVector, V::AbstractVector)
    n = nlp.n
    T = eltype(V)
    k = 2 * (n ÷ 2) + 2 * (n - 1)
    for i in 1:(k ÷ 2)
        V[2i - 1] = one(T); V[2i] = -one(T)
    end
    V[(k + 1):end] .= one(T)
    return V
end
function NLPModels.hess_structure!(nlp::EqBorderQP, I::AbstractVector, J::AbstractVector)
    I .= 1:nlp.n; J .= 1:nlp.n
    return I, J
end
function NLPModels.hess_coord!(nlp::EqBorderQP, x::AbstractVector, y::AbstractVector, V::AbstractVector; obj_weight = 1.0)
    V .= obj_weight
    return V
end

@testset "ShiftedPreset on SparseCondensedKKTSystem" begin
    n = 40
    nlp = EqBorderQP(n)
    me = n ÷ 2
    ineq = (me + 1):(me + n)
    ref = MadNLP.madnlp(nlp; preset = MadNLP.ShiftedPreset, print_level = MadNLP.ERROR)
    @test ref.status == MadNLP.SOLVE_SUCCEEDED
    # the dense row is active at the solution
    @test sum(ref.solution) ≈ n / 4 atol = 1.0e-6

    @testset "options" begin
        opt = MadNLP.MadNLPOptions{Float64}(nlp; preset = MadNLP.ShiftedPreset, kkt_system = MadNLP.SparseCondensedKKTSystem)
        @test opt.equality_treatment == MadNLP.EnforceEquality
        @test opt.fixed_variable_treatment == MadNLP.MakeParameter
        @test opt.kkt_options[:condense_max_nnz] == 20
        @test opt.kkt_options[:border_delta] == 0.0
        @test opt.dual_shift == MadNLP.preset_perturbation(Float64)
        @test opt.kkt_scaler isa MadNLP.RuizScaler
        @test opt.slack_reset
        @test opt.linear_solver == MadNLP.CHOLMODSolver
    end

    @testset "condense_max_nnz = $N" for (N, kept_dense) in ((20, true), (n + 1, false))
        # the given KKT options are merged over the preset's (border_delta = 0 stays)
        kkt_options = Dict{Symbol, Any}(:condense_max_nnz => N)
        solver = MadNLP.MadNLPSolver(
            nlp; preset = MadNLP.ShiftedPreset, kkt_system = MadNLP.SparseCondensedKKTSystem,
            kkt_options, print_level = MadNLP.ERROR,
        )
        kkt = MadNLP.get_kkt(solver)
        @test kkt.ind_ineq == collect(ineq)
        @test kkt.ind_eq == collect(1:me)
        # the equalities are kept; the dense inequality is kept when it has N or more nonzeros
        border = kept_dense ? [1:me; me + n] : collect(1:me)
        @test kkt.border.ind_D == border
        @test size(kkt.aug_com, 1) == n + length(border)
        @test MadNLP.block_structure(kkt) == (n, length(border))
        @test kkt.border.delta_base == 0.0

        res = MadNLP.solve!(solver)
        @test res.status == MadNLP.SOLVE_SUCCEEDED
        @test res.solution ≈ ref.solution atol = 1.0e-6
        @test res.multipliers ≈ ref.multipliers atol = 1.0e-6
    end

    # without the dual shift, the scaling and the pivot bound, one solve with the condensed system
    # is as accurate as one with the augmented system at the same iterate
    @testset "KKT solve as accurate as the augmented one" begin
        res = map((MadNLP.SparseCondensedKKTSystem, MadNLP.SparseKKTSystem)) do kkt_system
            solver = MadNLP.MadNLPSolver(
                nlp; preset = MadNLP.ShiftedPreset, kkt_system, dual_shift = 0.0,
                kkt_scaler = MadNLP.NoScaler(), cholmod_dbound = 0.0, print_level = MadNLP.ERROR, max_iter = 8,
            )
            MadNLP.solve!(solver)
            kkt = MadNLP.get_kkt(solver)
            MadNLP.build_kkt!(kkt)
            MadNLP.factorize_kkt!(kkt)
            b = MadNLP.UnreducedKKTVector(kkt)
            x = MadNLP.UnreducedKKTVector(kkt)
            r = MadNLP.UnreducedKKTVector(kkt)
            MadNLP.full(b) .= sin.(1:length(MadNLP.full(b)))
            copyto!(MadNLP.full(x), MadNLP.full(b))
            MadNLP.solve_kkt!(kkt, x)
            MadNLP.mul!(r, kkt, x, 1.0, 0.0)
            (norm(MadNLP.full(r) .- MadNLP.full(b), Inf), copy(MadNLP.full(x)))
        end
        (res_c, x_c), (res_a, x_a) = res
        @test res_c <= 10 * res_a + 1.0e-12
        @test x_c ≈ x_a rtol = 1.0e-6
    end
    # a Cholesky factorization cannot factor the equality border, even with no dense row kept
    @testset "Cholesky with equalities" begin
        @test_throws ErrorException MadNLP.MadNLPSolver(
            nlp; kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = MadNLP.CHOLMODSolver,
            cholmod_algorithm = MadNLP.CHOLESKY, equality_treatment = MadNLP.EnforceEquality,
            print_level = MadNLP.ERROR,
        )
    end
end
