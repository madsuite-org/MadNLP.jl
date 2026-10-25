# SparseCondensedKKTSystem keeps dense Jacobian rows out of the condensation and carries
# them as a border of the condensed matrix. The problem below has one dense row (the
# coupling constraint over every variable) and n sparse rows; with the dense-row
# thresholds lowered the row is kept, with the defaults it is condensed, and the two
# solves must agree with each other and with the augmented system.

using NLPModels

struct BorderQP{T} <: NLPModels.AbstractNLPModel{T, Vector{T}}
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    counters::NLPModels.Counters
    n::Int
end

function BorderQP(n::Int; T = Float64)
    m = n + 1   # n sparse rows x_i - x_{i+1} <= 1 (cyclic), one dense row sum(x) <= n/2
    nnzj = 2n + n
    nnzh = n
    meta = NLPModels.NLPModelMeta(
        n; ncon = m, nnzj = nnzj, nnzh = nnzh,
        x0 = zeros(T, n), lvar = fill(T(-10), n), uvar = fill(T(10), n),
        lcon = fill(T(-Inf), m), ucon = [ones(T, n); T(n) / 2],
        minimize = true, name = "BorderQP",
    )
    return BorderQP{T}(meta, NLPModels.Counters(), n)
end

NLPModels.obj(nlp::BorderQP, x::AbstractVector) = sum((x[i] - i)^2 for i in 1:nlp.n) / 2
function NLPModels.grad!(nlp::BorderQP, x::AbstractVector, g::AbstractVector)
    g .= x .- (1:nlp.n)
    return g
end
function NLPModels.cons!(nlp::BorderQP, x::AbstractVector, c::AbstractVector)
    n = nlp.n
    for i in 1:n
        c[i] = x[i] - x[mod1(i + 1, n)]
    end
    c[n+1] = sum(x)
    return c
end
function NLPModels.jac_structure!(nlp::BorderQP, I::AbstractVector, J::AbstractVector)
    n = nlp.n
    k = 0
    for i in 1:n
        I[k+1] = i; J[k+1] = i
        I[k+2] = i; J[k+2] = mod1(i + 1, n)
        k += 2
    end
    for j in 1:n
        I[k+j] = n + 1; J[k+j] = j
    end
    return I, J
end
function NLPModels.jac_coord!(nlp::BorderQP, x::AbstractVector, V::AbstractVector)
    n = nlp.n
    for i in 1:n
        V[2i-1] = one(eltype(V)); V[2i] = -one(eltype(V))
    end
    V[2n+1:3n] .= one(eltype(V))
    return V
end
function NLPModels.hess_structure!(nlp::BorderQP, I::AbstractVector, J::AbstractVector)
    I .= 1:nlp.n; J .= 1:nlp.n
    return I, J
end
function NLPModels.hess_coord!(nlp::BorderQP, x::AbstractVector, y::AbstractVector, V::AbstractVector; obj_weight = 1.0)
    V .= obj_weight
    return V
end

@testset "SparseCondensedKKTSystem: dense rows kept as a border" begin
    n = 40
    nlp = BorderQP(n)
    # the dense row is kept: thresholds below its nonzero count and no fill floor
    border = Dict{Symbol,Any}(:dense_row_abs => 8, :dense_row_rel => 0.1, :dense_row_fill_min => 0.0)
    ref = MadNLP.madnlp(nlp; kkt_system = MadNLP.SparseKKTSystem, linear_solver = MadNLP.UmfpackSolver,
                        print_level = MadNLP.ERROR, tol = 1e-8)
    @test ref.status == MadNLP.SOLVE_SUCCEEDED

    for (name, ls, lsopts) in (
            ("LDLSolver", MadNLP.LDLSolver, (;)),
            ("CHOLMOD-LDL", MadNLP.CHOLMODSolver, (; cholmod_algorithm = MadNLP.LDL)),
        )
        @testset "$name" begin
            solver = MadNLP.MadNLPSolver(nlp; kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = ls,
                                         kkt_options = border, print_level = MadNLP.ERROR, tol = 1e-8, lsopts...)
            kkt = MadNLP.get_kkt(solver)
            @test length(kkt.ind_D) == 1 && kkt.ind_D[1] == n + 1
            @test MadNLP.block_structure(kkt) == (n, 1)
            @test size(kkt.aug_com, 1) == n + 1
            res = MadNLP.solve!(solver)
            @test res.status == MadNLP.SOLVE_SUCCEEDED
            @test res.solution ≈ ref.solution atol = 1e-6
            # the kept-row multiplier is the one of the coupling constraint
            @test res.multipliers[n+1] ≈ ref.multipliers[n+1] atol = 1e-6

            # with the defaults the row is condensed and the system is the plain condensed one
            solver0 = MadNLP.MadNLPSolver(nlp; kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = ls,
                                          print_level = MadNLP.ERROR, tol = 1e-8, lsopts...)
            kkt0 = MadNLP.get_kkt(solver0)
            @test isempty(kkt0.ind_D)
            @test kkt0.aug_c === kkt0.aug_com && kkt0.jt_s === kkt0.jt_csc
            @test size(kkt0.aug_com, 1) == n
            res0 = MadNLP.solve!(solver0)
            @test res0.status == MadNLP.SOLVE_SUCCEEDED
            @test res0.solution ≈ ref.solution atol = 1e-6
        end
    end

    # a Cholesky solver cannot factor the border: the rows are condensed with a warning
    @test_logs (:warn, r"Cholesky") match_mode = :any begin
        solver = MadNLP.MadNLPSolver(nlp; kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = MadNLP.CHOLMODSolver,
                                     kkt_options = border, print_level = MadNLP.ERROR)
        @test isempty(MadNLP.get_kkt(solver).ind_D)
    end
end
