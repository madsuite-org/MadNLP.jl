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
    dense::Bool   # every row dense (x_i - x_{i+1} + 0 * sum(x) <= 1) instead of sparse
end

function BorderQP(n::Int; T = Float64, dense::Bool = false)
    m = n + 1   # n rows x_i - x_{i+1} <= 1 (cyclic), one dense row sum(x) <= n/2
    nnzj = (dense ? n * n : 2n) + n
    nnzh = n
    meta = NLPModels.NLPModelMeta(
        n; ncon = m, nnzj = nnzj, nnzh = nnzh,
        x0 = zeros(T, n), lvar = fill(T(-10), n), uvar = fill(T(10), n),
        lcon = fill(T(-Inf), m), ucon = [ones(T, n); T(n) / 2],
        minimize = true, name = "BorderQP",
    )
    return BorderQP{T}(meta, NLPModels.Counters(), n, dense)
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
    c[n + 1] = sum(x)
    return c
end
function NLPModels.jac_structure!(nlp::BorderQP, I::AbstractVector, J::AbstractVector)
    n = nlp.n
    k = 0
    if nlp.dense
        for i in 1:n, j in 1:n
            k += 1; I[k] = i; J[k] = j
        end
    else
        for i in 1:n
            I[k + 1] = i; J[k + 1] = i
            I[k + 2] = i; J[k + 2] = mod1(i + 1, n)
            k += 2
        end
    end
    for j in 1:n
        I[k + j] = n + 1; J[k + j] = j
    end
    return I, J
end
function NLPModels.jac_coord!(nlp::BorderQP, x::AbstractVector, V::AbstractVector)
    n = nlp.n
    o = one(eltype(V))
    if nlp.dense
        fill!(V, zero(eltype(V)))
        for i in 1:n
            V[(i - 1) * n + i] = o; V[(i - 1) * n + mod1(i + 1, n)] = -o
        end
        V[(n * n + 1):(n * n + n)] .= o
    else
        for i in 1:n
            V[2i - 1] = o; V[2i] = -o
        end
        V[(2n + 1):3n] .= o
    end
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
    border = Dict{Symbol, Any}(:dense_row_abs => 8, :dense_row_rel => 0.1, :dense_row_fill_min => 0.0)
    ref = MadNLP.madnlp(
        nlp; kkt_system = MadNLP.SparseKKTSystem, linear_solver = MadNLP.UmfpackSolver,
        print_level = MadNLP.ERROR, tol = 1.0e-8
    )
    @test ref.status == MadNLP.SOLVE_SUCCEEDED

    for (name, ls, lsopts) in (
            ("LDLSolver", MadNLP.LDLSolver, (;)),
            ("CHOLMOD-LDL", MadNLP.CHOLMODSolver, (; cholmod_algorithm = MadNLP.LDL)),
        )
        @testset "$name" begin
            solver = MadNLP.MadNLPSolver(
                nlp; kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = ls,
                kkt_options = border, print_level = MadNLP.ERROR, tol = 1.0e-8, lsopts...
            )
            kkt = MadNLP.get_kkt(solver)
            @test length(kkt.border.ind_D) == 1 && kkt.border.ind_D[1] == n + 1
            @test MadNLP.block_structure(kkt) == (n, 1)
            @test size(kkt.aug_com, 1) == n + 1
            res = MadNLP.solve!(solver)
            @test res.status == MadNLP.SOLVE_SUCCEEDED
            @test res.solution ≈ ref.solution atol = 1.0e-6
            # the kept-row multiplier is the one of the coupling constraint
            @test res.multipliers[n + 1] ≈ ref.multipliers[n + 1] atol = 1.0e-6

            # with the defaults the row is condensed and the system is the plain condensed one
            solver0 = MadNLP.MadNLPSolver(
                nlp; kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = ls,
                print_level = MadNLP.ERROR, tol = 1.0e-8, lsopts...
            )
            kkt0 = MadNLP.get_kkt(solver0)
            @test isempty(kkt0.border.ind_D)
            @test kkt0.border.aug_c === kkt0.aug_com && kkt0.border.jt_s === kkt0.jt_csc
            @test size(kkt0.aug_com, 1) == n
            res0 = MadNLP.solve!(solver0)
            @test res0.status == MadNLP.SOLVE_SUCCEEDED
            @test res0.solution ≈ ref.solution atol = 1.0e-6
        end
    end

    # every row dense: the condensed block is one dense n x n triangle, which is smaller
    # than a border of all the rows, so nothing is kept
    nlp_dense = BorderQP(n; dense = true)
    solver = MadNLP.MadNLPSolver(
        nlp_dense; kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = MadNLP.LDLSolver,
        kkt_options = border, print_level = MadNLP.ERROR, tol = 1.0e-8
    )
    @test isempty(MadNLP.get_kkt(solver).border.ind_D)
    @test MadNLP.solve!(solver).status == MadNLP.SOLVE_SUCCEEDED

    # a Cholesky solver cannot factor the border: the rows are condensed with a warning
    @test_logs (:warn, r"Cholesky") match_mode = :any begin
        solver = MadNLP.MadNLPSolver(
            nlp; kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = MadNLP.CHOLMODSolver,
            kkt_options = border, print_level = MadNLP.ERROR
        )
        @test isempty(MadNLP.get_kkt(solver).border.ind_D)
    end
end
