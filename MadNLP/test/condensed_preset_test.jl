# CondensedPreset: the equality constraints enforced and condensed through the equality shift,
# and the fallback to ShiftedPreset on a model whose condensed matrix would be dense or far larger
# than the augmented one. The problem below has n rows x_i - x_{i+1} <= 1 (equalities when `eq`)
# over `width` consecutive variables each (the others with zero coefficients), and one dense row,
# the coupling constraint over every variable.

using NLPModels

struct DenseRowQP{T} <: NLPModels.AbstractNLPModel{T, Vector{T}}
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    counters::NLPModels.Counters
    n::Int
    width::Int   # nonzeros of a row x_i - x_{i+1} <= 1: 2 (sparse) to n (every row dense)
    eq::Bool     # the n rows are equalities x_i - x_{i+1} = 0
end

function DenseRowQP(n::Int; T = Float64, width::Int = 2, eq::Bool = false)
    m = n + 1   # n rows x_i - x_{i+1} <= 1 (cyclic), one dense row sum(x) <= n/2
    nnzj = n * width + n
    nnzh = n
    meta = NLPModels.NLPModelMeta(
        n; ncon = m, nnzj = nnzj, nnzh = nnzh,
        x0 = zeros(T, n), lvar = fill(T(-10), n), uvar = fill(T(10), n),
        lcon = [fill(eq ? zero(T) : T(-Inf), n); T(-Inf)], ucon = [fill(eq ? zero(T) : one(T), n); T(n) / 2],
        minimize = true, name = "DenseRowQP",
    )
    return DenseRowQP{T}(meta, NLPModels.Counters(), n, width, eq)
end

NLPModels.obj(nlp::DenseRowQP, x::AbstractVector) = sum((x[i] - i)^2 for i in 1:nlp.n) / 2
function NLPModels.grad!(nlp::DenseRowQP, x::AbstractVector, g::AbstractVector)
    g .= x .- (1:nlp.n)
    return g
end
function NLPModels.cons!(nlp::DenseRowQP, x::AbstractVector, c::AbstractVector)
    n = nlp.n
    for i in 1:n
        c[i] = x[i] - x[mod1(i + 1, n)]
    end
    c[n + 1] = sum(x)
    return c
end
function NLPModels.jac_structure!(nlp::DenseRowQP, I::AbstractVector, J::AbstractVector)
    n, w = nlp.n, nlp.width
    k = 0
    for i in 1:n, d in 0:(w - 1)   # row i over x_i, x_{i+1}, ..., x_{i+w-1} (cyclic)
        k += 1; I[k] = i; J[k] = mod1(i + d, n)
    end
    for j in 1:n
        I[k + j] = n + 1; J[k + j] = j
    end
    return I, J
end
function NLPModels.jac_coord!(nlp::DenseRowQP, x::AbstractVector, V::AbstractVector)
    n, w = nlp.n, nlp.width
    o = one(eltype(V))
    fill!(V, zero(eltype(V)))
    for i in 1:n
        V[(i - 1) * w + 1] = o; V[(i - 1) * w + 2] = -o
    end
    V[(n * w + 1):(n * w + n)] .= o
    return V
end
function NLPModels.hess_structure!(nlp::DenseRowQP, I::AbstractVector, J::AbstractVector)
    I .= 1:nlp.n; J .= 1:nlp.n
    return I, J
end
function NLPModels.hess_coord!(nlp::DenseRowQP, x::AbstractVector, y::AbstractVector, V::AbstractVector; obj_weight = 1.0)
    V .= obj_weight
    return V
end

@testset "CondensedPreset: equalities condensed through the shift" begin
    n = 40
    nlp = DenseRowQP(n; eq = true)   # n equality rows and one dense inequality row
    ref = MadNLP.madnlp(nlp; print_level = MadNLP.ERROR, tol = 1.0e-8)
    @test ref.status == MadNLP.SOLVE_SUCCEEDED
    solver = MadNLP.MadNLPSolver(nlp; preset = MadNLP.CondensedPreset, dense_row_fallback = false, print_level = MadNLP.ERROR, tol = 1.0e-8)
    kkt = MadNLP.get_kkt(solver)
    @test kkt isa MadNLP.SparseCondensedKKTSystem && size(kkt.aug_com, 1) == n
    @test length(kkt.ind_eq) == n && length(kkt.ind_ineq) == 1
    @test kkt.eq_shift.base == 1.0e-8 && kkt.eq_shift.retry
    res = MadNLP.solve!(solver)
    @test res.status == MadNLP.SOLVE_SUCCEEDED
    @test MadNLP.MadCore.get_del_c(solver) == 0
    @test res.objective ≈ ref.objective rtol = 1.0e-6
    @test res.solution ≈ ref.solution atol = 1.0e-5
    # the cyclic equality rows are linearly dependent (they sum to zero), so their multipliers are
    # determined up to a constant; the multiplier of the dense inequality row is unique
    d = res.multipliers[1:n] .- ref.multipliers[1:n]
    @test maximum(d) - minimum(d) < 1.0e-5
    @test res.multipliers[n + 1] ≈ ref.multipliers[n + 1] atol = 1.0e-5
    # the retry off: the same solve with the option
    res0 = MadNLP.madnlp(nlp; preset = MadNLP.CondensedPreset, dense_row_fallback = false, equality_shift_retry = false, print_level = MadNLP.ERROR, tol = 1.0e-8)
    @test res0.status == MadNLP.SOLVE_SUCCEEDED
    @test res0.solution ≈ ref.solution atol = 1.0e-5
    # the equality shift as an option reaches the KKT system
    solver1 = MadNLP.MadNLPSolver(nlp; preset = MadNLP.CondensedPreset, dense_row_fallback = false, equality_shift = 1.0e-6, print_level = MadNLP.ERROR)
    @test MadNLP.get_kkt(solver1).eq_shift.base == 1.0e-6

    # a model with a dense Jacobian row (more than dense_row_abs nonzeros and dense_row_rel of the
    # variables) is solved with ShiftedPreset instead, unless dense_row_fallback is off
    big = DenseRowQP(6000)   # one row over every variable
    @test MadNLP.MadNLPCore.has_dense_rows(big, 512, 0.1, Inf) && !MadNLP.MadNLPCore.has_dense_rows(nlp, 512, 0.1, Inf)
    opts(m; kw...) = MadNLP.MadNLPCore.load_options(m; preset = MadNLP.CondensedPreset, print_level = MadNLP.ERROR, kw...).interior_point
    @test opts(big).preset === MadNLP.ShiftedPreset && opts(big).kkt_system === MadNLP.SparseKKTSystem
    @test opts(big; dense_row_fallback = false).preset === MadNLP.CondensedPreset
    @test opts(nlp).preset === MadNLP.CondensedPreset
    # the thresholds are options: the 40-nonzero row of the small model is dense below them
    @test opts(nlp; dense_row_abs = 8).preset === MadNLP.ShiftedPreset
    res2 = MadNLP.madnlp(nlp; preset = MadNLP.CondensedPreset, dense_row_abs = 8, print_level = MadNLP.ERROR, tol = 1.0e-8)
    @test res2.status == MadNLP.SOLVE_SUCCEEDED
    @test res2.solution ≈ ref.solution atol = 1.0e-5

    # so is a model whose rows are not dense but whose condensed matrix would be far larger than the
    # augmented one: 400 rows of 100 nonzeros (squared counts 4 million, against an augmented matrix
    # of 41,200 nonzeros, a ratio near 100) against dense_fill_rel = 20
    wide = DenseRowQP(400; width = 100)
    @test !MadNLP.MadNLPCore.has_dense_rows(wide, 512, 0.1, Inf) && MadNLP.MadNLPCore.has_dense_rows(wide, 512, 0.1, 20.0)
    @test opts(wide).preset === MadNLP.ShiftedPreset
    @test opts(wide; dense_fill_rel = 200.0).preset === MadNLP.CondensedPreset
    @test opts(wide; dense_row_fallback = false).preset === MadNLP.CondensedPreset
    refw = MadNLP.madnlp(wide; preset = MadNLP.AugmentedPreset, print_level = MadNLP.ERROR, tol = 1.0e-8)
    res3 = MadNLP.madnlp(wide; preset = MadNLP.CondensedPreset, print_level = MadNLP.ERROR, tol = 1.0e-8)
    @test refw.status == MadNLP.SOLVE_SUCCEEDED && res3.status == MadNLP.SOLVE_SUCCEEDED
    @test res3.objective ≈ refw.objective rtol = 1.0e-6   # the objective is near 1e7, so x agrees to about 1e-5
    @test res3.solution ≈ refw.solution atol = 1.0e-4
end
