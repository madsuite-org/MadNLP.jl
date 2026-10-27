using CUDA
using NLPModels

# The augmented KKT system assembled on the device. The QP has variable bounds, so the
# COO form of the KKT matrix carries two entries per diagonal slot (the Hessian diagonal
# and the primal-dual diagonal), which the device transfer! must sum; its callbacks are
# broadcasts and run on the device.
struct BoundedQPGPU{T, VT} <: NLPModels.AbstractNLPModel{T, VT}
    meta::NLPModels.NLPModelMeta{T, VT}
    counters::NLPModels.Counters
    n::Int
    idx::VT
    jvals::VT
end

function BoundedQPGPU(x0::VT) where {T, VT <: AbstractVector{T}}
    n = length(x0)
    meta = NLPModels.NLPModelMeta(
        n; ncon = n, nnzj = 2n, nnzh = n,
        x0 = x0, lvar = fill!(similar(x0), T(-1)), uvar = fill!(similar(x0), T(n) / 2),
        lcon = fill!(similar(x0, n), T(-Inf)), ucon = fill!(similar(x0, n), one(T)),
        minimize = true, name = "BoundedQPGPU",
    )
    return BoundedQPGPU{T, VT}(meta, NLPModels.Counters(), n, VT(collect(T, 1:n)), VT(repeat([one(T), -one(T)], n)))
end

NLPModels.obj(nlp::BoundedQPGPU, x::AbstractVector) = sum((x .- nlp.idx) .^ 2) / 2
NLPModels.grad!(nlp::BoundedQPGPU, x::AbstractVector, g::AbstractVector) = (g .= x .- nlp.idx; g)
function NLPModels.cons!(nlp::BoundedQPGPU, x::AbstractVector, c::AbstractVector)
    n = nlp.n
    c[1:n-1] .= view(x, 1:n-1) .- view(x, 2:n)
    c[n:n] .= view(x, n:n) .- view(x, 1:1)
    return c
end
function NLPModels.jac_structure!(nlp::BoundedQPGPU, I::AbstractVector, J::AbstractVector)
    n = nlp.n
    rows = Int[]; cols = Int[]
    for i in 1:n
        push!(rows, i, i); push!(cols, i, mod1(i + 1, n))
    end
    copyto!(I, rows); copyto!(J, cols)
    return I, J
end
NLPModels.jac_coord!(nlp::BoundedQPGPU, x::AbstractVector, V::AbstractVector) = (copyto!(V, nlp.jvals); V)
function NLPModels.hess_structure!(nlp::BoundedQPGPU, I::AbstractVector, J::AbstractVector)
    copyto!(I, collect(1:nlp.n)); copyto!(J, collect(1:nlp.n))
    return I, J
end
NLPModels.hess_coord!(nlp::BoundedQPGPU, x::AbstractVector, y::AbstractVector, V::AbstractVector; obj_weight = 1.0) = (fill!(V, obj_weight); V)

@testset "MadNLP -- CUDSSSolver: SparseKKTSystem on the device" begin
    n = 40
    results_ref = MadNLP.madnlp(BoundedQPGPU(zeros(Float64, n)); kkt_system=MadNLP.SparseKKTSystem,
                                linear_solver=MadNLP.UmfpackSolver, print_level=MadNLP.ERROR, tol=1e-8)
    @test results_ref.status == MadNLP.SOLVE_SUCCEEDED
    # the solution sits on the upper bounds for the large indices, so the bound duals
    # and the duplicate diagonal are exercised
    @test any(results_ref.solution .≈ n / 2)

    solver = MadNLPSolver(BoundedQPGPU(CUDA.zeros(Float64, n)); print_level=MadNLP.ERROR,
                          kkt_system=MadNLP.SparseKKTSystem, linear_solver=MadNLPGPU.CUDSSSolver, tol=1e-8)
    results = MadNLP.solve!(solver)
    @test results.status == MadNLP.SOLVE_SUCCEEDED
    # cuDSS without pivoting and UMFPACK take different regularization paths, so the
    # iteration counts differ; the solutions agree to the tolerance
    @test results.objective ≈ results_ref.objective rtol=1e-7
    @test Array(results.solution) ≈ results_ref.solution rtol=1e-6
    @test Array(results.multipliers) ≈ results_ref.multipliers rtol=1e-6
end
