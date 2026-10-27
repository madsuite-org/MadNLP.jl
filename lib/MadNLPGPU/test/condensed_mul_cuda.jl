using CUDA
using NLPModels
using LinearAlgebra: norm

# mul!(w, kkt, x, alpha, beta) of the condensed KKT system on the device must be
# w = beta * w + alpha * K x for every beta, as on the host; the check is the
# linearity in beta on a QP whose callbacks are broadcasts and run on the device.
struct CyclicQPGPU{T, VT} <: NLPModels.AbstractNLPModel{T, VT}
    meta::NLPModels.NLPModelMeta{T, VT}
    counters::NLPModels.Counters
    n::Int
    idx::VT
    jvals::VT
end

function CyclicQPGPU(x0::VT) where {T, VT <: AbstractVector{T}}
    n = length(x0)
    meta = NLPModels.NLPModelMeta(
        n; ncon = n, nnzj = 2n, nnzh = n,
        x0 = x0, lvar = fill!(similar(x0), T(-10)), uvar = fill!(similar(x0), T(10)),
        lcon = fill!(similar(x0, n), T(-Inf)), ucon = fill!(similar(x0, n), one(T)),
        minimize = true, name = "CyclicQPGPU",
    )
    return CyclicQPGPU{T, VT}(meta, NLPModels.Counters(), n, VT(collect(T, 1:n)), VT(repeat([one(T), -one(T)], n)))
end

NLPModels.obj(nlp::CyclicQPGPU, x::AbstractVector) = sum((x .- nlp.idx) .^ 2) / 2
NLPModels.grad!(nlp::CyclicQPGPU, x::AbstractVector, g::AbstractVector) = (g .= x .- nlp.idx; g)
function NLPModels.cons!(nlp::CyclicQPGPU, x::AbstractVector, c::AbstractVector)
    n = nlp.n
    c[1:n-1] .= view(x, 1:n-1) .- view(x, 2:n)
    c[n:n] .= view(x, n:n) .- view(x, 1:1)
    return c
end
function NLPModels.jac_structure!(nlp::CyclicQPGPU, I::AbstractVector, J::AbstractVector)
    n = nlp.n
    rows = Int[]; cols = Int[]
    for i in 1:n
        push!(rows, i, i); push!(cols, i, mod1(i + 1, n))
    end
    copyto!(I, rows); copyto!(J, cols)
    return I, J
end
NLPModels.jac_coord!(nlp::CyclicQPGPU, x::AbstractVector, V::AbstractVector) = (copyto!(V, nlp.jvals); V)
function NLPModels.hess_structure!(nlp::CyclicQPGPU, I::AbstractVector, J::AbstractVector)
    copyto!(I, collect(1:nlp.n)); copyto!(J, collect(1:nlp.n))
    return I, J
end
NLPModels.hess_coord!(nlp::CyclicQPGPU, x::AbstractVector, y::AbstractVector, V::AbstractVector; obj_weight = 1.0) = (fill!(V, obj_weight); V)

@testset "MadNLP -- CUDSSSolver: condensed mul! is linear in beta" begin
    n = 30
    solver = MadNLPSolver(CyclicQPGPU(CUDA.zeros(Float64, n)); print_level=MadNLP.ERROR,
                          kkt_system=MadNLP.SparseCondensedKKTSystem, linear_solver=MadNLPGPU.CUDSSSolver, max_iter=3)
    MadNLP.solve!(solver)      # a few iterations so that the KKT matrix carries values
    kkt = MadNLP.get_kkt(solver)
    x = MadNLP.UnreducedKKTVector(kkt)
    w = MadNLP.UnreducedKKTVector(kkt)
    copyto!(MadNLP.full(x), CUDA.rand(Float64, length(MadNLP.full(x))))
    w0 = CUDA.rand(Float64, length(MadNLP.full(w)))

    # beta = 0: w = K x, whatever w held before
    copyto!(MadNLP.full(w), w0)
    MadNLP.mul!(w, kkt, x, 1.0, 0.0)
    kx = copy(MadNLP.full(w))
    copyto!(MadNLP.full(w), 2 .* w0 .+ 1)
    MadNLP.mul!(w, kkt, x, 1.0, 0.0)
    @test Array(MadNLP.full(w)) ≈ Array(kx) atol=1e-10
    # beta = 1: w = w0 + K x
    copyto!(MadNLP.full(w), w0)
    MadNLP.mul!(w, kkt, x, 1.0, 1.0)
    @test Array(MadNLP.full(w)) ≈ Array(w0 .+ kx) atol=1e-10
    # general alpha, beta
    copyto!(MadNLP.full(w), w0)
    MadNLP.mul!(w, kkt, x, -2.0, 0.5)
    @test Array(MadNLP.full(w)) ≈ Array(0.5 .* w0 .- 2.0 .* kx) atol=1e-10
end
