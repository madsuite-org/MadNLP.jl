# Solve small problems on the GPU during precompilation, so the solver's GPU kernels
# (KKT assembly, cuDSS calls, vector broadcasts and reductions) are compiled and cached
# in this extension's package image. Needs a functional GPU at precompile time; the
# cached cubins target that GPU's architecture.

# A chained problem whose callbacks run on the GPU, like an ExaModels model:
#   min Σ (xᵢ - 0.3)²  s.t.  xᵢ² + xᵢ₊₁² = 1 for odd i, ≤ 1 for even i,  -2 ≤ x ≤ 2.
# The inequality rows exercise the condensed KKT path that only inequalities reach.
struct GPUChainModel{T, VT, VI} <: NLPModels.AbstractNLPModel{T, VT}
    meta::NLPModels.NLPModelMeta{T, VT}
    counters::NLPModels.Counters
    jrows::VI
    jcols::VI
end

function GPUChainModel(n::Int)
    jrows = vec(repeat((1:n-1)', 2))          # constraint i touches x[i] and x[i+1]
    jcols = vec(vcat((1:n-1)', (2:n)'))
    meta = NLPModels.NLPModelMeta(
        n;
        ncon = n - 1,
        nnzj = 2 * (n - 1),
        nnzh = n,
        x0 = CUDACore.fill(0.5, n),
        lvar = CUDACore.fill(-2.0, n),
        uvar = CUDACore.fill(2.0, n),
        y0 = CUDACore.zeros(Float64, n - 1),
        lcon = CuVector([isodd(i) ? 0.0 : -Inf for i in 1:n-1]),
        ucon = CUDACore.zeros(Float64, n - 1),
    )
    return GPUChainModel(meta, NLPModels.Counters(), CuVector(jrows), CuVector(jcols))
end

NLPModels.obj(m::GPUChainModel, x::AbstractVector) = sum(abs2, x .- 0.3)
NLPModels.grad!(m::GPUChainModel, x::AbstractVector, g::AbstractVector) = (g .= 2 .* (x .- 0.3); g)
function NLPModels.cons!(m::GPUChainModel, x::AbstractVector, c::AbstractVector)
    n = length(x)
    c .= view(x, 1:n-1) .^ 2 .+ view(x, 2:n) .^ 2 .- 1
    return c
end
NLPModels.jac_structure!(m::GPUChainModel, I::AbstractVector, J::AbstractVector) =
    (copyto!(I, m.jrows); copyto!(J, m.jcols); (I, J))
function NLPModels.jac_coord!(m::GPUChainModel, x::AbstractVector, J::AbstractVector)
    n = length(x)
    view(J, 1:2:2n-2) .= 2 .* view(x, 1:n-1)
    view(J, 2:2:2n-2) .= 2 .* view(x, 2:n)
    return J
end
function NLPModels.hess_structure!(m::GPUChainModel, I::AbstractVector, J::AbstractVector)
    n = length(I)
    I .= 1:n; J .= 1:n
    return I, J
end
function NLPModels.hess_coord!(m::GPUChainModel, x::AbstractVector, y::AbstractVector,
                               H::AbstractVector; obj_weight = 1.0)
    n = length(x)
    H .= 2 * obj_weight
    view(H, 1:n-1) .+= 2 .* y
    view(H, 2:n) .+= 2 .* y
    return H
end

PrecompileTools.@setup_workload begin
    if CUDACore.functional()
        __init__()
        PrecompileTools.@compile_workload begin
            # Build models in here, not in the setup block: constructing them compiles
            # GPU kernels (the bounds analysis runs `findall`), and only code compiled
            # inside `@compile_workload` is kept in the package image.
            wrapped = MadNLP.SparseWrapperModel(CuVector, MadNLP.HS15Model())
            MadNLP.madnlp(wrapped; print_level = MadNLP.ERROR)
            for n in (100, 20000)
                MadNLP.madnlp(GPUChainModel(n); print_level = MadNLP.ERROR)
            end
            # The restoration phases, which these problems never enter on their own: run
            # them on a solved instance (the same solver type `madnlp` builds), at both
            # sizes, since large problems use multi-block reduction kernels.
            for n in (100, 20000)
                solver = MadNLP.MadNLPSolver(MadNLP.ModelBarrier(GPUChainModel(n));
                                             print_level = MadNLP.ERROR, max_iter = 30)
                MadNLP.solve!(solver)
                MadNLP.restore!(solver)
                MadNLP.robust!(solver)
            end
        end
    end
end
