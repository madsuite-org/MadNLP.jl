# A chained problem with callbacks written as broadcasts over the model's array type, so
# the same model runs on the CPU and on any GPU array type (like an ExaModels model):
#   min Σ (xᵢ - 0.3)²  s.t.  xᵢ² + xᵢ₊₁² = 1 for odd i, ≤ 1 for even i,  -2 ≤ x ≤ 2,
# optionally with x₁ fixed at 0.5. The inequality rows exercise the condensed KKT path
# that only inequalities reach; the fixed variable, the fixed-variable handler.
struct PrecompileChainModel{T, VT, VI} <: NLPModels.AbstractNLPModel{T, VT}
    meta::NLPModels.NLPModelMeta{T, VT}
    counters::NLPModels.Counters
    jrows::VI
    jcols::VI
end

_device(like, v::Vector) = like isa Array ? v : copyto!(similar(like, eltype(v), length(v)), v)

function PrecompileChainModel(n::Int, like::AbstractVector{T}; fixed::Bool = false) where {T}
    jrows = vec(repeat((1:n-1)', 2))          # constraint i touches x[i] and x[i+1]
    jcols = vec(vcat((1:n-1)', (2:n)'))
    meta = NLPModels.NLPModelMeta(
        n;
        ncon = n - 1,
        nnzj = 2 * (n - 1),
        nnzh = n,
        x0 = fill!(similar(like, n), T(0.5)),
        lvar = _device(like, T[(fixed && i == 1) ? T(0.5) : T(-2) for i in 1:n]),
        uvar = _device(like, T[(fixed && i == 1) ? T(0.5) : T(2) for i in 1:n]),
        y0 = fill!(similar(like, n - 1), zero(T)),
        lcon = _device(like, T[isodd(i) ? zero(T) : T(-Inf) for i in 1:n-1]),
        ucon = fill!(similar(like, n - 1), zero(T)),
    )
    return PrecompileChainModel(meta, NLPModels.Counters(), _device(like, jrows), _device(like, jcols))
end

NLPModels.obj(m::PrecompileChainModel, x::AbstractVector) = sum(abs2, x .- 0.3)
NLPModels.grad!(m::PrecompileChainModel, x::AbstractVector, g::AbstractVector) = (g .= 2 .* (x .- 0.3); g)
function NLPModels.cons!(m::PrecompileChainModel, x::AbstractVector, c::AbstractVector)
    n = length(x)
    c .= view(x, 1:n-1) .^ 2 .+ view(x, 2:n) .^ 2 .- 1
    return c
end
NLPModels.jac_structure!(m::PrecompileChainModel, I::AbstractVector, J::AbstractVector) =
    (copyto!(I, m.jrows); copyto!(J, m.jcols); (I, J))
function NLPModels.jac_coord!(m::PrecompileChainModel, x::AbstractVector, J::AbstractVector)
    n = length(x)
    view(J, 1:2:2n-2) .= 2 .* view(x, 1:n-1)
    view(J, 2:2:2n-2) .= 2 .* view(x, 2:n)
    return J
end
function NLPModels.jprod!(m::PrecompileChainModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
    n = length(x)
    Jv .= 2 .* view(x, 1:n-1) .* view(v, 1:n-1) .+ 2 .* view(x, 2:n) .* view(v, 2:n)
    return Jv
end
function NLPModels.jtprod!(m::PrecompileChainModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
    n = length(x)
    fill!(Jtv, zero(eltype(Jtv)))
    view(Jtv, 1:n-1) .+= 2 .* view(x, 1:n-1) .* v
    view(Jtv, 2:n) .+= 2 .* view(x, 2:n) .* v
    return Jtv
end
function NLPModels.hess_structure!(m::PrecompileChainModel, I::AbstractVector, J::AbstractVector)
    n = length(I)
    I .= 1:n; J .= 1:n
    return I, J
end
function NLPModels.hess_coord!(m::PrecompileChainModel, x::AbstractVector, y::AbstractVector,
                               H::AbstractVector; obj_weight = 1.0)
    n = length(x)
    H .= 2 * obj_weight
    view(H, 1:n-1) .+= 2 .* y
    view(H, 2:n) .+= 2 .* y
    return H
end

"""
    _precompile_solves(like; sizes = (100, 20000))

Solve `PrecompileChainModel`s stored in arrays like `like` (a `Vector`, `CuVector`, ...)
with and without a fixed variable at each of `sizes`, since large problems use different
reduction and scan kernels on a GPU, and run the restoration phases on a solved
instance, which these problems never enter on their own. Models are built here so that, when this runs inside `@compile_workload`,
the code compiled while building them is kept too.
"""
function _precompile_solves(like::AbstractVector; sizes = (100, 20000))
    for n in sizes, fixed in (false, true)
        madnlp(PrecompileChainModel(n, like; fixed); print_level = MadNLP.ERROR)
    end
    for n in sizes
        solver = MadNLPSolver(ModelBarrier(PrecompileChainModel(n, like));
                              print_level = MadNLP.ERROR, max_iter = 30)
        solve!(solver)
        restore!(solver)
        robust!(solver)
    end
    return nothing
end

@setup_workload begin
    # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the size of the
    # precompile file and potentially make loading faster.

    struct HS15Model{T} <: NLPModels.AbstractNLPModel{T,Vector{T}}
        meta::NLPModels.NLPModelMeta{T, Vector{T}}
        counters::NLPModels.Counters
    end

    function HS15Model(;T = Float64, x0=zeros(T,2), y0=zeros(T,2))
        return HS15Model(
            NLPModels.NLPModelMeta(
                2,     #nvar
                ncon = 2,
                nnzj = 4,
                nnzh = 3,
                x0 = x0,
                y0 = y0,
                lvar = T[-Inf, -Inf],
                uvar = T[0.5, Inf],
                lcon = T[1.0, 0.0],
                ucon = T[Inf, Inf],
                minimize = true
            ),
            NLPModels.Counters()
        )
    end

    function NLPModels.obj(nlp::HS15Model, x::AbstractVector)
        return 100.0 * (x[2] - x[1]^2)^2 + (1.0 - x[1])^2
    end

    function NLPModels.grad!(nlp::HS15Model, x::AbstractVector, g::AbstractVector)
        z = x[2] - x[1]^2
        g[1] = -400.0 * z * x[1] - 2.0 * (1.0 - x[1])
        g[2] = 200.0 * z
        return
    end

    function NLPModels.cons!(nlp::HS15Model, x::AbstractVector, c::AbstractVector)
        c[1] = x[1] * x[2]
        c[2] = x[1] + x[2]^2
    end

    function NLPModels.jac_structure!(nlp::HS15Model, I::AbstractVector{T}, J::AbstractVector{T}) where T
        copyto!(I, [1, 1, 2, 2])
        copyto!(J, [1, 2, 1, 2])
    end

    function NLPModels.jac_coord!(nlp::HS15Model, x::AbstractVector, J::AbstractVector)
        J[1] = x[2]    # (1, 1)
        J[2] = x[1]    # (1, 2)
        J[3] = 1.0     # (2, 1)
        J[4] = 2*x[2]  # (2, 2)
        return J
    end

    function NLPModels.jprod!(nlp::HS15Model, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
        jv[1] = x[2] * v[1] + x[1] * v[2]
        jv[2] = v[1] + 2 * x[2] * v[2]
        return jv
    end

    function NLPModels.jtprod!(nlp::HS15Model, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
        jv[1] = x[2] * v[1] + v[2]
        jv[2] = x[1] * v[1] + 2 * x[2] * v[2]
        return jv
    end

    function NLPModels.hess_structure!(nlp::HS15Model, I::AbstractVector{T}, J::AbstractVector{T}) where T
        copyto!(I, [1, 2, 2])
        copyto!(J, [1, 1, 2])
    end

    function NLPModels.hess_coord!(nlp::HS15Model, x, y, H::AbstractVector; obj_weight=1.0)
        # Objective
        H[1] = obj_weight * (-400.0 * x[2] + 1200.0 * x[1]^2 + 2.0)
        H[2] = obj_weight * (-400.0 * x[1])
        H[3] = obj_weight * 200.0
        # First constraint
        H[2] += y[1] * 1.0
        # Second constraint
        H[3] += y[2] * 2.0
        return H
    end

    nlp = HS15Model()
    __init__()
    
    @compile_workload begin
        madnlp(nlp; print_level=MadNLP.ERROR)
        _precompile_solves(Float64[])
    end
end
