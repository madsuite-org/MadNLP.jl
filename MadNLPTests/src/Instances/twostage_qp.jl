"""
    TwoStageQP{T, VT, MT, VI} <: NLPModels.AbstractNLPModel{T, VT}

Diagonal-Hessian two-stage stochastic QP, used to exercise
`SchurComplementKKTSystem` (CPU and GPU) without depending on ExaModels.

Variable layout:   `x = [v_{1,1},...,v_{1,nv}, v_{2,1},...,v_{ns,nv}, d_1,...,d_nd]`
Constraint layout: `c = [c_{1,1},...,c_{1,nc}, c_{2,1},...,c_{ns,nc}]`

Construct via [`build_twostage_qp`](@ref). Storage device follows the
`x0_template` argument (CPU `Vector{T}` or e.g. `CuVector{T}` for GPU tests).
"""
struct TwoStageQP{T, VT <: AbstractVector{T}, MT <: AbstractMatrix{T}, VI <: AbstractVector{Int}} <: NLPModels.AbstractNLPModel{T, VT}
    meta::NLPModels.NLPModelMeta{T, VT}
    counters::NLPModels.Counters
    H_diag::VT
    A::MT
    g0::VT
    hvals::VT      # precomputed Hessian COO values (= H_diag; constant in x)
    jvals::VT      # precomputed Jacobian COO values (constant in x)
    hrows::VI
    hcols::VI
    jrows::VI
    jcols::VI
end

function NLPModels.obj(qp::TwoStageQP{T}, x::AbstractVector{T}) where {T}
    return T(0.5) * dot(x, qp.H_diag .* x) + dot(qp.g0, x)
end

function NLPModels.grad!(qp::TwoStageQP, x::AbstractVector, g::AbstractVector)
    g .= qp.H_diag .* x .+ qp.g0
    return g
end

NLPModels.cons!(qp::TwoStageQP, x::AbstractVector, c::AbstractVector) = mul!(c, qp.A, x)

function NLPModels.jac_structure!(qp::TwoStageQP, I::AbstractVector, J::AbstractVector)
    copyto!(I, qp.jrows)
    copyto!(J, qp.jcols)
    return
end

function NLPModels.jac_coord!(qp::TwoStageQP, x::AbstractVector, J::AbstractVector)
    copyto!(J, qp.jvals)
    return J
end

function NLPModels.hess_structure!(qp::TwoStageQP, I::AbstractVector, J::AbstractVector)
    copyto!(I, qp.hrows)
    copyto!(J, qp.hcols)
    return
end

function NLPModels.hess_coord!(qp::TwoStageQP{T}, x::AbstractVector, l::AbstractVector, hess::AbstractVector; obj_weight = one(T)) where {T}
    hess .= obj_weight .* qp.hvals
    return hess
end

"""
    build_twostage_qp(x0_template = Vector{Float64}(); ns, nv, nd, nc, hess_v, hess_d, g_v, g_d, A_v, A_d, lcon, ucon, lvar_v, uvar_v, lvar_d, uvar_d) -> TwoStageQP

Build a `TwoStageQP` instance. Pass `x0_template` to select the storage device
(default CPU `Vector{Float64}`; pass e.g. `CUDA.zeros(Float64, n)` for GPU).

`hess_v`, `g_v` are size `(nv, ns)`. `hess_d`, `g_d` are size `(nd,)`.
`A_v` is `(nc, nv, ns)`, `A_d` is `(nc, nd, ns)`.
`lcon`, `ucon` are `(nc, ns)`. `lvar_v`, `uvar_v` are `(nv, ns)`.
`lvar_d`, `uvar_d` are `(nd,)`.
"""
function build_twostage_qp(
        x0_template::AbstractVector{T} = Vector{Float64}(undef, 0);
        ns::Int, nv::Int, nd::Int, nc::Int,
        hess_v::Matrix{T}, hess_d::Vector{T},
        g_v::Matrix{T}, g_d::Vector{T},
        A_v::Array{T, 3}, A_d::Array{T, 3},
        lcon::Matrix{T}, ucon::Matrix{T},
        lvar_v::Matrix{T}, uvar_v::Matrix{T},
        lvar_d::Vector{T}, uvar_d::Vector{T},
    ) where {T}
    n = ns * nv + nd
    m = ns * nc
    off = ns * nv

    # Stage everything on host first.
    x0_h = zeros(T, n)
    lvar_h = Vector{T}(undef, n)
    uvar_h = Vector{T}(undef, n)
    g0_h = Vector{T}(undef, n)
    H_diag_h = Vector{T}(undef, n)

    @inbounds for k in 1:ns, j in 1:nv
        i = (k - 1) * nv + j
        lvar_h[i] = lvar_v[j, k]
        uvar_h[i] = uvar_v[j, k]
        g0_h[i] = g_v[j, k]
        H_diag_h[i] = hess_v[j, k]
    end
    @inbounds for j in 1:nd
        i = off + j
        lvar_h[i] = lvar_d[j]
        uvar_h[i] = uvar_d[j]
        g0_h[i] = g_d[j]
        H_diag_h[i] = hess_d[j]
    end

    glcon_h = Vector{T}(undef, m)
    gucon_h = Vector{T}(undef, m)
    @inbounds for k in 1:ns, i in 1:nc
        idx = (k - 1) * nc + i
        glcon_h[idx] = lcon[i, k]
        gucon_h[idx] = ucon[i, k]
    end

    A_h = zeros(T, m, n)
    @inbounds for k in 1:ns, i in 1:nc
        row = (k - 1) * nc + i
        for j in 1:nv
            A_h[row, (k - 1) * nv + j] = A_v[i, j, k]
        end
        for j in 1:nd
            A_h[row, off + j] = A_d[i, j, k]
        end
    end

    nnzj = ns * nc * (nv + nd)
    jrows_h = Vector{Int}(undef, nnzj)
    jcols_h = Vector{Int}(undef, nnzj)
    p = 1
    @inbounds for k in 1:ns, i in 1:nc
        row = (k - 1) * nc + i
        for j in 1:nv
            jrows_h[p] = row; jcols_h[p] = (k - 1) * nv + j; p += 1
        end
        for j in 1:nd
            jrows_h[p] = row; jcols_h[p] = off + j; p += 1
        end
    end

    nnzh = n
    hrows_h = collect(1:n)
    hcols_h = collect(1:n)

    jvals_h = T[A_h[jrows_h[k], jcols_h[k]] for k in 1:nnzj]
    hvals_h = copy(H_diag_h)  # diagonal Hessian → values match H_diag

    to_device(v_h::AbstractVector) = (v = similar(x0_template, length(v_h)); copyto!(v, v_h); v)
    to_device(M_h::AbstractMatrix) = (M = similar(x0_template, size(M_h)...); copyto!(M, M_h); M)

    x0 = to_device(x0_h)
    lvar = to_device(lvar_h)
    uvar = to_device(uvar_h)
    g0 = to_device(g0_h)
    H_diag = to_device(H_diag_h)
    A = to_device(A_h)
    hvals = to_device(hvals_h)
    jvals = to_device(jvals_h)
    glcon = to_device(glcon_h)
    gucon = to_device(gucon_h)
    y0 = to_device(zeros(T, m))

    meta = NLPModels.NLPModelMeta(
        n;
        ncon = m,
        nnzj = nnzj,
        nnzh = nnzh,
        x0 = x0,
        y0 = y0,
        lvar = lvar,
        uvar = uvar,
        lcon = glcon,
        ucon = gucon,
        minimize = true,
    )

    return TwoStageQP(
        meta, NLPModels.Counters(),
        H_diag, A, g0, hvals, jvals,
        hrows_h, hcols_h, jrows_h, jcols_h,
    )
end

"""
    schur_opts(; ns, nv, nd, nc) -> Dict

Convenience: build the `kkt_options` dict for `SchurComplementKKTSystem`
with the four scenario dimensions.
"""
schur_opts(; ns, nv, nd, nc) = Dict{Symbol, Any}(
    :schur_ns => ns, :schur_nv => nv, :schur_nd => nd, :schur_nc => nc,
)
