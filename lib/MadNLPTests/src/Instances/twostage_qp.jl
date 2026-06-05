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
    schur_opts(; ns, nv, nd, nc, var_scen=nothing, con_scen=nothing) -> Dict

Convenience: build the `kkt_options` dict for `SchurComplementKKTSystem` with the
four scenario dimensions. When `var_scen`/`con_scen` are supplied (per-variable /
per-constraint scenario tags, 0 = design), they are passed through so the Schur
build partitions by tag instead of assuming the contiguous `[v_1..v_ns, d]` /
`[c_1..c_ns]` layout — required for models with non-contiguous orderings and/or
design-only constraints.
"""
function schur_opts(; ns, nv, nd, nc, var_scen = nothing, con_scen = nothing)
    opts = Dict{Symbol, Any}(
        :schur_ns => ns, :schur_nv => nv, :schur_nd => nd, :schur_nc => nc,
    )
    var_scen === nothing || (opts[:schur_var_scen] = var_scen)
    con_scen === nothing || (opts[:schur_con_scen] = con_scen)
    return opts
end

"""
    build_twostage_qp_general(x0_template = Vector{Float64}();
        ns, nv, nd, permute = true) -> (qp, var_scen, con_scen, kkt_opts)

Build a strictly-convex two-stage QP that exercises the *general* Schur path:
design-only **equality** and **inequality** constraints, plus a **non-contiguous**
global ordering (design variables are NOT last and each scenario's variables are
scattered). Strict convexity (positive-diagonal Hessian) gives a unique optimum, so
the Schur solve must match the reference `SparseKKTSystem` solve exactly.

Per scenario: `nv` variables, one equality and one inequality constraint coupling
that scenario's variables to all `nd` design variables. Globally: one design-only
equality (`Σ d = 1`) and one design-only inequality (`-5 ≤ d_1 - d_2 ≤ 5`,
inactive). With `permute = true` the canonical contiguous layout is reordered to
`[design vars, then component-major scenario vars]` / `[design cons, then scenario
cons]`; the returned `var_scen` / `con_scen` describe that order.

Returns the `TwoStageQP`, the tag vectors, and a ready `kkt_options` dict.
"""
function build_twostage_qp_general(
        x0_template::AbstractVector{T} = Vector{Float64}(undef, 0);
        ns::Int, nv::Int, nd::Int, permute::Bool = true,
    ) where {T}
    nd >= 2 || error("build_twostage_qp_general needs nd >= 2 for the design constraints")

    # Per-scenario: 1 equality + 1 inequality. Design: 1 equality + 1 inequality.
    nc = 2
    n = ns * nv + nd
    m = ns * nc + 2

    # --- Canonical (contiguous) assembly: vars [v_1..v_ns, d], cons [scen.., design..] ---
    cvar(k, j) = (k - 1) * nv + j          # canonical scenario var index
    cdes(l) = ns * nv + l               # canonical design var index
    cscon(k, t) = (k - 1) * nc + t         # canonical scenario constraint (t=1 eq, 2 ineq)
    cdcon(t) = ns * nc + t              # canonical design constraint (t=1 eq, 2 ineq)

    H = zeros(T, n)
    g = zeros(T, n)
    lvar = fill(T(-50), n)
    uvar = fill(T(50), n)
    A = zeros(T, m, n)
    lcon = zeros(T, m)
    ucon = zeros(T, m)

    @inbounds for k in 1:ns, j in 1:nv
        i = cvar(k, j)
        H[i] = 2 + T(0.1) * (j + k)
        g[i] = -2 * (T(j) + T(k))   # pulls v[k,j] toward a positive target
    end
    @inbounds for l in 1:nd
        i = cdes(l)
        H[i] = 2 + T(0.1) * l
        g[i] = T(0.5) * l
    end

    @inbounds for k in 1:ns
        # scenario equality: Σ_j v[k,j] + Σ_l d[l] = 0
        re = cscon(k, 1)
        for j in 1:nv
            A[re, cvar(k, j)] = 1
        end
        for l in 1:nd
            A[re, cdes(l)] = 1
        end
        lcon[re] = 0; ucon[re] = 0
        # scenario inequality (wide → inactive): -10 ≤ Σ_j v[k,j] - d[1] ≤ 10
        ri = cscon(k, 2)
        for j in 1:nv
            A[ri, cvar(k, j)] = 1
        end
        A[ri, cdes(1)] = -1
        lcon[ri] = -10; ucon[ri] = 10
    end
    # design equality: Σ_l d[l] = 1
    let re = cdcon(1)
        for l in 1:nd
            A[re, cdes(l)] = 1
        end
        lcon[re] = 1; ucon[re] = 1
    end
    # design inequality (wide → inactive): -5 ≤ d[1] - d[2] ≤ 5
    let ri = cdcon(2)
        A[ri, cdes(1)] = 1; A[ri, cdes(2)] = -1
        lcon[ri] = -5; ucon[ri] = 5
    end

    canon_var_scen = vcat((fill(k, nv) for k in 1:ns)..., fill(0, nd))
    canon_con_scen = vcat((fill(k, nc) for k in 1:ns)..., 0, 0)

    # --- Permutation: design first, then component-major scenario vars/cons ---
    if permute
        var_order = Int[]
        for l in 1:nd
            push!(var_order, cdes(l))
        end
        for j in 1:nv, k in 1:ns
            push!(var_order, cvar(k, j))
        end
        con_order = Int[]
        push!(con_order, cdcon(1)); push!(con_order, cdcon(2))
        for t in 1:nc, k in 1:ns
            push!(con_order, cscon(k, t))
        end
    else
        var_order = collect(1:n)
        con_order = collect(1:m)
    end

    H_p = H[var_order]; g_p = g[var_order]
    lvar_p = lvar[var_order]; uvar_p = uvar[var_order]
    A_p = A[con_order, var_order]
    lcon_p = lcon[con_order]; ucon_p = ucon[con_order]
    var_scen = canon_var_scen[var_order]
    con_scen = canon_con_scen[con_order]

    # --- COO structure (dense Jacobian nonzeros; diagonal Hessian) ---
    jrows = Int[]; jcols = Int[]; jvals = T[]
    @inbounds for i in 1:m, j in 1:n
        if A_p[i, j] != 0
            push!(jrows, i); push!(jcols, j); push!(jvals, A_p[i, j])
        end
    end
    hrows = collect(1:n); hcols = collect(1:n); hvals = copy(H_p)

    to_device(v_h::AbstractVector) = (v = similar(x0_template, length(v_h)); copyto!(v, v_h); v)
    to_device(M_h::AbstractMatrix) = (M = similar(x0_template, size(M_h)...); copyto!(M, M_h); M)

    meta = NLPModels.NLPModelMeta(
        n;
        ncon = m,
        nnzj = length(jrows),
        nnzh = n,
        x0 = to_device(zeros(T, n)),
        y0 = to_device(zeros(T, m)),
        lvar = to_device(lvar_p),
        uvar = to_device(uvar_p),
        lcon = to_device(lcon_p),
        ucon = to_device(ucon_p),
        minimize = true,
    )

    qp = TwoStageQP(
        meta, NLPModels.Counters(),
        to_device(H_p), to_device(A_p), to_device(g_p),
        to_device(hvals), to_device(jvals),
        hrows, hcols, jrows, jcols,
    )

    kkt_opts = schur_opts(; ns, nv, nd, nc, var_scen, con_scen)
    return qp, var_scen, con_scen, kkt_opts
end
