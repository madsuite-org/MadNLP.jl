"""
    BorderedRows{T, VT, MT, VI}

The dense Jacobian rows that a [`SparseCondensedKKTSystem`](@ref) keeps out of the
condensation and carries, with their multipliers, as a border of the condensed matrix
(see [`create_kkt_system`](@ref)). With no such row the structure is empty: `jt_s` is
`jt_csc`, `aug_c` is `aug_com`, `dbuf_s` is `diag_buffer` and the maps have length zero.
"""
struct BorderedRows{T, VT, MT, VI}
    ind_S::VI          # rows condensed
    ind_D::VI          # rows kept as the border
    jt_s::MT           # the ind_S columns of jt_csc
    smap::VI           # nonzeros(jt_csc) -> nonzeros(jt_s)
    aug_c::MT          # condensed block of the ind_S rows
    dbuf_s::VT         # diag_buffer[ind_S]
    cmap::VI           # nonzeros(aug_c) -> nonzeros(aug_com)
    dmap_pos::VI       # positions of the J_D entries in nonzeros(aug_com)
    dmap_src::VI       # their sources in nonzeros(jt_csc)
    epos::VI           # positions of the border diagonal in nonzeros(aug_com)
    pd_buffer::VT      # solve buffer of length n + m_D
    delta::Vector{T}   # current perturbation of the border diagonal (host scalar)
    delta_base::T
    delta_exp::T
end

Base.isempty(b::BorderedRows) = isempty(b.ind_D)
num_border_rows(b::BorderedRows) = length(b.ind_D)

"""
    AugmentedRuizState{T, VT, VI}

The pattern and buffers of [`AugmentedRuizScaler`](@ref) on a [`SparseCondensedKKTSystem`](@ref):
the augmented matrix `[H + Σx  Jᵀ; J  a]` with both triangles, stored by columns, each nonzero
mapped to its source in the KKT system, and the scaling `d` of its `n + m` rows. Built once from
the sparsity patterns; `augmented_ruiz_scaling!` gathers the values and runs the sweeps on the
device of the KKT system at every factorization.
"""
mutable struct AugmentedRuizState{T, VT <: AbstractVector{T}, VI <: AbstractVector}
    sweeps::Int
    colptr::VI
    rowval::VI
    long_cols::VI      # columns with more than SCALING_LONG_COLUMN nonzeros (GPU sweeps)
    absnz::VT          # |entry| on the pattern
    d::VT
    dnew::VT
    a::VT              # the dual diagonal: Σd, less 1 / Σs on the inequality rows
    h_pos::VI          # off-diagonal entries of H, both triangles, and their nonzero of hess_com
    h_src::VI
    hd_pos::VI         # diagonal entries of H: position, nonzero of hess_com, variable
    hd_src::VI
    hd_var::VI
    pd_pos::VI         # diagonal entries without an entry of H: position, variable
    pd_var::VI
    j_pos::VI          # entries of J, both triangles, and their nonzero of jt_csc
    j_src::VI
    a_pos::VI          # the dual diagonal: position, row
    a_row::VI
    bpos::VI           # n + ind_D, the rows of d that scale the border
end

function AugmentedRuizState(::Type{T}, hess_com, jt_csc, aug_com, ind_D::Vector{Int}, sweeps::Int) where {T}
    hcolptr, hrow = scaling_host_pattern(hess_com)   # lower triangle
    jcolptr, jrow = scaling_host_pattern(jt_csc)     # column c = constraint c, row i = variable i
    n = length(hcolptr) - 1
    m = length(jcolptr) - 1
    rows = Int[]; cols = Int[]; kind = Int[]; src = Int[]
    entry(i, j, k, s) = (push!(rows, i); push!(cols, j); push!(kind, k); push!(src, s))
    hasdiag = falses(n)
    for j in 1:n, k in hcolptr[j]:(hcolptr[j + 1] - 1)
        i = hrow[k]
        if i == j
            hasdiag[j] = true
            entry(i, i, 2, k)
        else
            entry(i, j, 1, k); entry(j, i, 1, k)
        end
    end
    for i in 1:n
        hasdiag[i] || entry(i, i, 3, i)
    end
    for c in 1:m, k in jcolptr[c]:(jcolptr[c + 1] - 1)
        i = jrow[k]
        entry(n + c, i, 4, k); entry(i, n + c, 4, k)
    end
    for c in 1:m
        entry(n + c, n + c, 5, c)
    end
    S = sparse(rows, cols, 1:length(rows), n + m, n + m)
    nnz(S) == length(rows) || error("AugmentedRuizScaler: the Hessian or Jacobian pattern has a repeated entry")
    q = nonzeros(S)   # position in the pattern -> entry
    pos(kd) = findall(e -> kind[e] == kd, q)
    p1, p2, p3, p4, p5 = pos(1), pos(2), pos(3), pos(4), pos(5)
    Ti = scaling_index_type(aug_com)
    dv(v) = scaling_device(aug_com, v)
    di(v) = dv(Ti.(v))
    long = findall(>(SCALING_LONG_COLUMN), diff(S.colptr))
    N = n + m
    return AugmentedRuizState(
        sweeps, di(S.colptr), di(S.rowval), di(long),
        dv(zeros(T, length(q))), dv(ones(T, N)), dv(ones(T, N)), dv(zeros(T, m)),
        di(p1), di(src[q[p1]]), di(p2), di(src[q[p2]]), di(rows[q[p2]]), di(p3), di(rows[q[p3]]),
        di(p4), di(src[q[p4]]), di(p5), di(rows[q[p5]] .- n), di(n .+ ind_D),
    )
end


"""
    SparseCondensedKKTSystem{T, VT, MT, QN} <: AbstractCondensedKKTSystem{T, VT, MT, QN}

Implement the [`AbstractCondensedKKTSystem`](@ref) in sparse COO format.

"""
struct SparseCondensedKKTSystem{T, VT, MT, QN, LS, VI, VI32, VTu1, VTu2, EXT, AS} <: AbstractCondensedKKTSystem{T, VT, MT, QN}
    # Hessian
    hess::VT
    hess_raw::SparseMatrixCOO{T, Int32, VT, VI32}
    hess_com::MT
    hess_csc_map::Union{Nothing, VI}

    # Jacobian
    jac::VT
    jt_coo::SparseMatrixCOO{T, Int32, VT, VI32}
    jt_csc::MT
    jt_csc_map::Union{Nothing, VI}

    quasi_newton::QN
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT

    # buffer
    buffer::VT
    buffer2::VT
    buffer3::VT

    # Augmented system: the condensed block of the sparse rows, bordered by the dense rows
    aug_com::MT

    # slack diagonal buffer
    diag_buffer::VT
    dptr::VTu1
    hptr::VTu1
    jptr::VTu2

    # dense Jacobian rows kept out of the condensation as a border of aug_com
    border::BorderedRows{T, VT, MT, VI}

    # LinearSolver
    linear_solver::LS

    # Info
    ind_ineq::VI
    ind_eq::VI
    # equality rows condensed with D = 1 / (δ_i - Σd), δ_i = eq_shift[1] (times ‖J_i‖² when
    # eq_scaled); eq_shift[1] == 0 keeps them in the border. eq_shift[2] is 1 once the refinement
    # feedback has moved δ since the last build (eq_adaptive), so δ moves at most once per factorization.
    eq_shift::Vector{T}
    eq_adaptive::Bool
    eq_scaled::Bool
    # δ = clamp(1 / max D_I, eq_shift[1], max(1e-6, eq_shift[1])) from the inequality weights (eq_shift[1]
    # is the lower bound, raised by retry_with_larger_shift!)
    eq_from_weights::Bool
    eq_rownorm::VT     # ‖J_i‖² of every row (eq_scaled), else empty
    eq_ones::VT        # ones(n) (eq_scaled), else empty
    jt_sq::MT          # the pattern of jt_csc with squared values (eq_scaled), else jt_csc
    ind_lb::VI
    ind_ub::VI

    # dual shift of the eliminated slack rows: the factorized matrix condenses each row with
    # D = Σs / (1 - (Σd - slack_shift) Σs), which keeps D below 1 / slack_shift; the KKT operator
    # (mul!) does not see it, so iterative refinement solves the unshifted system
    slack_shift::T

    # the pattern and buffers of AugmentedRuizScaler (nothing with another scaler)
    aug_scaling::AS

    # extra
    ext::EXT
end

"""
    create_kkt_system(::Type{SparseCondensedKKTSystem}, cb, linear_solver; kwargs...)

Condensing a dense Jacobian row turns the condensed matrix dense (the row's outer
product). Rows with more than `dense_row_abs` nonzeros and more than `dense_row_rel`
times the number of variables are therefore kept out of the condensation, with their
multipliers, as a border: the matrix handed to the linear solver is

    [ H + Σx + J_Sᵀ D_S J_S   J_Dᵀ            ]
    [ J_D                     Σd_D - Σs_D⁻¹ - δ ]

of size n + m_D, whose inertia (n, m_D, 0) is known, so a static LDLᵀ factorization
applies; `δ = border_delta * μ^border_delta_exp` keeps the border diagonal away from
zero. An equality row has no slack to eliminate and is always kept, with the border diagonal
`Σd_D`.

By default, among the candidate inequality rows the densest are kept, as many as minimize
the estimated nonzeros of the factored matrix (a condensed block is at most a dense n × n
triangle, a kept row costs 2 nnz + 1 border entries), and only when that saves more than
`dense_row_fill_min` nonzeros and the linear solver can factor an indefinite matrix (a
solver set to a Cholesky factorization condenses every row). With no row kept the system
is the plain condensed one.

With `condense_max_nnz = N` (an integer) the rule is instead a fixed threshold: an
inequality row is condensed when it has fewer than `N` nonzeros and kept otherwise.
"""
function create_kkt_system(
    ::Type{SparseCondensedKKTSystem},
    cb::SparseCallback{T,VT},
    linear_solver::Type;
    opt_linear_solver=default_options(linear_solver),
    hessian_approximation=ExactHessian,
    qn_options=QuasiNewtonOptions(),
    kkt_scaler::AbstractScaler=NoScaler(),
    primal_shift=0.0,
    dual_shift=0.0,
    dense_row_abs::Int = 512,
    dense_row_rel::Float64 = 0.10,
    dense_row_fill_min::Float64 = 1e7,
    border_delta::Float64 = 1e-8,
    border_delta_exp::Float64 = 0.25,
        condense_max_nnz::Union{Nothing, Int} = nothing,
    condense_equalities::Bool = false,
    equality_shift::Float64 = 1e-10,
    equality_shift_adaptive::Bool = false,
    equality_shift_scaled::Bool = false,
    equality_shift_from_weights::Bool = false,
    equality_shift_max::Float64 = 1e-4,
    primal_shift_ratio::Float64 = 0.0,
    equality_shift_weight_factor::Float64 = 1.0,
    equality_shift_pivot_target::Float64 = 0.0,
) where {T, VT}
    ind_ineq = cb.ind_ineq
    n = cb.nvar
    m = cb.ncon
    n_slack = length(ind_ineq)
    ind_ineq_h = Array(ind_ineq)
    ind_eq_h = setdiff(1:m, ind_ineq_h)

    # Evaluate sparsity pattern
    jac_sparsity_I = create_array(cb, Int32, cb.nnzj)
    jac_sparsity_J = create_array(cb, Int32, cb.nnzj)
    _jac_sparsity_wrapper!(cb,jac_sparsity_I, jac_sparsity_J)

    quasi_newton = create_quasi_newton(hessian_approximation, cb, n; options=qn_options)
    hess_sparsity_I, hess_sparsity_J = build_hessian_structure(cb, hessian_approximation)

    force_lower_triangular!(hess_sparsity_I,hess_sparsity_J)

    n_jac = length(jac_sparsity_I)
    n_hess = length(hess_sparsity_I)
    n_tot = n + n_slack
    nlb = length(cb.ind_lb)
    nub = length(cb.ind_ub)

    ind_D = if condense_max_nnz === nothing
        _select_dense_rows(jac_sparsity_I, n, m, n_hess, dense_row_abs, dense_row_rel, dense_row_fill_min, ind_ineq_h)
    else
        _select_rows_by_nnz(jac_sparsity_I, m, condense_max_nnz, condense_equalities ? collect(1:m) : ind_ineq_h)
    end
    # the equality rows kept in the border: all of them, or with condense_equalities only the
    # dense ones selected above
    eq_kept = condense_equalities ? intersect(ind_D, ind_eq_h) : ind_eq_h
    if !_factors_indefinite(opt_linear_solver)
        isempty(eq_kept) || error("$(linear_solver) is set to a Cholesky factorization and cannot factor the bordered condensed matrix, which an equality-constrained problem requires.")
        if !isempty(ind_D)
            Base.@warn "$(linear_solver) is set to a Cholesky factorization and cannot factor the bordered condensed matrix; the $(length(ind_D)) dense Jacobian rows are condensed."
            ind_D = Int[]
        end
    end
    ind_D = sort!(union(ind_D, eq_kept))
    mD = length(ind_D)

    reg = VT(undef, n_tot)
    pr_diag = VT(undef, n_tot)
    du_diag = VT(undef, m)
    l_diag = VT(undef, nlb)
    u_diag = VT(undef, nub)
    l_lower = VT(undef, nlb)
    u_lower = VT(undef, nub)
    buffer = VT(undef, m)
    buffer2 = VT(undef, m)
    buffer3 = VT(undef, m)
    hess = VT(undef, n_hess)
    jac = VT(undef, n_jac)
    diag_buffer = VT(undef, m)
    fill!(jac, zero(T))

    hess_raw = SparseMatrixCOO(n, n, hess_sparsity_I, hess_sparsity_J, hess)
    jt_coo = SparseMatrixCOO(
        n, m,
        jac_sparsity_J,
        jac_sparsity_I,
        jac,
    )

    jt_csc, jt_csc_map = coo_to_csc(jt_coo)
    hess_com, hess_csc_map = coo_to_csc(hess_raw)

    aug_com, dptr, hptr, jptr, border = _build_bordered_condensed_aug_symbolic(
        hess_com, jt_csc, ind_D, VT, ind_ineq, diag_buffer, T(border_delta), T(border_delta_exp)
    )

    # tell the linear solver the saddle-point partition before it analyzes
    _linear_solver = LinearSolverBarrier(create_kkt_linear_solver(linear_solver, aug_com, n; opt = opt_linear_solver, kkt_scaler, primal_shift = max(primal_shift, primal_shift_ratio * equality_shift), dual_shift))
    aug_scaling = kkt_scaler isa AugmentedRuizScaler ? AugmentedRuizState(T, hess_com, jt_csc, aug_com, ind_D, kkt_scaler.sweeps) : nothing
    ext = get_sparse_condensed_ext(VT, hess_com, jptr, jt_csc_map, hess_csc_map)
    return SparseCondensedKKTSystem(
        hess, hess_raw, hess_com, hess_csc_map,
        jac, jt_coo, jt_csc, jt_csc_map,
        quasi_newton,
        reg, pr_diag, du_diag,
        l_diag, u_diag, l_lower, u_lower,
        buffer, buffer2, buffer3,
        aug_com, diag_buffer, dptr, hptr, jptr,
        border,
        _linear_solver,
        ind_ineq, _to_index_vector(ind_ineq, ind_eq_h), [condense_equalities ? T(equality_shift) : zero(T), zero(T), T(equality_shift), T(equality_shift_max), T(primal_shift_ratio), T(equality_shift_weight_factor), T(equality_shift), T(equality_shift_pivot_target), zero(T), zero(T), zero(T)],
        condense_equalities && (equality_shift_adaptive || equality_shift_from_weights), condense_equalities && equality_shift_scaled,
        condense_equalities && equality_shift_from_weights,
        (condense_equalities && equality_shift_scaled ? (fill!(VT(undef, m), one(T)), fill!(VT(undef, n), one(T)), copy(jt_csc)) : (VT(undef, 0), VT(undef, 0), jt_csc))...,
        cb.ind_lb, cb.ind_ub,
        T(dual_shift),
        aug_scaling,
        ext
    )
end

_to_index_vector(like, v) = (o = similar(like, Int, length(v)); copyto!(o, v); o)

# The inequality rows with condense_max_nnz or more nonzeros, sorted: every inequality row with
# fewer is condensed.
function _select_rows_by_nnz(jac_sparsity_I, m, condense_max_nnz, ind_ineq)
    cnt = zeros(Int, m)
    for r in Array(jac_sparsity_I)
        cnt[r] += 1
    end
    return sort!(filter(i -> cnt[i] >= condense_max_nnz, ind_ineq))
end

# The inequality rows (ind_ineq) with more than dense_row_abs nonzeros and more than
# dense_row_rel * n are the candidates; the densest k of them are kept as the border, k chosen to minimize the
# estimated nonzeros of the factored matrix: the condensed block is bounded by a dense
# n x n triangle, so once the rows left condensed fill it, keeping more rows only adds
# border entries (2 nnz + 1 each). The border is used when the saving exceeds
# dense_row_fill_min. Returns the kept rows, sorted.
function _select_dense_rows(jac_sparsity_I, n, m, n_hess, dense_row_abs, dense_row_rel, dense_row_fill_min, ind_ineq = 1:m)
    cnt = zeros(Int, m)
    for r in Array(jac_sparsity_I)
        cnt[r] += 1
    end
    tri(c) = c * (c + 1) / 2
    cand = sort!(filter(i -> cnt[i] > dense_row_abs && cnt[i] > dense_row_rel * n, collect(ind_ineq)); by = i -> -cnt[i])
    dense_cap = tri(n)
    base = n_hess + sum(i -> tri(cnt[i]), ind_ineq; init = 0.0)   # every inequality row condensed, before the cap
    best_k, best_total, saved, border = 0, min(dense_cap, base), 0.0, 0.0
    for (k, i) in enumerate(cand)
        saved += tri(cnt[i])
        border += 2 * cnt[i] + 1
        total = min(dense_cap, base - saved) + border
        if total < best_total
            best_k, best_total = k, total
        end
    end
    if min(dense_cap, base) - best_total < dense_row_fill_min
        best_k = 0
    end
    return sort!(cand[1:best_k])
end

# The symbolic structure of the condensed matrix with the rows ind_D kept as a border:
# the condensed block of the other rows (aug_c, with the maps dptr/hptr/jptr of the
# condensed assembly) embedded in the bordered pattern [C 0; J_D diag] (aug_com), and the
# maps that scatter values into it. With ind_D empty the condensed block is the whole
# matrix and the BorderedRows aliases it. The symbolic work runs on host copies of the
# patterns; the maps are returned on the array backend of ind_ineq.
function _build_bordered_condensed_aug_symbolic(hess_com, jt_csc, ind_D, ::Type{VT}, ind_ineq, diag_buffer, delta, delta_exp) where {VT}
    T = eltype(VT)
    n, m = size(jt_csc)
    mD = length(ind_D)
    tovi(v) = (o = similar(ind_ineq, Int, length(v)); copyto!(o, v); o)
    ind_S = setdiff(1:m, ind_D)
    if mD == 0
        aug_com, dptr, hptr, jptr = build_condensed_aug_symbolic(hess_com, jt_csc)
        e = tovi(Int[])
        border = BorderedRows(tovi(ind_S), tovi(ind_D), jt_csc, e, aug_com, diag_buffer, e, e, e, e, VT(undef, n), [delta], delta, delta_exp)
        return aug_com, dptr, hptr, jptr, border
    end
    mS = length(ind_S)
    # the ind_S columns of jt (pattern fixed; values refreshed through smap)
    jcolptr = Array(SparseArrays.getcolptr(jt_csc))
    jrowval = Array(SparseArrays.rowvals(jt_csc))
    smap = Int[]
    scolptr = Vector{Int32}(undef, mS + 1)
    scolptr[1] = 1
    for (jj, j) in enumerate(ind_S)
        rng = jcolptr[j]:(jcolptr[j + 1] - 1)
        append!(smap, rng)
        scolptr[jj + 1] = scolptr[jj] + Int32(length(rng))
    end
    jt_s = typeof(jt_csc)(SparseMatrixCSC{T, Int32}(n, mS, scolptr, jrowval[smap], zeros(T, length(smap))))
    dbuf_s = VT(undef, mS)

    # condensed block of the sparse rows
    aug_c, dptr, hptr, jptr = build_condensed_aug_symbolic(hess_com, jt_s)

    # bordered pattern [C 0; J_D diag] (lower triangle) from disjoint COO parts
    Nb = n + mD
    acolptr = Array(SparseArrays.getcolptr(aug_c))
    arowval = Array(SparseArrays.rowvals(aug_c))
    cI = Vector{Int}(undef, length(arowval))
    cJ = Vector{Int}(undef, length(arowval))
    for j in 1:n, k in acolptr[j]:(acolptr[j + 1] - 1)
        cI[k] = arowval[k]
        cJ[k] = j
    end
    dI = Int[]; dJ = Int[]; dsrc = Int[]
    for (dd, d) in enumerate(ind_D)
        for k in jcolptr[d]:(jcolptr[d + 1] - 1)
            push!(dI, n + dd); push!(dJ, Int(jrowval[k])); push!(dsrc, k)
        end
    end
    eI = collect((n + 1):Nb)
    L = length(cI) + length(dI) + length(eI)
    Sidx = sparse([cI; dI; eI], [cJ; dJ; eI], Float64.(1:L), Nb, Nb)
    @assert nnz(Sidx) == L
    pos = Vector{Int}(undef, L)
    for k in 1:L
        pos[Int(Sidx.nzval[k])] = k
    end
    cmap = pos[1:length(cI)]
    dmap_pos = pos[(length(cI) + 1):(length(cI) + length(dI))]
    epos = pos[(length(cI) + length(dI) + 1):end]
    aug_com = typeof(jt_csc)(SparseMatrixCSC{T, Int32}(Nb, Nb, Int32.(Sidx.colptr), Int32.(Sidx.rowval), zeros(T, L)))
    border = BorderedRows(
        tovi(ind_S), tovi(ind_D), jt_s, tovi(smap), aug_c, dbuf_s,
        tovi(cmap), tovi(dmap_pos), tovi(dsrc), tovi(epos), VT(undef, Nb), [delta], delta, delta_exp
    )
    return aug_com, dptr, hptr, jptr, border
end

# Sorting of vectors of tuples in the symbolic setup. A GPU backend whose arrays cannot sort
# tuples overloads these.
_sort_tuples!(v; kw...) = sort!(v; kw...)
_sortperm_tuples(v; kw...) = sortperm(v; kw...)
_permute_tuples!(v, p) = permute!(v, p)

# A solver whose options select a Cholesky factorization (cholmod_algorithm,
# cudss_algorithm, ...) cannot factor the bordered matrix; anything else is assumed
# to handle a symmetric quasidefinite matrix.
function _factors_indefinite(opt::AbstractOptions)
    for name in fieldnames(typeof(opt))
        v = getfield(opt, name)
        if v isa LinearFactorization && v == CHOLESKY
            return false
        end
    end
    return true
end

get_sparse_condensed_ext(::Type{Vector{T}},args...) where T = nothing

num_variables(kkt::SparseCondensedKKTSystem) = length(kkt.pr_diag)
# the condensed block of the sparse rows is positive definite and the border carries
# one negative pivot per kept row
function is_inertia_correct(kkt::SparseCondensedKKTSystem, num_pos, num_zero, num_neg)
    return (num_zero == 0) && (num_pos == size(kkt.hess_com, 1)) && (num_neg == num_border_rows(kkt.border))
end
# the slacks and the inequality duals of the condensed rows are eliminated, so
# num_variables (which counts the slacks) is NOT the primal block size.
block_structure(kkt::SparseCondensedKKTSystem) = (size(kkt.hess_com, 1), num_border_rows(kkt.border))
should_regularize_dual(kkt::SparseCondensedKKTSystem, num_pos, num_zero, num_neg) = true
# with the equalities condensed through the shift (eq_shift > 0) the matrix is positive definite
# without delta_c, and a per-iteration delta_c would make every step that of a regularized system
# (on case9241_pegase at delta = 1e-6: 110 iterations without it, the iteration limit with it)
always_regularize_dual(kkt::SparseCondensedKKTSystem) = iszero(kkt.eq_shift[1])

Base.size(kkt::SparseCondensedKKTSystem,n::Int) = size(kkt.aug_com,n)

function compress_jacobian!(kkt::SparseCondensedKKTSystem{T, VT, MT}) where {T, VT, MT<:SparseMatrixCSC{T, Int32}}
    ns = length(kkt.ind_ineq)
    transfer!(kkt.jt_csc, kkt.jt_coo, kkt.jt_csc_map)
end

function jtprod!(y::AbstractVector, kkt::SparseCondensedKKTSystem, x::AbstractVector)
    n = size(kkt.hess_com, 1)
    mul!(view(y, 1:n), kkt.jt_csc, x)
    view(y, (n + 1):length(y)) .= .-view(x, kkt.ind_ineq)
    return y
end

function _sym_length(Jt)
    len = 0
    for i=1:size(Jt,2)
        n = Jt.colptr[i+1] - Jt.colptr[i]
        len += div(n^2 + n, 2)
    end
    return len
end

function _build_condensed_aug_symbolic_hess(H, sym, sym2)
    for i in 1:size(H,2)
        for j in H.colptr[i]:H.colptr[i+1]-1
            c = H.rowval[j]
            sym[j] = (0,j,0)
            sym2[j] = (c,i)
        end
    end
end

function _build_condensed_aug_symbolic_jt(Jt, sym, sym2)

    cnt = 0
    for i in 1:size(Jt,2)
        for j in Jt.colptr[i]:Jt.colptr[i+1]-1
            for k in j:Jt.colptr[i+1]-1
                c1 = Jt.rowval[j]
                c2 = Jt.rowval[k]
                sym[cnt+=1] = (i,j,k)
                sym2[cnt] = (c2,c1)
            end
        end
    end
end

function getptr(array; by = (x,y)->x != y)
    bitarray = similar(array, Bool, length(array)+1)
    fill!(bitarray, true)
    bitarray[2:end-1] .= by.(@view(array[1:end-1]),  @view(array[2:end]))
    findall(bitarray)
end

nzval(H) = H.nzval

@inbounds function build_condensed_aug_symbolic(H::AbstractSparseMatrix{Tv,Ti}, Jt) where {Tv, Ti}
    nnzjtsj = _sym_length(Jt)

    sym = similar(nzval(H), Tuple{Int,Int,Int},
        size(H,2) + nnz(H) + nnzjtsj
    )
    sym2 = similar(nzval(H), Tuple{Int,Int},
        size(H,2) + nnz(H) + nnzjtsj
    )
    colptr = fill!(
        similar(nzval(H), Ti, size(H,1)+1),
        one(Tv)
    )

    n = size(H,2)

    # N.B.: we should ensure that zind is allocated on the proper device.
    zind = similar(nzval(H), Int, size(H, 2))
    zind .= 1:size(H, 2)
    map!(
        i->(-1,i,0),
        @view(sym[1:n]),
        zind,
    )
    map!(
        i->(i,i),
        @view(sym2[1:n]),
        zind,
    )

    _build_condensed_aug_symbolic_hess(
        H,
        @view(sym[n+1:n+nnz(H)]),
        @view(sym2[n+1:n+nnz(H)])
    )
    _build_condensed_aug_symbolic_jt(
        Jt,
        @view(sym[n+nnz(H)+1:n+nnz(H) + nnzjtsj]),
        @view(sym2[n+nnz(H)+1:n+nnz(H)+nnzjtsj])
    )

    p = _sortperm_tuples(sym2; by = ((i,j),) -> (j,i))
    _permute_tuples!(sym, p)
    _permute_tuples!(sym2, p)

    by(x,y) = x != y

    bitarray = similar(sym2, Bool, length(sym2))
    fill!(bitarray, true)
    bitarray[2:end] .= by.(@view(sym2[1:end-1]),  @view(sym2[2:end]))
    # `similar` + `map!` into concretely typed arrays, and `findall`'s result converted to
    # one: `map` over views of these is only inferred as an abstract (GPU) array, and code
    # compiled for abstract arrays here is invalidated whenever a package adds a `getindex`
    # method (ExaModels does), which recompiles the KKT system's construction. `convert`
    # rather than a type assertion: some GPU packages' `findall` allocates in the default
    # memory rather than `sym`'s.
    IV = typeof(similar(sym, Int, 0))
    guide = cumsum!(similar(sym, Int, length(bitarray)), bitarray)

    b = convert(IV, findall(x->x[1] == -1, sym))
    dptr = similar(sym, Tuple{Int32,Int32}, length(b))
    map!((x,y)->(Int32(x),Int32(y[2])), dptr, @view(guide[b]), @view(sym[b]))

    b = convert(IV, findall(x->x[1] == 0, sym))
    hptr = similar(sym, Tuple{Int32,Int32}, length(b))
    map!((x,y)->(Int32(x),Int32(y[2])), hptr, @view(guide[b]), @view(sym[b]))

    b = convert(IV, findall(x->x[1] != -1 && x[1] != 0, sym))
    jptr = similar(sym, Tuple{Int32,eltype(sym)}, length(b))
    map!((x,y)->(Int32(x),y), jptr, @view(guide[b]), @view(sym[b]))

    ptr = convert(IV, findall(bitarray))
    rowval = similar(sym2, Int32, length(ptr))
    map!(((row,col),)->Int32(row), rowval, @view(sym2[ptr]))

    by2(x,y) = x[2] != y[2]
    bitarray[2:end] .= by2.(@view(sym2[1:end-1]),  @view(sym2[2:end]))
    ptr2 = convert(IV, findall(bitarray))

    first, last = _first_and_last_col(sym2,ptr2)

    fill!(
        @view(colptr[1:first]),
        1
    )

    _set_colptr!(colptr, ptr2, sym2, guide)

    fill!(
        @view(colptr[last+1:end]),
        length(ptr)+1
    )

    aug_com = _get_sparse_csc(
        size(H),
        colptr,
        rowval,
        similar(nzval(H), length(ptr))
    )

    return aug_com, dptr, hptr, jptr
end

function _get_sparse_csc(dims, colptr, rowval, nzval)
    SparseMatrixCSC(
        dims...,
        colptr,
        rowval,
        nzval
    )
end

function _first_and_last_col(sym2,ptr2)
    first= sym2[1][2]
    last = sym2[ptr2[end]][2]
    return (first, last)
end

function _set_colptr!(colptr, ptr2, sym2, guide)
    for i in @view(ptr2[2:end])

        (~,prevcol) = sym2[i-1]
        (row,col) = sym2[i]

        fill!(@view(colptr[prevcol+1:col]), guide[i])
    end
end

@inbounds function _build_condensed_aug_coord!(aug_com::SparseMatrixCSC{Tv,Ti}, pr_diag, H, Jt, diag_buffer, dptr, hptr, jptr) where {Tv, Ti}
    fill!(aug_com.nzval, zero(Tv))

    @simd for idx in eachindex(hptr)
        i,j = hptr[idx]
        aug_com.nzval[i] += H.nzval[j]
    end

    @simd for idx in eachindex(dptr)
        i,j = dptr[idx]
        aug_com.nzval[i] += pr_diag[j]
    end

    @simd for idx in eachindex(jptr)
        (i,(j,k,l)) = jptr[idx]
        aug_com.nzval[i] += diag_buffer[j] * Jt.nzval[k] * Jt.nzval[l]
    end
end

# the condensed block H + Σx + Jtᵀ D Jt into aug, from the Jacobian columns Jt and the
# diagonal D; the symbolic maps dptr/hptr/jptr were built for exactly these arguments
function _build_condensed_aug_coord!(kkt::AbstractCondensedKKTSystem{T,VT,MT}, aug::MT, Jt::MT, D::VT) where {T, VT, MT <: SparseMatrixCSC{T}}
    _build_condensed_aug_coord!(aug, kkt.pr_diag, kkt.hess_com, Jt, D, kkt.dptr, kkt.hptr, kkt.jptr)
end

build_condensed_aug_coord!(kkt::AbstractCondensedKKTSystem) = _build_condensed_aug_coord!(kkt, kkt.aug_com, kkt.jt_csc, kkt.diag_buffer)

# The weights of the condensed equality rows: D_i = 1 / (δ_i - Σd_i), with δ_i = δ ‖J_i‖² when
# eq_scaled (a row of zero norm takes ‖J_i‖² = 1e-8).
function _set_eq_weights!(kkt::SparseCondensedKKTSystem{T}, Σd) where {T}
    kkt.eq_shift[2] = zero(T)
    δ = kkt.eq_shift[1]
    override = kkt.eq_shift[11] > 0
    override && (δ = kkt.eq_shift[11])
    if kkt.eq_from_weights && !override
        if !isempty(kkt.ind_ineq)
            Dmax = maximum(view(kkt.diag_buffer, kkt.ind_ineq))
            δ = clamp(kkt.eq_shift[6] / max(Dmax, floatmin(T)), δ, max(T(1e-6), δ))
        end
        # a lower bound raised by a retry decays tenfold per factorization back to the base
        kkt.eq_shift[1] = max(kkt.eq_shift[3], kkt.eq_shift[1] / 10)
    end
    kkt.eq_shift[7] = δ
    # the primal shift of the factorization follows the equality shift (primal_shift_ratio > 0)
    kkt.eq_shift[5] > 0 && _set_primal_shift!(kkt.linear_solver.wrapped, kkt.eq_shift[5] * δ)
    if kkt.eq_scaled
        nonzeros(kkt.jt_sq) .= nonzeros(kkt.jt_csc) .^ 2
        mul!(kkt.eq_rownorm, kkt.jt_sq', kkt.eq_ones)
        view(kkt.diag_buffer, kkt.ind_eq) .= one(T) ./ (δ .* max.(view(kkt.eq_rownorm, kkt.ind_eq), T(1e-8)) .- view(Σd, kkt.ind_eq))
    else
        view(kkt.diag_buffer, kkt.ind_eq) .= one(T) ./ (δ .- view(Σd, kkt.ind_eq))
    end
    return
end

_set_primal_shift!(M, s) = nothing

# A fresh factorization uses the base equality shift (any adjustment of the last one is dropped).
function begin_factorization!(kkt::SparseCondensedKKTSystem{T}) where {T}
    kkt.eq_shift[10] = zero(T)
    kkt.eq_shift[11] = zero(T)
    return
end

# After a refined solve failed: adjust the equality shift of this factorization, at most twice, a
# hundredfold down if the refinement still contracted (the shifted matrix is too far from the
# operator) and up if it stagnated (the factorization is inaccurate), within [1e-10, 1e-4].
# Returns whether the matrix changed (the caller rebuilds, refactorizes and solves again).
function improve_kkt!(kkt::SparseCondensedKKTSystem{T}) where {T}
    (!iszero(kkt.eq_shift[1]) && !isempty(kkt.ind_eq) && kkt.eq_shift[10] < 2) || return false
    δ = kkt.eq_shift[7]
    δnew = kkt.eq_shift[9] <= T(0.9) ? max(δ / 100, T(1e-10)) : min(δ * 100, T(1e-4))
    δnew == δ && return false
    kkt.eq_shift[10] += 1
    kkt.eq_shift[11] = δnew
    return true
end

# Pivot feedback (equality_shift_pivot_target = τ > 0): after a factorization, the ratio r of the
# smallest to the largest pivot of the condensed matrix is about δ λ_min / ‖J_E‖², proportional to δ,
# so δ τ / r brings it to τ. Below τ / 10 the factorization is refactorized at once with the new δ;
# otherwise the new δ applies from the next factorization. Within [equality_shift, equality_shift_max].
# Returns whether the KKT system wants to be rebuilt and refactorized.
function pivot_feedback!(kkt::SparseCondensedKKTSystem{T}) where {T}
    τ = kkt.eq_shift[8]
    (τ > 0 && !isempty(kkt.ind_eq)) || return false
    r = pivot_ratio(kkt.linear_solver)
    r === nothing && return false
    r = max(T(r), floatmin(T))
    δ_used = kkt.eq_shift[7]
    δ = clamp(δ_used * τ / r, kkt.eq_shift[3], kkt.eq_shift[4])
    kkt.eq_shift[1] = δ
    return r < τ / 10 && δ > δ_used
end

# Refinement feedback (eq_adaptive): a refinement that contracts steadily but slowly means δ is
# too large (the preconditioner is far from the operator), one that stagnates or diverges means the
# factorization is inaccurate and δ is too small. δ moves tenfold, at most once per factorization,
# within [1e-10, 1e-4]; the next build_kkt! uses it.
const EQ_SHIFT_MIN = 1e-10
const EQ_SHIFT_MAX = 1e-4
function refinement_feedback!(kkt::SparseCondensedKKTSystem{T}, steps, r_first, r_last, converged) where {T}
    steps >= 2 || return
    ρ = (r_last / max(r_first, floatmin(T)))^(one(T) / (steps - 1))
    kkt.eq_shift[9] = ρ
    (kkt.eq_adaptive && !kkt.eq_from_weights && iszero(kkt.eq_shift[2])) || return
    δ = kkt.eq_shift[1]
    if !converged && ρ > T(0.9)
        δ = min(δ * 10, kkt.eq_shift[4])
    elseif steps >= 4 && ρ <= T(0.9)
        δ = max(δ / 10, T(EQ_SHIFT_MIN))
    end
    if δ != kkt.eq_shift[1]
        kkt.eq_shift[1] = δ
        kkt.eq_shift[2] = one(T)
    end
    return
end

# Before the restoration phase (eq_adaptive): retry with δ a hundredfold larger, up to EQ_SHIFT_MAX.
function retry_with_larger_shift!(kkt::SparseCondensedKKTSystem{T}) where {T}
    # only from the base shift: if the retried step fails too, or the shift is still raised, the
    # restoration phase runs as usual
    (kkt.eq_adaptive && kkt.eq_shift[1] <= kkt.eq_shift[3] && kkt.eq_shift[1] < kkt.eq_shift[4]) || return false
    kkt.eq_shift[1] = min(kkt.eq_shift[1] * 100, kkt.eq_shift[4])
    return true
end

function build_kkt!(kkt::SparseCondensedKKTSystem{T}) where {T}
    n = size(kkt.hess_com, 1)
    ns = length(kkt.ind_ineq)

    Σs = view(kkt.pr_diag, (n + 1):(n + ns))
    Σd = kkt.du_diag

    b = kkt.border
    if isempty(b) && isempty(kkt.ind_eq)
        # no row kept and no equality row: the slacks are the rows
        kkt.diag_buffer .= Σs ./ (1 .- (Σd .- kkt.slack_shift) .* Σs)
        build_condensed_aug_coord!(kkt)
        update_kkt_scaling!(kkt, kkt.linear_solver)
        return
    end
    if isempty(b)
        # every row condensed, the equality rows through the shift: D = 1 / (δ - Σd)
        fill!(kkt.diag_buffer, zero(T))
        view(kkt.diag_buffer, kkt.ind_ineq) .= Σs ./ (1 .- (view(Σd, kkt.ind_ineq) .- kkt.slack_shift) .* Σs)
        _set_eq_weights!(kkt, Σd)
        build_condensed_aug_coord!(kkt)
        update_kkt_scaling!(kkt, kkt.linear_solver)
        return
    end
    # D = Σs / (1 - Σd Σs) on the inequality rows (zero on the equality rows, which are kept);
    # the border diagonal Σd - 1/Σs on the inequality rows, Σd on the equality rows
    fill!(kkt.diag_buffer, zero(T))
    view(kkt.diag_buffer, kkt.ind_ineq) .= Σs ./ (1 .- (view(Σd, kkt.ind_ineq) .- kkt.slack_shift) .* Σs)
    if !iszero(kkt.eq_shift[1])
        # condensed equality rows: D = 1 / (δ - Σd); the kept ones are zeroed with ind_D below
        _set_eq_weights!(kkt, Σd)
    end
    kkt.buffer2 .= Σd
    view(kkt.buffer2, kkt.ind_ineq) .-= one(T) ./ Σs
    # condensed block of the sparse rows, then the border
    nonzeros(b.jt_s) .= view(nonzeros(kkt.jt_csc), b.smap)
    b.dbuf_s .= view(kkt.diag_buffer, b.ind_S)
    _build_condensed_aug_coord!(kkt, b.aug_c, b.jt_s, b.dbuf_s)
    view(nonzeros(kkt.aug_com), b.cmap) .= nonzeros(b.aug_c)
    view(nonzeros(kkt.aug_com), b.dmap_pos) .= view(nonzeros(kkt.jt_csc), b.dmap_src)
    view(nonzeros(kkt.aug_com), b.epos) .= view(kkt.buffer2, b.ind_D) .- b.delta[1]
    update_kkt_scaling!(kkt, kkt.linear_solver)
    return
end

# A scaling of the factorized matrix computed from the KKT system rather than from that matrix
# (AugmentedRuizScaler); the linear solvers that scale specialize this.
update_kkt_scaling!(kkt, linear_solver) = nothing

get_jacobian(kkt::SparseCondensedKKTSystem) = kkt.jac

