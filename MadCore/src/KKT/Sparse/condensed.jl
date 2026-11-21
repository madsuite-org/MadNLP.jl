# The state of the equality shift and of the inertia retry (inertia_retry!), with its options.
@kwdef mutable struct EqualityShift{T}
    base::T                       # the shift of a fresh factorization (equality_shift)
    retry::Bool                   # the retry on (equality_shift_retry)
    factor::T                     # a retry multiplies the shift by this (equality_shift_retry_factor)
    count::Int                    # retries per step (equality_shift_retry_count)
    max::T                        # the largest shift a retry may reach (equality_shift_max)
    backoff_max::Int              # the longest backoff, in iterations (equality_shift_retry_backoff)
    window::Int                   # watchdog window, in iterations (equality_shift_watchdog_window)
    hold::Int                     # watchdog hold, in iterations (equality_shift_watchdog_hold)
    ratio::T                      # the watchdog expects the primal infeasibility below this ratio of its start (equality_shift_watchdog_ratio)
    current::T = base             # the shift of the last factorization
    override::T = zero(T)         # the shift of the next factorization, set by a retry (0: the base)
    retries::Int = 0              # retries in the current step
    step::Int = -1                # that step
    backoff::Int = 0              # the current backoff, in iterations: doubles after each step whose retries all failed
    backoff_until::Int = 0        # no retry before this iteration (backoff after failed retries)
    window_start::Int = 0         # first iteration of the current watchdog window
    window_inf_pr::T = zero(T)    # the primal infeasibility then
    window_retries::Int = 0       # retries used in the window
    hold_until::Int = 0           # no retry before this iteration (watchdog)
end


"""
    SparseCondensedKKTSystem{T, VT, MT, QN} <: AbstractCondensedKKTSystem{T, VT, MT, QN}

Implement the [`AbstractCondensedKKTSystem`](@ref) in sparse COO format.

"""
struct SparseCondensedKKTSystem{T, VT, MT, QN, LS, VI, VI32, VTu1, VTu2, EXT} <: AbstractCondensedKKTSystem{T, VT, MT, QN}
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

    # Augmented system
    aug_com::MT

    # slack diagonal buffer
    diag_buffer::VT
    dptr::VTu1
    hptr::VTu1
    jptr::VTu2

    # LinearSolver
    linear_solver::LS

    # Info
    ind_ineq::VI
    ind_eq::VI
    # the equality rows are condensed with the weight D = 1 / (δ - Σd), δ the equality shift
    eq_shift::EqualityShift{T}
    ind_lb::VI
    ind_ub::VI

    # extra
    ext::EXT
end

"""
    create_kkt_system(::Type{SparseCondensedKKTSystem}, cb, linear_solver; kwargs...)

The condensed matrix H + Σx + Jᵀ D J is handed to the linear solver. An inequality row is
condensed through its slack, with the weight D = Σs / (1 - Σd Σs); an equality row has no slack
and its multiplier is eliminated through the equality shift δ (`equality_shift`), with the weight
D = 1 / (δ - Σd), which keeps the condensed matrix positive definite. Iterative refinement, which
applies the unshifted KKT operator, removes the bias of the shift. Condensing a dense Jacobian
row turns the condensed matrix dense (the row's outer product): `CondensedPreset` solves such a
model with `ShiftedPreset` instead.
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
    equality_shift::Float64 = 1e-8,
    equality_shift_retry::Bool = true,
    equality_shift_retry_factor::Float64 = 100.0,
    equality_shift_retry_count::Int = 2,
    equality_shift_max::Float64 = 1e-4,
    equality_shift_retry_backoff::Int = 64,
    equality_shift_watchdog_window::Int = 20,
    equality_shift_watchdog_hold::Int = 100,
    equality_shift_watchdog_ratio::Float64 = 0.5,
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

    aug_com, dptr, hptr, jptr = build_condensed_aug_symbolic(hess_com, jt_csc)

    # tell the linear solver the saddle-point partition before it analyzes
    _linear_solver = LinearSolverBarrier(create_kkt_linear_solver(linear_solver, aug_com, n; opt = opt_linear_solver, kkt_scaler, primal_shift, dual_shift))
    ext = get_sparse_condensed_ext(VT, hess_com, jptr, jt_csc_map, hess_csc_map)
    return SparseCondensedKKTSystem(
        hess, hess_raw, hess_com, hess_csc_map,
        jac, jt_coo, jt_csc, jt_csc_map,
        quasi_newton,
        reg, pr_diag, du_diag,
        l_diag, u_diag, l_lower, u_lower,
        buffer, buffer2, buffer3,
        aug_com, diag_buffer, dptr, hptr, jptr,
        _linear_solver,
        ind_ineq, _to_index_vector(ind_ineq, ind_eq_h),
        EqualityShift{T}(;
            base = equality_shift, retry = equality_shift_retry, factor = equality_shift_retry_factor,
            count = equality_shift_retry_count, max = equality_shift_max, backoff_max = equality_shift_retry_backoff,
            window = equality_shift_watchdog_window, hold = equality_shift_watchdog_hold, ratio = equality_shift_watchdog_ratio,
        ),
        cb.ind_lb, cb.ind_ub,
        ext
    )
end

_to_index_vector(like, v) = (o = similar(like, Int, length(v)); copyto!(o, v); o)

# Sorting of vectors of tuples in the symbolic setup. A GPU backend whose arrays cannot sort
# tuples overloads these.
_sort_tuples!(v; kw...) = sort!(v; kw...)
_sortperm_tuples(v; kw...) = sortperm(v; kw...)
_permute_tuples!(v, p) = permute!(v, p)

get_sparse_condensed_ext(::Type{Vector{T}},args...) where T = nothing

num_variables(kkt::SparseCondensedKKTSystem) = length(kkt.pr_diag)
# the condensed matrix is positive definite
function is_inertia_correct(kkt::SparseCondensedKKTSystem, num_pos, num_zero, num_neg)
    return (num_zero == 0) && (num_pos == size(kkt.hess_com, 1)) && (num_neg == 0)
end
# the slacks and the duals are eliminated, so num_variables (which counts the slacks) is NOT the
# primal block size.
block_structure(kkt::SparseCondensedKKTSystem) = (size(kkt.hess_com, 1), 0)
should_regularize_dual(kkt::SparseCondensedKKTSystem, num_pos, num_zero, num_neg) = true

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


# The weights of the condensed equality rows: D_i = 1 / (δ - Σd_i), with δ the base shift or, once,
# the override set by an inertia retry for the factorization being built.
function _set_eq_weights!(kkt::SparseCondensedKKTSystem{T}, Σd) where {T}
    e = kkt.eq_shift
    δ = e.override > 0 ? e.override : e.base
    e.override = zero(T)
    e.current = δ
    view(kkt.diag_buffer, kkt.ind_eq) .= one(T) ./ (δ .- view(Σd, kkt.ind_eq))
    return
end

# Wrong inertia at iteration k: before the Hessian is perturbed, the factorization is retried with δ
# `factor` times larger (100), at most `count` times per step (2), up to `max` (1e-4). A Cholesky
# factorization of the condensed matrix fails from cancellation once the reduced-Hessian curvature
# falls below the rounding error of the 1/δ terms, and the Hessian perturbation that follows makes
# the step a regularized Newton step that stalls near the solution (cont5_2_1_l: δ = 1e-8 needs
# δ_w ≈ 1e-3 on 85% of the iterations and stalls at the iteration limit; δ = 1e-6 needs δ_w on 17%
# and converges). The refinement still targets the unshifted system. Returns whether the matrix
# changed (the caller rebuilds and refactorizes).
#
# Two guards: (a) after a step whose retries all failed (a genuinely indefinite matrix), no retry
# for the next 2^n iterations after n consecutive such steps, at most `backoff_max` (64), reset by
# a step whose retry succeeded (arki0009 and NARX_CFy spent two factorizations per iteration on
# retries that never succeeded); (b) a retried step whose refinement does not converge can leave
# the equality residual in place (case8387_pegase_rect, case2853_sdet_polar: the primal
# infeasibility stalled above the tolerance for hundreds of iterations), so every `window`
# iterations (20), if retries were used and the primal infeasibility is above the tolerance and
# not below `ratio` (1/2) of its value at the start of the window, no retry for the next `hold`
# iterations (100).
function inertia_retry!(kkt::SparseCondensedKKTSystem{T}, k::Integer, inf_pr, tol) where {T}
    e = kkt.eq_shift
    (e.retry && !isempty(kkt.ind_eq)) || return false
    if k != e.step
        # a new step: the last one ended on a retried factorization when its retry succeeded, which
        # resets the backoff; a step whose retries all failed left the count one past `count`
        0 < e.retries <= e.count && (e.backoff = 0)
        e.retries = 0
        e.step = k
    end
    if e.retries >= e.count
        if e.retries == e.count
            e.backoff = min(2 * max(e.backoff, 1), e.backoff_max)
            e.backoff_until = k + e.backoff
            e.retries = e.count + 1
        end
        return false
    end
    k < e.backoff_until && return false
    k < e.hold_until && return false
    if k - e.window_start >= e.window
        if e.window_retries > 0 && inf_pr > tol && inf_pr > e.ratio * e.window_inf_pr
            e.hold_until = k + e.hold
        end
        e.window_start = k
        e.window_inf_pr = inf_pr
        e.window_retries = 0
        k < e.hold_until && return false
    end
    e.current >= e.max && return false
    e.window_retries += 1
    e.retries += 1
    e.override = min(e.current * e.factor, e.max)
    return true
end

function build_kkt!(kkt::SparseCondensedKKTSystem{T}) where {T}
    n = size(kkt.hess_com, 1)
    ns = length(kkt.ind_ineq)

    Σs = view(kkt.pr_diag, (n + 1):(n + ns))
    Σd = kkt.du_diag

    if isempty(kkt.ind_eq)
        # every row an inequality: the slacks are the rows
        kkt.diag_buffer .= Σs ./ (1 .- Σd .* Σs)
    else
        # D = Σs / (1 - Σd Σs) on the inequality rows and 1 / (δ - Σd) on the equality rows
        fill!(kkt.diag_buffer, zero(T))
        view(kkt.diag_buffer, kkt.ind_ineq) .= Σs ./ (1 .- view(Σd, kkt.ind_ineq) .* Σs)
        _set_eq_weights!(kkt, Σd)
    end
    build_condensed_aug_coord!(kkt)
    return
end

get_jacobian(kkt::SparseCondensedKKTSystem) = kkt.jac

