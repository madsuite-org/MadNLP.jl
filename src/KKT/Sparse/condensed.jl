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
    ind_lb::VI
    ind_ub::VI

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
zero. Among the candidate rows the densest are kept, as many as minimize the estimated
nonzeros of the factored matrix (a condensed block is at most a dense n × n triangle, a
kept row costs 2 nnz + 1 border entries), and only when that saves more than
`dense_row_fill_min` nonzeros and the linear solver can factor an indefinite matrix (a
solver set to a Cholesky factorization condenses every row). With no row kept the
system is the plain condensed one.
"""
function create_kkt_system(
    ::Type{SparseCondensedKKTSystem},
    cb::SparseCallback{T,VT},
    linear_solver::Type;
    opt_linear_solver=default_options(linear_solver),
    hessian_approximation=ExactHessian,
    qn_options=QuasiNewtonOptions(),
    dense_row_abs::Int = 512,
    dense_row_rel::Float64 = 0.10,
    dense_row_fill_min::Float64 = 1e7,
    border_delta::Float64 = 1e-8,
    border_delta_exp::Float64 = 0.25,
) where {T, VT}
    ind_ineq = cb.ind_ineq
    n = cb.nvar
    m = cb.ncon
    n_slack = length(ind_ineq)

    if n_slack != m
        error("SparseCondensedKKTSystem does not support equality constrained NLPs.")
    end

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

    ind_D = _select_dense_rows(jac_sparsity_I, n, m, n_hess, dense_row_abs, dense_row_rel, dense_row_fill_min)
    if !isempty(ind_D) && !_factors_indefinite(opt_linear_solver)
        Base.@warn "$(linear_solver) is set to a Cholesky factorization and cannot factor the bordered condensed matrix; the $(length(ind_D)) dense Jacobian rows are condensed."
        ind_D = Int[]
    end
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
    set_block_structure!(opt_linear_solver, n, mD)
    _linear_solver = linear_solver(aug_com; opt = opt_linear_solver)
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
        ind_ineq, cb.ind_lb, cb.ind_ub,
        ext
    )
end

# The rows with more than dense_row_abs nonzeros and more than dense_row_rel * n are the
# candidates; the densest k of them are kept as the border, k chosen to minimize the
# estimated nonzeros of the factored matrix: the condensed block is bounded by a dense
# n x n triangle, so once the rows left condensed fill it, keeping more rows only adds
# border entries (2 nnz + 1 each). The border is used when the saving exceeds
# dense_row_fill_min. Returns the kept rows, sorted.
function _select_dense_rows(jac_sparsity_I, n, m, n_hess, dense_row_abs, dense_row_rel, dense_row_fill_min)
    cnt = zeros(Int, m)
    for r in Array(jac_sparsity_I)
        cnt[r] += 1
    end
    tri(c) = c * (c + 1) / 2
    cand = sort!(findall(i -> cnt[i] > dense_row_abs && cnt[i] > dense_row_rel * n, 1:m); by = i -> -cnt[i])
    dense_cap = tri(n)
    base = n_hess + sum(tri, cnt; init = 0.0)             # every row condensed, before the cap
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

Base.size(kkt::SparseCondensedKKTSystem,n::Int) = size(kkt.aug_com,n)

function compress_jacobian!(kkt::SparseCondensedKKTSystem{T, VT, MT}) where {T, VT, MT<:SparseMatrixCSC{T, Int32}}
    ns = length(kkt.ind_ineq)
    transfer!(kkt.jt_csc, kkt.jt_coo, kkt.jt_csc_map)
end

function jtprod!(y::AbstractVector, kkt::SparseCondensedKKTSystem, x::AbstractVector)
    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc, 2)

    mul!(view(y, 1:n), kkt.jt_csc, x)
    y[size(kkt.jt_csc,1)+1:end] .= -x
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
    dptr = similar(nzval(H), Tuple{Ti,Ti},
        size(H,2)
    )
    hptr = similar(nzval(H), Tuple{Ti,Ti},
        nnz(H)
    )
    jptr = similar(nzval(H), Tuple{Ti,Tuple{Ti,Ti,Ti}},
        nnzjtsj
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

    p = sortperm(sym2; by = ((i,j),) -> (j,i))
    permute!(sym, p)
    permute!(sym2, p)

    by(x,y) = x != y

    bitarray = similar(sym2, Bool, length(sym2))
    fill!(bitarray, true)
    bitarray[2:end] .= by.(@view(sym2[1:end-1]),  @view(sym2[2:end]))
    guide = cumsum(bitarray)

    b = findall(x->x[1] == -1, sym)
    dptr = map((x,y)->(Int32(x),Int32(y[2])), @view(guide[b]), @view(sym[b]))

    b = findall(x->x[1] == 0, sym)
    hptr = map((x,y)->(Int32(x),Int32(y[2])), @view(guide[b]), @view(sym[b]))

    b = findall(x->x[1] != -1 && x[1] != 0, sym)
    jptr = map((x,y)->(Int32(x),y), @view(guide[b]), @view(sym[b]))


    ptr = findall(bitarray)
    rowval = map(((row,col),)->Int32(row), @view(sym2[ptr]))

    by2(x,y) = x[2] != y[2]
    bitarray[2:end] .= by2.(@view(sym2[1:end-1]),  @view(sym2[2:end]))
    ptr2 = findall(bitarray)

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

function build_kkt!(kkt::SparseCondensedKKTSystem{T}) where {T}
    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc, 2)

    Σs = view(kkt.pr_diag, (n + 1):(n + m))
    Σd = kkt.du_diag

    kkt.diag_buffer .= Σs ./ (1 .- Σd .* Σs)
    b = kkt.border
    if isempty(b)
        build_condensed_aug_coord!(kkt)
        return
    end
    # condensed block of the sparse rows, then the border
    nonzeros(b.jt_s) .= view(nonzeros(kkt.jt_csc), b.smap)
    b.dbuf_s .= view(kkt.diag_buffer, b.ind_S)
    _build_condensed_aug_coord!(kkt, b.aug_c, b.jt_s, b.dbuf_s)
    view(nonzeros(kkt.aug_com), b.cmap) .= nonzeros(b.aug_c)
    view(nonzeros(kkt.aug_com), b.dmap_pos) .= view(nonzeros(kkt.jt_csc), b.dmap_src)
    view(nonzeros(kkt.aug_com), b.epos) .= view(Σd, b.ind_D) .- 1 ./ view(Σs, b.ind_D) .- b.delta[1]
    return
end

get_jacobian(kkt::SparseCondensedKKTSystem) = kkt.jac

