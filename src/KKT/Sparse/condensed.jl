
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
    ind_lb::VI
    ind_ub::VI

    # extra
    ext::EXT
end

function create_kkt_system(
    ::Type{SparseCondensedKKTSystem},
    cb::SparseCallback{T,VT},
    linear_solver::Type;
    opt_linear_solver=default_options(linear_solver),
    hessian_approximation=ExactHessian,
    qn_options=QuasiNewtonOptions(),
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


    reg = VT(undef, n_tot)
    pr_diag = VT(undef, n_tot)
    du_diag = VT(undef, m)
    l_diag = VT(undef, nlb)
    u_diag = VT(undef, nub)
    l_lower = VT(undef, nlb)
    u_lower = VT(undef, nub)
    buffer = VT(undef, m)
    buffer2 = VT(undef, m)
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

    aug_com, dptr, hptr, jptr = build_condensed_aug_symbolic(
        hess_com,
        jt_csc
    )

    _linear_solver = linear_solver(aug_com; opt = opt_linear_solver, pos = n_tot)
    ext = get_sparse_condensed_ext(VT, hess_com, jptr, jt_csc_map, hess_csc_map)
    return SparseCondensedKKTSystem(
        hess, hess_raw, hess_com, hess_csc_map,
        jac, jt_coo, jt_csc, jt_csc_map,
        quasi_newton,
        reg, pr_diag, du_diag,
        l_diag, u_diag, l_lower, u_lower,
        buffer, buffer2,
        aug_com, diag_buffer, dptr, hptr, jptr,
        _linear_solver,
        ind_ineq, cb.ind_lb, cb.ind_ub,
        ext
    )
end

get_sparse_condensed_ext(::Type{Vector{T}},args...) where T = nothing

num_variables(kkt::SparseCondensedKKTSystem) = length(kkt.pr_diag)
function is_inertia_correct(kkt::SparseCondensedKKTSystem, num_pos, num_zero, num_neg)
    return (num_zero == 0) && (num_pos == size(kkt.aug_com, 1))
end
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

function build_condensed_aug_coord!(kkt::AbstractCondensedKKTSystem{T,VT,MT}) where {T, VT, MT <: SparseMatrixCSC{T}}
    _build_condensed_aug_coord!(
        kkt.aug_com, kkt.pr_diag, kkt.hess_com, kkt.jt_csc, kkt.diag_buffer,
        kkt.dptr, kkt.hptr, kkt.jptr
    )
end

function build_kkt!(kkt::SparseCondensedKKTSystem)

    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc, 2)


    Σx = view(kkt.pr_diag, 1:n)
    Σs = view(kkt.pr_diag, n+1:n+m)
    Σd = kkt.du_diag

    kkt.diag_buffer .= Σs ./ ( 1 .- Σd .* Σs)
    build_condensed_aug_coord!(kkt)
end

get_jacobian(kkt::SparseCondensedKKTSystem) = kkt.jac

