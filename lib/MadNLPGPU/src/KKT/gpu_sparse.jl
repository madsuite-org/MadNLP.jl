######################################################
##### GPU wrappers for SparseCondensedKKTSystem  #####
######################################################

# Generic GPU sparse functions dispatching on AbstractGPUVector / AbstractGPUArray.
# Per-backend stubs for functions needing concrete sparse CSC types
# (transfer!, nzval, _get_sparse_csc, get_tril_to_full, _sym_length)
# remain in per-extension files.

#=
    MadNLP.mul! with SparseCondensedKKTSystem + GPU
=#

function MadNLP.mul!(
        w::MadNLP.AbstractKKTVector{T, VT},
        kkt::MadNLP.SparseCondensedKKTSystem,
        x::MadNLP.AbstractKKTVector,
        alpha = one(T),
        beta = zero(T),
    ) where {T, VT <: AbstractGPUVector{T}}
    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc, 2)

    # Decompose results
    xx = view(MadNLP.full(x), 1:n)
    xs = view(MadNLP.full(x), (n + 1):(n + m))
    xz = view(MadNLP.full(x), (n + m + 1):(n + 2 * m))

    # Decompose buffers
    wx = view(MadNLP.full(w), 1:n)
    ws = view(MadNLP.full(w), (n + 1):(n + m))
    wz = view(MadNLP.full(w), (n + m + 1):(n + 2 * m))

    MadNLP.mul!(wx, kkt.hess_com, xx, alpha, beta)
    MadNLP.mul!(wx, kkt.hess_com', xx, alpha, one(T))
    MadNLP.mul!(wx, kkt.jt_csc, xz, alpha, one(T))
    if !isempty(kkt.ext.diag_map_to)
        backend = get_backend(MadNLP.full(w))
        _diag_operation_kernel!(backend)(
            wx,
            kkt.hess_com.nzVal,
            xx,
            alpha,
            kkt.ext.diag_map_to,
            kkt.ext.diag_map_fr;
            ndrange = length(kkt.ext.diag_map_to),
        )
    end

    MadNLP.mul!(wz, kkt.jt_csc', xx, alpha, beta)
    MadNLP.axpy!(-alpha, xs, wz)
    ws .= beta .* ws .- alpha .* xz
    return MadNLP._kktmul!(
        w,
        x,
        kkt.reg,
        kkt.du_diag,
        kkt.l_lower,
        kkt.u_lower,
        kkt.l_diag,
        kkt.u_diag,
        alpha,
        beta,
    )
end

function MadNLP.mul_hess_blk!(
        wx::VT,
        kkt::Union{MadNLP.SparseKKTSystem, MadNLP.SparseCondensedKKTSystem},
        t,
    ) where {T, VT <: AbstractGPUVector{T}}
    n = size(kkt.hess_com, 1)
    wxx = @view(wx[1:n])
    tx = @view(t[1:n])

    MadNLP.mul!(wxx, kkt.hess_com, tx, one(T), zero(T))
    MadNLP.mul!(wxx, kkt.hess_com', tx, one(T), one(T))
    if !isempty(kkt.ext.diag_map_to)
        backend = get_backend(wx)
        _diag_operation_kernel!(backend)(
            wxx,
            kkt.hess_com.nzVal,
            tx,
            one(T),
            kkt.ext.diag_map_to,
            kkt.ext.diag_map_fr;
            ndrange = length(kkt.ext.diag_map_to),
        )
    end

    fill!(@view(wx[(n + 1):end]), 0)
    wx .+= t .* kkt.pr_diag
    return
end

#=
    get_sparse_condensed_ext
=#

function MadNLP.get_sparse_condensed_ext(
        ::Type{VT},
        hess_com,
        jptr,
        jt_map,
        hess_map,
    ) where {T, VT <: AbstractGPUVector{T}}
    zvals = adapt(get_backend(hess_map), collect(1:length(hess_map)))
    hess_com_ptr = map((i, j) -> (i, j), hess_map, zvals)
    if length(hess_com_ptr) > 0 # otherwise error is thrown
        sort!(hess_com_ptr)
    end

    jvals = adapt(get_backend(jt_map), collect(1:length(jt_map)))
    jt_csc_ptr = map((i, j) -> (i, j), jt_map, jvals)
    if length(jt_csc_ptr) > 0 # otherwise error is thrown
        sort!(jt_csc_ptr)
    end

    by = (i, j) -> i[1] != j[1]
    jptrptr = MadNLP.getptr(jptr, by = by)
    hess_com_ptrptr = MadNLP.getptr(hess_com_ptr, by = by)
    jt_csc_ptrptr = MadNLP.getptr(jt_csc_ptr, by = by)

    diag_map_to, diag_map_fr = get_diagonal_mapping(hess_com.colPtr, hess_com.rowVal)

    return (
        jptrptr = jptrptr,
        hess_com_ptr = hess_com_ptr,
        hess_com_ptrptr = hess_com_ptrptr,
        jt_csc_ptr = jt_csc_ptr,
        jt_csc_ptrptr = jt_csc_ptrptr,
        diag_map_to = diag_map_to,
        diag_map_fr = diag_map_fr,
    )
end

#=
    get_diagonal_mapping
=#

function get_diagonal_mapping(colptr, rowval)
    nnz = length(rowval)
    if nnz == 0
        return similar(colptr, 0), similar(colptr, 0)
    end
    inds1 = findall(
        map(
            (x, y) -> ((x <= nnz) && (x != y)),
            @view(colptr[1:(end - 1)]),
            @view(colptr[2:end])
        ),
    )
    if length(inds1) == 0
        return similar(rowval, 0), similar(colptr, 0)
    end
    ptrs = colptr[inds1]
    rows = rowval[ptrs]
    inds2 = findall(inds1 .== rows)
    if length(inds2) == 0
        return similar(rows, 0), similar(ptrs, 0)
    end

    return rows[inds2], ptrs[inds2]
end

#=
    MadNLP._first_and_last_col
=#

function MadNLP._first_and_last_col(sym2::AbstractGPUVector, ptr2)
    @allowscalar begin
        first = sym2[1][2]
        last = sym2[ptr2[end]][2]
    end
    return (first, last)
end

#=
    getij helper
=#

function getij(idx, n)
    j = ceil(Int, ((2n + 1) - sqrt((2n + 1)^2 - 8 * idx)) / 2)
    i = idx - div((j - 1) * (2n - j), 2)
    return (i, j)
end

#=
    MadNLP._set_colptr!
=#

function MadNLP._set_colptr!(colptr::AbstractGPUVector, ptr2, sym2, guide)
    if length(ptr2) > 1 # otherwise error is thrown
        backend = get_backend(colptr)
        _set_colptr_kernel!(backend)(
            colptr,
            sym2,
            ptr2,
            guide;
            ndrange = length(ptr2) - 1,
        )
    end
    return
end

#=
    MadNLP.tril_to_full! (dense GPU matrix version)
=#

function MadNLP.tril_to_full!(dense::AbstractGPUMatrix{T}) where {T}
    n = size(dense, 1)
    backend = get_backend(dense)
    _tril_to_full_kernel!(backend)(dense; ndrange = div(n^2 + n, 2))
    return
end

#=
    MadNLP.force_lower_triangular!
=#

function MadNLP.force_lower_triangular!(I::AbstractGPUVector{T}, J) where {T}
    if !isempty(I)
        backend = get_backend(I)
        _force_lower_triangular_kernel!(backend)(I, J; ndrange = length(I))
    end
    return
end

#=
    MadNLP._set_con_scale_sparse!
=#

function MadNLP._set_con_scale_sparse!(
        con_scale::VT,
        jac_I,
        jac_buffer,
    ) where {T, VT <: AbstractGPUVector{T}}
    ind_jac = adapt(get_backend(jac_I), collect(1:length(jac_I)))
    inds = map((i, j) -> (i, j), jac_I, ind_jac)
    !isempty(inds) && sort!(inds)
    ptr = MadNLP.getptr(inds; by = ((x1, x2), (y1, y2)) -> x1 != y1)
    if length(ptr) > 1
        backend = get_backend(con_scale)
        _set_con_scale_sparse_kernel!(backend)(
            con_scale,
            ptr,
            inds,
            jac_I,
            jac_buffer;
            ndrange = length(ptr) - 1,
        )
    end
    return
end

#=
    MadNLP.coo_to_csc
=#

function MadNLP.coo_to_csc(
        coo::MadNLP.SparseMatrixCOO{T, I, VT, VI},
    ) where {T, I, VT <: AbstractGPUArray, VI <: AbstractGPUArray}
    zvals = adapt(get_backend(coo.I), collect(1:length(coo.I)))
    coord = map((i, j, k) -> ((i, j), k), coo.I, coo.J, zvals)
    if length(coord) > 0
        sort!(coord, lt = (((i, j), k), ((n, m), l)) -> (j, i) < (m, n))
    end

    mapptr = MadNLP.getptr(coord; by = ((x1, x2), (y1, y2)) -> x1 != y1)

    colptr = similar(coo.I, size(coo, 2) + 1)

    coord_csc = coord[@view(mapptr[1:(end - 1)])]

    backend = get_backend(coo.I)
    if length(coord_csc) > 0
        _set_coo_to_colptr_kernel!(backend)(
            colptr,
            coord_csc,
            ndrange = length(coord_csc),
        )
    else
        fill!(colptr, one(Int))
    end

    rowval = map(x -> x[1][1], coord_csc)
    nzval = similar(rowval, T)

    csc = MadNLP._get_sparse_csc(size(coo), colptr, rowval, nzval)

    cscmap = similar(coo.I, Int)
    if length(mapptr) > 1
        _set_coo_to_csc_map_kernel!(backend)(
            cscmap,
            mapptr,
            coord,
            ndrange = length(mapptr) - 1,
        )
    end

    return csc, cscmap
end

#=
    MadNLP.build_condensed_aug_coord!
=#

# the condensed block into `aug` from the Jacobian columns `Jt` and the diagonal `D`:
# the whole system (aug_com, jt_csc, diag_buffer) or, with dense rows kept as a
# border, the block of the condensed rows (aug_c, jt_s, dbuf_s)
function MadNLP._build_condensed_aug_coord!(
        kkt::MadNLP.AbstractCondensedKKTSystem{T, VT}, aug, Jt, D,
    ) where {T, VT <: AbstractGPUVector{T}}
    fill!(MadNLP.nzval(aug), zero(T))
    backend = get_backend(kkt.pr_diag)
    if length(kkt.hptr) > 0
        _transfer_hessian_kernel!(backend)(
            MadNLP.nzval(aug),
            kkt.hptr,
            MadNLP.nzval(kkt.hess_com);
            ndrange = length(kkt.hptr),
        )
    end
    if length(kkt.dptr) > 0
        _transfer_hessian_kernel!(backend)(
            MadNLP.nzval(aug),
            kkt.dptr,
            kkt.pr_diag;
            ndrange = length(kkt.dptr),
        )
    end
    if length(kkt.ext.jptrptr) > 1 # otherwise error is thrown
        _transfer_jtsj_kernel!(backend)(
            MadNLP.nzval(aug),
            kkt.jptr,
            kkt.ext.jptrptr,
            MadNLP.nzval(Jt),
            D;
            ndrange = length(kkt.ext.jptrptr) - 1,
        )
    end
    return
end

#=
    MadNLP.compress_hessian!
=#

function MadNLP.compress_hessian!(
        kkt::MadNLP.AbstractSparseKKTSystem{T, VT},
    ) where {T, VT <: AbstractGPUVector{T}}
    fill!(MadNLP.nzval(kkt.hess_com), zero(T))
    backend = get_backend(kkt.pr_diag)
    if length(kkt.ext.hess_com_ptrptr) > 1
        _transfer_to_csc_kernel!(backend)(
            MadNLP.nzval(kkt.hess_com),
            kkt.ext.hess_com_ptr,
            kkt.ext.hess_com_ptrptr,
            kkt.hess_raw.V;
            ndrange = length(kkt.ext.hess_com_ptrptr) - 1,
        )
    end
    return
end

#=
    MadNLP.compress_jacobian!
=#

function MadNLP.compress_jacobian!(
        kkt::MadNLP.SparseCondensedKKTSystem{T, VT},
    ) where {T, VT <: AbstractGPUVector{T}}
    fill!(MadNLP.nzval(kkt.jt_csc), zero(T))
    backend = get_backend(kkt.pr_diag)
    if length(kkt.ext.jt_csc_ptrptr) > 1 # otherwise error is thrown
        _transfer_to_csc_kernel!(backend)(
            MadNLP.nzval(kkt.jt_csc),
            kkt.ext.jt_csc_ptr,
            kkt.ext.jt_csc_ptrptr,
            kkt.jt_coo.V;
            ndrange = length(kkt.ext.jt_csc_ptrptr) - 1,
        )
    end
    return
end

#=
    MadNLP._build_condensed_aug_symbolic_hess
=#

function MadNLP._build_condensed_aug_symbolic_hess(
        H,
        sym,
        sym2::AbstractGPUVector,
    )
    if size(H, 2) > 0
        backend = get_backend(sym2)
        _build_condensed_aug_symbolic_hess_kernel!(backend)(
            sym,
            sym2,
            H.colPtr,
            H.rowVal;
            ndrange = size(H, 2),
        )
    end
    return
end

#=
    MadNLP._build_condensed_aug_symbolic_jt
=#

function MadNLP._build_condensed_aug_symbolic_jt(
        Jt,
        sym,
        sym2::AbstractGPUVector,
    )
    if size(Jt, 2) > 0
        _offsets = map(
            (i, j) -> div((j - i)^2 + (j - i), 2),
            @view(Jt.colPtr[1:(end - 1)]),
            @view(Jt.colPtr[2:end])
        )
        offsets = cumsum(_offsets)
        backend = get_backend(sym2)
        _build_condensed_aug_symbolic_jt_kernel!(backend)(
            sym,
            sym2,
            Jt.colPtr,
            Jt.rowVal,
            offsets;
            ndrange = size(Jt, 2),
        )
    end
    return
end

#=
    MadNLP._build_scale_augmented_system_coo!
=#

function MadNLP._build_scale_augmented_system_coo!(dest, src, scaling::AbstractGPUArray, n, m)
    backend = get_backend(scaling)
    _scale_augmented_system_coo_kernel!(backend)(
        dest.V,
        src.I,
        src.J,
        src.V,
        scaling,
        n,
        m;
        ndrange = nnz(src),
    )
    return
end
