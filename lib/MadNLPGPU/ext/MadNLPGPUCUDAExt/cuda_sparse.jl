######################################################
##### CUDA stubs for SparseCondensedKKTSystem    #####
##### (functions requiring concrete CUSPARSE types) ##
######################################################

# Scatter COO values into CSC nonzeros, accumulating duplicates (matches the CPU
# `_transfer!`: `nzval[map[k]] += V[k]`). The augmented KKT's COO has duplicate
# (i,i) entries (primal-dual diagonal + Hessian diagonal) that must be summed, so
# a plain non-accumulating scatter (copyto!) would drop one of each pair.
#
# Done the same way as the condensed path (atomic-free, deterministic): group the
# COO sources by destination CSC slot once (sort + getptr) and let one thread sum
# each group with `_transfer_to_csc_kernel!`. SparseKKTSystem has no `ext` to hold
# the grouping, so it is memoized here for the lifetime of the csc map object.
#
# The cache must not keep the map alive: it is keyed by `objectid(cscmap)` (a
# CuArray cannot be a hash key, hashing it reads its elements, and an IdDict key is
# held strongly), so the dict references no array and the map's finalizer evicts
# the entry. The finalizer follows the pattern of the manual: it must not block on
# the lock, so when the lock is taken it re-registers itself and retries later.
const _TRANSFER_GROUPING = Dict{UInt, Any}()
const _TRANSFER_GROUPING_LOCK = ReentrantLock()

function _evict_grouping(m)
    if trylock(_TRANSFER_GROUPING_LOCK)
        try
            delete!(_TRANSFER_GROUPING, objectid(m))
        finally
            unlock(_TRANSFER_GROUPING_LOCK)
        end
    else
        finalizer(_evict_grouping, m)
    end
    return
end

function _csc_grouping(cscmap::AbstractVector)
    key = objectid(cscmap)
    lock(_TRANSFER_GROUPING_LOCK)
    try
        cached = get(_TRANSFER_GROUPING, key, nothing)
        cached === nothing || return cached
    finally
        unlock(_TRANSFER_GROUPING_LOCK)
    end
    zvals = CuArray(collect(1:length(cscmap)))
    ptr = map((s, k) -> (s, k), cscmap, zvals)   # (destination CSC slot, COO source)
    length(ptr) > 0 && sort!(ptr)                # group by destination slot
    ptrptr = MadNLP.getptr(ptr; by = (a, b) -> a[1] != b[1])
    grouping = (ptr, ptrptr)
    lock(_TRANSFER_GROUPING_LOCK)
    try
        _TRANSFER_GROUPING[key] = grouping
    finally
        unlock(_TRANSFER_GROUPING_LOCK)
    end
    finalizer(_evict_grouping, cscmap)
    return grouping
end

function MadNLP.transfer!(
        dest::cuSPARSE.CuSparseMatrixCSC,
        src::MadNLP.SparseMatrixCOO,
        cscmap,
    )
    ptr, ptrptr = _csc_grouping(cscmap)
    fill!(dest.nzVal, zero(eltype(dest.nzVal)))
    if length(ptrptr) > 1
        backend = get_backend(dest.nzVal)
        MadNLPGPU._transfer_to_csc_kernel!(backend)(
            dest.nzVal, ptr, ptrptr, src.V;
            ndrange = length(ptrptr) - 1,
        )
        synchronize(backend)
    end
    return dest
end

MadNLP.nzval(H::cuSPARSE.CuSparseMatrixCSC) = H.nzVal

function MadNLP._get_sparse_csc(dims, colptr::CuVector, rowval, nzval)
    return cuSPARSE.CuSparseMatrixCSC(colptr, rowval, nzval, dims)
end

function MadNLP._sym_length(Jt::cuSPARSE.CuSparseMatrixCSC)
    return mapreduce(
        (x, y) -> begin
            z = x - y
            div(z^2 + z, 2)
        end,
        +,
        @view(Jt.colPtr[2:end]),
        @view(Jt.colPtr[1:(end - 1)])
    )
end

function MadNLP.get_tril_to_full(csc::cuSPARSE.CuSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    cscind = MadNLP.SparseMatrixCSC{Int, Ti}(
        Symmetric(
            MadNLP.SparseMatrixCSC{Int, Ti}(
                size(csc)...,
                Array(csc.colPtr),
                Array(csc.rowVal),
                collect(1:MadNLP.nnz(csc)),
            ),
            :L,
        ),
    )
    return cuSPARSE.CuSparseMatrixCSC{Tv, Ti}(
            CuArray(cscind.colptr),
            CuArray(cscind.rowval),
            CuVector{Tv}(undef, MadNLP.nnz(cscind)),
            size(csc),
        ),
        view(csc.nzVal, CuArray(cscind.nzval))
end

