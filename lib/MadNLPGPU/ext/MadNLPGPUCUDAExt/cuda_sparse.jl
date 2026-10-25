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
# the grouping, so it is memoized here, keyed by the (fixed) csc map object.
# Keyed by object identity (objectid): a CuArray cannot be a hash-keyed dict key
# because hashing it reads its elements (scalar indexing). The grouping depends
# only on the fixed sparsity pattern, so one entry per csc map lives for the
# solver's lifetime.
const _TRANSFER_GROUPING = Base.IdDict{Any, Any}()

function _csc_grouping(cscmap::AbstractVector)
    return get!(_TRANSFER_GROUPING, cscmap) do
        zvals = CuArray(collect(1:length(cscmap)))
        ptr = map((s, k) -> (s, k), cscmap, zvals)   # (destination CSC slot, COO source)
        length(ptr) > 0 && sort!(ptr)                # group by destination slot
        ptrptr = MadNLP.getptr(ptr; by = (a, b) -> a[1] != b[1])
        # Evict this entry when the (fixed, per-KKT) map object is collected, so the
        # cached ptr/ptrptr GPU arrays don't outlive their KKT. Without this the cache
        # pins every solved case's device memory (CUDA.reclaim can't free IdDict-held
        # arrays), inflating later cases' init by accumulating GPU memory.
        finalizer(m -> delete!(_TRANSFER_GROUPING, m), cscmap)
        return (ptr, ptrptr)
    end
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

