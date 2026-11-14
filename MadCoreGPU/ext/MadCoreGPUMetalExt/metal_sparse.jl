######################################################
##### Metal stubs for SparseCondensedKKTSystem   #####
######################################################

function MadCore.transfer!(dest::MtlSparseMatrixCSC, src::MadCore.SparseMatrixCOO, map)
    MadCoreGPU._transfer_coo_to_csc!(dest.nzVal, src.V, map)
    return dest
end

MadCore.nzval(H::MtlSparseMatrixCSC) = H.nzVal

function MadCore._get_sparse_csc(dims, colptr::MtlVector, rowval, nzval)
    return MtlSparseMatrixCSC(colptr, rowval, nzval, dims)
end

function MadCore._sym_length(Jt::MtlSparseMatrixCSC)
    colptr = Array(Jt.colPtr)
    return sum(((colptr[j + 1] - colptr[j])^2 + (colptr[j + 1] - colptr[j])) ÷ 2 for j in 1:(length(colptr) - 1); init = 0)
end

function MadCore.get_tril_to_full(csc::MtlSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    cscind = SparseMatrixCSC{Int, Ti}(
        Symmetric(
            SparseMatrixCSC{Int, Ti}(
                size(csc)...,
                Array(csc.colPtr),
                Array(csc.rowVal),
                collect(1:SparseArrays.nnz(csc)),
            ),
            :L,
        ),
    )
    return MtlSparseMatrixCSC(cscind.colptr, cscind.rowval, MtlVector{Tv}(undef, SparseArrays.nnz(cscind)), size(csc)),
        view(csc.nzVal, MtlArray(cscind.nzval))
end
