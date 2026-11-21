# The sparse matrix of this backend, built by the sparse KKT systems of MadCore.
function MadCore._get_sparse_csc(dims, colptr::MtlVector, rowval, nzval)
    return MtlSparseMatrixCSC(colptr, rowval, nzval, dims)
end
