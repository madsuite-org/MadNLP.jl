# The sparse matrix of this backend, built by the sparse KKT systems of MadCore.
function MadCore._get_sparse_csc(dims, colptr::ROCVector, rowval, nzval)
    return rocSPARSE.ROCSparseMatrixCSC(colptr, rowval, nzval, dims)
end
