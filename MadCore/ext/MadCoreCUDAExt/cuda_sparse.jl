# The sparse matrix of this backend, built by the sparse KKT systems of MadCore.
function MadCore._get_sparse_csc(dims, colptr::CuVector, rowval, nzval)
    return cuSPARSE.CuSparseMatrixCSC(colptr, rowval, nzval, dims)
end
