######################################################
##### ROCm stubs for SparseCondensedKKTSystem    #####
##### (functions requiring concrete rocSPARSE types) #
######################################################

function MadCore.transfer!(
    dest::rocSPARSE.ROCSparseMatrixCSC,
    src::MadCore.SparseMatrixCOO,
    map,
)
    return copyto!(view(dest.nzVal, map), src.V)
end

MadCore.nzval(H::rocSPARSE.ROCSparseMatrixCSC) = H.nzVal

function MadCore._get_sparse_csc(dims, colptr::ROCVector, rowval, nzval)
    return rocSPARSE.ROCSparseMatrixCSC(colptr, rowval, nzval, dims)
end

function MadCore._sym_length(Jt::rocSPARSE.ROCSparseMatrixCSC)
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

function MadCore.get_tril_to_full(csc::rocSPARSE.ROCSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    cscind = MadCore.SparseMatrixCSC{Int,Ti}(
        Symmetric(
            MadCore.SparseMatrixCSC{Int,Ti}(
                size(csc)...,
                Array(csc.colPtr),
                Array(csc.rowVal),
                collect(1:MadCore.nnz(csc)),
            ),
            :L,
        ),
    )
    return rocSPARSE.ROCSparseMatrixCSC{Tv,Ti}(
        ROCArray(cscind.colptr),
        ROCArray(cscind.rowval),
        ROCVector{Tv}(undef, MadCore.nnz(cscind)),
        size(csc),
    ),
    view(csc.nzVal, ROCArray(cscind.nzval))
end

