######################################################
##### CUDA stubs for SparseCondensedKKTSystem    #####
##### (functions requiring concrete CUSPARSE types) ##
######################################################

function MadCore.transfer!(
        dest::cuSPARSE.CuSparseMatrixCSC,
        src::MadCore.SparseMatrixCOO,
        cscmap,
    )
    MadCoreGPU._transfer_coo_to_csc!(dest.nzVal, src.V, cscmap)
    return dest
end

MadCore.nzval(H::cuSPARSE.CuSparseMatrixCSC) = H.nzVal

function MadCore._get_sparse_csc(dims, colptr::CuVector, rowval, nzval)
    return cuSPARSE.CuSparseMatrixCSC(colptr, rowval, nzval, dims)
end

function MadCore._sym_length(Jt::cuSPARSE.CuSparseMatrixCSC)
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

function MadCore.get_tril_to_full(csc::cuSPARSE.CuSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    cscind = MadCore.SparseMatrixCSC{Int, Ti}(
        Symmetric(
            MadCore.SparseMatrixCSC{Int, Ti}(
                size(csc)...,
                Array(csc.colPtr),
                Array(csc.rowVal),
                collect(1:MadCore.nnz(csc)),
            ),
            :L,
        ),
    )
    return cuSPARSE.CuSparseMatrixCSC{Tv, Ti}(
            CuArray(cscind.colptr),
            CuArray(cscind.rowval),
            CuVector{Tv}(undef, MadCore.nnz(cscind)),
            size(csc),
        ),
        view(csc.nzVal, CuArray(cscind.nzval))
end
