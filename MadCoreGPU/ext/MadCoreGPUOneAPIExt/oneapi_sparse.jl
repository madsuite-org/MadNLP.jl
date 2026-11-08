######################################################
##### oneAPI stubs for SparseCondensedKKTSystem  #####
##### (functions requiring concrete oneMKL types) ####
######################################################

function MadCore.transfer!(
        dest::oneMKL.oneSparseMatrixCSC,
        src::MadCore.SparseMatrixCOO,
        map,
    )
    return copyto!(view(dest.nzVal, map), src.V)
end

MadCore.nzval(H::oneMKL.oneSparseMatrixCSC) = H.nzVal

function MadCore._get_sparse_csc(dims, colptr::oneVector, rowval, nzval)
    return oneMKL.oneSparseMatrixCSC(colptr, rowval, nzval, dims)
end

function MadCore._sym_length(Jt::oneMKL.oneSparseMatrixCSC)
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

function MadCore.get_tril_to_full(csc::oneMKL.oneSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
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
    return oneMKL.oneSparseMatrixCSC{Tv, Ti}(
            oneArray(cscind.colptr),
            oneArray(cscind.rowval),
            oneVector{Tv}(undef, MadCore.nnz(cscind)),
            size(csc),
        ),
        view(csc.nzVal, oneArray(cscind.nzval))
end
