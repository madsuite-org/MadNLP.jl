# rocSPARSE matrices in MadCore's scaled factorization
MadCore.scaling_nzval(A::rocSPARSE.ROCSparseMatrixCSC) = A.nzVal
MadCore.scaling_index_type(::rocSPARSE.ROCSparseMatrixCSC{T, Ti}) where {T, Ti} = Ti
MadCore.scaling_host_pattern(A::rocSPARSE.ROCSparseMatrixCSC) = (Vector{Int}(Array(A.colPtr)), Vector{Int}(Array(A.rowVal)))
MadCore.scaling_device(::rocSPARSE.ROCSparseMatrixCSC, v::Vector) = ROCArray(v)
