# oneMKL sparse matrices in MadCore's scaled factorization
MadCore.scaling_nzval(A::oneMKL.oneSparseMatrixCSC) = A.nzVal
MadCore.scaling_index_type(::oneMKL.oneSparseMatrixCSC{T, Ti}) where {T, Ti} = Ti
MadCore.scaling_host_pattern(A::oneMKL.oneSparseMatrixCSC) = (Vector{Int}(Array(A.colPtr)), Vector{Int}(Array(A.rowVal)))
MadCore.scaling_device(::oneMKL.oneSparseMatrixCSC, v::Vector) = oneArray(v)
