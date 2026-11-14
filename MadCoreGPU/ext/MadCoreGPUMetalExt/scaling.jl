# Metal sparse matrices in MadCore's scaled factorization
MadCore.scaling_nzval(A::MtlSparseMatrixCSC) = A.nzVal
MadCore.scaling_index_type(::MtlSparseMatrixCSC{T, Ti}) where {T, Ti} = Ti
MadCore.scaling_host_pattern(A::MtlSparseMatrixCSC) = (Vector{Int}(Array(A.colPtr)), Vector{Int}(Array(A.rowVal)))
MadCore.scaling_device(::MtlSparseMatrixCSC, v::Vector) = MtlArray(v)
