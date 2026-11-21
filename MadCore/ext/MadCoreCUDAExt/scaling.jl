# cuSPARSE matrices in MadCore's scaled factorization
MadCore.scaling_nzval(A::cuSPARSE.CuSparseMatrixCSC) = A.nzVal
MadCore.scaling_index_type(::cuSPARSE.CuSparseMatrixCSC{T, Ti}) where {T, Ti} = Ti
MadCore.scaling_host_pattern(A::cuSPARSE.CuSparseMatrixCSC) = (Vector{Int}(Array(A.colPtr)), Vector{Int}(Array(A.rowVal)))
MadCore.scaling_device(::cuSPARSE.CuSparseMatrixCSC, v::Vector) = CuArray(v)
