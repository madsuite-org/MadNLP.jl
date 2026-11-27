# cuDSS helper functions
export CudssMatrix, CudssBatchedMatrix, CudssData, CudssConfig

## Matrix

"""
    matrix = CudssMatrix(::Type{T}, n::Integer; nbatch::Integer=1)
    matrix = CudssMatrix(::Type{T}, m::Integer, n::Integer; nbatch::Integer=1)
    matrix = CudssMatrix(b::CuVector{T})
    matrix = CudssMatrix(B::CuMatrix{T})
    matrix = CudssMatrix(A::CuSparseMatrixCSR{T,INT}, struture::String, view::Char; index::Char='O')
    matrix = CudssMatrix(rowPtr::CuVector{INT}, colVal::CuVector{INT}, nzVal::CuVector{T}, struture::String, view::Char; index::Char='O')
    matrix = CudssMatrix(rowPtr::CuVector{INT}, colVal::CuVector{INT}, nzVal::CuMatrix{T}, struture::String, view::Char; index::Char='O')

The parameter type `T` is restricted to `Float32`, `Float64`, `ComplexF32`, or `ComplexF64`, while `INT` is restricted to `Int32` or `Int64`.

`CudssMatrix` is a wrapper for `CuVector`, `CuMatrix` and `CuSparseMatrixCSR`.
`CudssMatrix` can also represent a batch of `CuSparseMatrixCSR` sharing the same sparsity pattern.
`CudssMatrix` is used to pass the matrix of the sparse linear system, as well as solution and right-hand side.

`structure` specifies the stucture for the sparse matrix:
- `"G"`: General matrix -- LDU factorization;
- `"S"`: Real symmetric matrix -- LDLᵀ factorization;
- `"H"`: Complex Hermitian matrix -- LDLᴴ factorization;
- `"SPD"`: Symmetric positive-definite matrix -- LLᵀ factorization;
- `"HPD"`: Hermitian positive-definite matrix -- LLᴴ factorization.

`view` specifies matrix view for the sparse matrix:
- `'L'`: Lower-triangular matrix and all values above the main diagonal are ignored;
- `'U'`: Upper-triangular matrix and all values below the main diagonal are ignored;
- `'F'`: Full matrix.

`index` specifies indexing base for the sparse matrix:
- `'Z'`: 0-based indexing;
- `'O'`: 1-based indexing.
"""
mutable struct CudssMatrix{T,INT} <: AbstractCudssMatrix{T,INT}
    value_type::Type{T}
    index_type::Type{INT}
    matrix::cudssMatrix_t
    nbatch::Int64
    nrows::Int64
    ncols::Int64
    nz::Int64
    # GC roots for the Julia arrays whose device pointers were handed to cuDSS.
    # cuDSS only stores the raw pointers, so without keeping the arrays alive here
    # they can be collected while the descriptor (and any pending async phase) still
    # references them -- segfaults / EXECUTION_FAILED that masquerade as library bugs.
    # `refs` is refreshed by `cudss_update` whenever the underlying pointers change.
    refs::Any

    function CudssMatrix(::Type{T}, n::Integer; nbatch::Integer=1) where T <: BlasFloat
        nz = n * nbatch
        matrix_ref = Ref{cudssMatrix_t}()
        cudssMatrixCreateDn(matrix_ref, n, 1, n, CU_NULL, T, 'C')
        obj = new{T,Cint}(T, Cint, matrix_ref[], nbatch, n, 1, nz, nothing)
        finalizer(cudssMatrixDestroy, obj)
        obj
    end

    function CudssMatrix(::Type{T}, m::Integer, n::Integer; nbatch::Integer=1, transposed::Bool=false) where T <: BlasFloat
        nz = n * m * nbatch
        matrix_ref = Ref{cudssMatrix_t}()
        if transposed
            cudssMatrixCreateDn(matrix_ref, n, m, m, CU_NULL, T, 'R')
            obj = new{T,Cint}(T, Cint, matrix_ref[], nbatch, n, m, nz, nothing)
        else
            cudssMatrixCreateDn(matrix_ref, m, n, m, CU_NULL, T, 'C')
            obj = new{T,Cint}(T, Cint, matrix_ref[], nbatch, m, n, nz, nothing)
        end
        finalizer(cudssMatrixDestroy, obj)
        obj
    end

    function CudssMatrix(b::CuVector{T}) where T <: BlasFloat
        m = length(b)
        matrix_ref = Ref{cudssMatrix_t}()
        cudssMatrixCreateDn(matrix_ref, m, 1, m, b, T, 'C')
        obj = new{T,Cint}(T, Cint, matrix_ref[], 1, m, 1, m, b)
        finalizer(cudssMatrixDestroy, obj)
        obj
    end

    function CudssMatrix(B::CuMatrix{T}; transposed::Bool=false) where T <: BlasFloat
        m,n = size(B)
        nz = m * n
        matrix_ref = Ref{cudssMatrix_t}()
        if transposed
            cudssMatrixCreateDn(matrix_ref, n, m, m, B, T, 'R')
            obj = new{T,Cint}(T, Cint, matrix_ref[], 1, n, m, nz, B)
        else
            cudssMatrixCreateDn(matrix_ref, m, n, m, B, T, 'C')
            obj = new{T,Cint}(T, Cint, matrix_ref[], 1, m, n, nz, B)
        end
        finalizer(cudssMatrixDestroy, obj)
        obj
    end

    function CudssMatrix(rowPtr::CuVector{INT}, colVal::CuVector{INT}, nzVal::CuVector{T}, structure::String, view::Char; index::Char='O') where {T <: BlasFloat, INT <: CudssInt}
        n = length(rowPtr) - 1
        nz_batch = length(colVal)
        nz_total = length(nzVal)
        nbatch = nz_total ÷ nz_batch
        matrix_ref = Ref{cudssMatrix_t}()
        cudssMatrixCreateCsr(matrix_ref, n, n, nz_batch, rowPtr, CU_NULL,
                             colVal, nzVal, INT, INT, T, structure,
                             view, index)
        obj = new{T,INT}(T, INT, matrix_ref[], nbatch, n, n, nz_total, (rowPtr, colVal, nzVal))
        finalizer(cudssMatrixDestroy, obj)
        obj
    end

    function CudssMatrix(rowPtr::CuVector{INT}, colVal::CuVector{INT}, nzVal::CuMatrix{T}, structure::String, view::Char; index::Char='O') where {T <: BlasFloat, INT <: CudssInt}
        n = length(rowPtr) - 1
        nz_batch, nbatch = size(nzVal)
        matrix_ref = Ref{cudssMatrix_t}()
        cudssMatrixCreateCsr(matrix_ref, n, n, nz_batch, rowPtr, CU_NULL,
                             colVal, nzVal, INT, INT, T, structure,
                             view, index)
        obj = new{T,INT}(T, INT, matrix_ref[], nbatch, n, n, nz_batch * nbatch, (rowPtr, colVal, nzVal))
        finalizer(cudssMatrixDestroy, obj)
        obj
    end

   function CudssMatrix(A::CuSparseMatrixCSR{T,INT}, structure::String, view::Char; index::Char='O') where {T <: BlasFloat, INT <: CudssInt}
        CudssMatrix(A.rowPtr, A.colVal, A.nzVal, structure, view; index)
    end
end

Base.unsafe_convert(::Type{cudssMatrix_t}, matrix::CudssMatrix) = matrix.matrix

"""
    matrix = CudssBatchedMatrix(b::Vector{CuVector{T}})
    matrix = CudssBatchedMatrix(B::Vector{CuMatrix{T}})
    matrix = CudssBatchedMatrix(A::Vector{CuSparseMatrixCSR{T,INT}}, struture::String, view::Char; index::Char='O')

The parameter type `T` is restricted to `Float32`, `Float64`, `ComplexF32`, or `ComplexF64`, while `INT` is restricted to `Int32` or `Int64`.

`CudssBatchedMatrix` is a wrapper for `Vector{CuVector}`, `Vector{CuMatrix}` and `Vector{CuSparseMatrixCSR}`.
`CudssBatchedMatrix` is used to pass the matrices of the sparse linear systems, as well as solutions and right-hand sides.

`structure` specifies the stucture for the sparse matrices:
- `"G"`: General matrix -- LDU factorization;
- `"S"`: Real symmetric matrix -- LDLᵀ factorization;
- `"H"`: Complex Hermitian matrix -- LDLᴴ factorization;
- `"SPD"`: Symmetric positive-definite matrix -- LLᵀ factorization;
- `"HPD"`: Hermitian positive-definite matrix -- LLᴴ factorization.

`view` specifies matrix view for the sparse matrices:
- `'L'`: Lower-triangular matrix and all values above the main diagonal are ignored;
- `'U'`: Upper-triangular matrix and all values below the main diagonal are ignored;
- `'F'`: Full matrix.

`index` specifies indexing base for the sparse matrices:
- `'Z'`: 0-based indexing;
- `'O'`: 1-based indexing.
"""
mutable struct CudssBatchedMatrix{T,INT,M} <: AbstractCudssMatrix{T,INT}
    value_type::Type{T}
    index_type::Type{INT}
    matrix::cudssMatrix_t
    nbatch::Int64
    nrows::Vector{INT}
    ncols::Vector{INT}
    nnzA::Vector{INT}
    Mptrs::M
    # GC roots for the per-batch source arrays. `Mptrs` is a device array of raw
    # pointers into them, so the batch (and its element arrays) must be kept alive
    # here for as long as this descriptor is; see the note on `CudssMatrix.refs`.
    refs::Any

    function CudssBatchedMatrix(b::Vector{<:CuVector{T}}) where T <: BlasFloat
        matrix_ref = Ref{cudssMatrix_t}()
        nbatch = length(b)
        nrows = Cint[length(bᵢ) for bᵢ in b]
        ncols = Cint[1 for i = 1:nbatch]
        ld = nrows
        Mptrs = unsafe_cudss_batch(b)
        M = typeof(Mptrs)
        cudssMatrixCreateBatchDn(matrix_ref, nbatch, nrows, ncols, ld, Mptrs, Cint, T, 'C')
        obj = new{T,Cint,M}(T, Cint, matrix_ref[], nbatch, nrows, ncols, Cint[], Mptrs, b)
        finalizer(cudssBatchedMatrixDestroy, obj)
        obj
    end

    function CudssBatchedMatrix(B::Vector{<:CuMatrix{T}}; transposed::Bool=false) where T <: BlasFloat
        matrix_ref = Ref{cudssMatrix_t}()
        nbatch = length(B)
        nrows = Cint[size(Bᵢ,1) for Bᵢ in B]
        ncols = Cint[size(Bᵢ,2) for Bᵢ in B]
        ld = nrows
        Mptrs = unsafe_cudss_batch(B)
        M = typeof(Mptrs)
        if transposed
            cudssMatrixCreateBatchDn(matrix_ref, nbatch, ncols, nrows, ld, Mptrs, Cint, T, 'R')
        else
            cudssMatrixCreateBatchDn(matrix_ref, nbatch, nrows, ncols, ld, Mptrs, Cint, T, 'C')
        end
        obj = new{T,Cint,M}(T, Cint, matrix_ref[], nbatch, nrows, ncols, Cint[], Mptrs, B)
        finalizer(cudssBatchedMatrixDestroy, obj)
        obj
    end

    function CudssBatchedMatrix(A::Vector{CuSparseMatrixCSR{T,INT}}, structure::String, view::Char; index::Char='O') where {T <: BlasFloat, INT <: CudssInt}
        matrix_ref = Ref{cudssMatrix_t}()
        nbatch = length(A)
        nrows = INT[size(Aᵢ,1) for Aᵢ in A]
        ncols = INT[size(Aᵢ,2) for Aᵢ in A]
        nnzA = INT[nnz(Aᵢ) for Aᵢ in A]
        rowPtrs, colVals, nzVals = unsafe_cudss_batch(A)
        cudssMatrixCreateBatchCsr(matrix_ref, nbatch, nrows, ncols, nnzA, rowPtrs,
                                  CUPTR_C_NULL, colVals, nzVals, INT, INT, T, structure,
                                  view, index)
        Mptrs = (rowPtrs, colVals, nzVals)
        M = typeof(Mptrs)
        obj = new{T,INT,M}(T, INT, matrix_ref[], nbatch, nrows, ncols, nnzA, Mptrs, A)
        finalizer(cudssBatchedMatrixDestroy, obj)
        obj
    end
end

function cudssBatchedMatrixDestroy(matrix::CudssBatchedMatrix)
    cudssMatrixDestroy(matrix.matrix)
    if matrix.Mptrs isa Tuple
        # sparse matrix
        unsafe_free!(matrix.Mptrs[1])  # rowPtrs
        unsafe_free!(matrix.Mptrs[2])  # colVals
        unsafe_free!(matrix.Mptrs[3])  # nzVals
    else
        # dense vector or matrix
        unsafe_free!(matrix.Mptrs)
    end
end

Base.unsafe_convert(::Type{cudssMatrix_t}, matrix::CudssBatchedMatrix) = matrix.matrix

## Data

"""
    data = CudssData()
    data = CudssData(cudss_handle::cudssHandle_t)

`CudssData` holds internal data (e.g., LU factors arrays).
"""
mutable struct CudssData
    handle::cudssHandle_t
    data::cudssData_t

    function CudssData(cudss_handle::cudssHandle_t)
        data_ref = Ref{cudssData_t}()
        cudssDataCreate(cudss_handle, data_ref)
        obj = new(cudss_handle, data_ref[])
        finalizer(cudssDataDestroy, obj)
        obj
    end

    function CudssData()
        cudss_handle = handle()
        CudssData(cudss_handle)
    end
end

Base.unsafe_convert(::Type{cudssData_t}, data::CudssData) = data.data

function cudssDataDestroy(data::CudssData)
    cudssDataDestroy(data.handle, data.data)
end

## Configuration

"""
    config = CudssConfig()

`CudssConfig` stores configuration settings for the solver.
"""
mutable struct CudssConfig
    config::cudssConfig_t

    function CudssConfig()
        config_ref = Ref{cudssConfig_t}()
        cudssConfigCreate(config_ref)
        obj = new(config_ref[])
        finalizer(cudssConfigDestroy, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cudssConfig_t}, config::CudssConfig) = config.config
