# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

abstract type AbstractSparseMatrixCOO{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti} end

mutable struct SparseMatrixCOO{Tv,Ti,VTv<:AbstractVector{Tv},VTi<:AbstractVector{Ti}} <: AbstractSparseMatrixCOO{Tv,Ti}
    m::Int
    n::Int
    I::VTi
    J::VTi
    V::VTv
end
size(A::SparseMatrixCOO) = (A.m,A.n)
getindex(A::SparseMatrixCOO{Tv,Ti},i::Int,j::Int) where {Tv, Ti <: Integer} = sum(A.V[(A.I.==i) .* (A.J.==j)])
nnz(A::SparseMatrixCOO) = length(A.I)

function findIJ(S::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    numnz = nnz(S)
    I = Vector{Ti}(undef,numnz)
    J = Vector{Ti}(undef,numnz)

    cnt = 1
    @inbounds for col = 1 : size(S, 2), k = getcolptr(S)[col] : (getcolptr(S)[col+1]-1)
        I[cnt] = rowvals(S)[k]
        J[cnt] = col
        cnt += 1
    end

    return I,J
end

get_mapping(dest::Matrix{Tv}, src::SparseMatrixCOO{Tv,Ti}) where {Tv,Ti} = nothing

function diag!(dest::AbstractVector{T}, src::AbstractMatrix{T}) where T
    @assert length(dest) == size(src, 1)
    @inbounds for i in eachindex(dest)
        dest[i] = src[i, i]
    end
end
get_tril_to_full(csc::AbstractSparseMatrix{Tv, Ti}) where {Tv, Ti} = get_tril_to_full(Tv, csc)
function get_tril_to_full(T, csc::AbstractSparseMatrix{Tv, Ti}) where {Tv, Ti <: Integer}
    m, n = size(csc)
    cscind = SparseMatrixCSC{Int,Ti}(Symmetric(
            SparseMatrixCSC{Int, Ti}(m, n, host_array(getcolptr(csc)), host_array(rowvals(csc)), collect(1:nnz(csc))), :L
        )
    )
    full = _get_sparse_csc(
        (m, n),
        device_array(csc, cscind.colptr),
        device_array(csc, cscind.rowval),
        similar(nonzeros(csc), T, nnz(cscind)),
    )
    return full, view(nonzeros(csc), device_array(csc, cscind.nzval))
end

# Host and device copies of the index arrays of a sparse matrix. The GPU sparse
# matrices (`GPUArrays.AbstractGPUSparseMatrixCSC`) go through the generic methods.
host_array(v::Vector) = v
host_array(v::AbstractVector) = Array(v)
device_array(::SparseMatrixCSC, v::Vector) = v
device_array(A::AbstractSparseMatrix, v::Vector) = copyto!(similar(nonzeros(A), eltype(v), length(v)), v)
function tril_to_full!(dense::Matrix{T}) where T
    for i=1:size(dense,1)
        Threads.@threads for j=i:size(dense,2)
            @inbounds dense[i,j]=dense[j,i]
        end
    end
end

function coo_to_csc(coo::SparseMatrixCOO)
    cscind = sparse(
        coo.I,
        coo.J,
        fill!(similar(coo.I,nnz(coo)), 1),
        coo.m,
        coo.n
    )
    nzval = similar(coo.V, nnz(cscind))
    fill!(nzval, 0)

    csc = SparseMatrixCSC(
        coo.m,coo.n,cscind.colptr,cscind.rowval,nzval,
    )
    map = get_mapping(csc, coo)

    return csc, map
end

function _get_coo_to_csc(I,J,cscind,map)
    for i=1:length(I)
        @inbounds map[i] = cscind[I[i],J[i]]
    end
end
function _transfer!(vec1, vec2, map)
    T = eltype(vec1)
    fill!(vec1, zero(T))
    for i=1:length(map)
        @inbounds vec1[map[i]] += vec2[i]
    end
end

function transfer!(dest::AbstractSparseMatrix, src::SparseMatrixCOO, map)
    _transfer!(nonzeros(dest), src.V, map)
    return dest
end

function get_mapping(dest::SparseMatrixCSC{Tv1,Ti1}, src::SparseMatrixCOO{Tv2,Ti2}) where {Tv1,Tv2,Ti1,Ti2}
    map = similar(src.V, Int, nnz(src))
    dest.nzval .= 1:nnz(dest)
    _get_coo_to_csc(src.I, src.J, dest, map)
    return map
end

function Matrix{Tv}(coo::SparseMatrixCOO{Tv,Ti}) where {Tv,Ti<:Integer}
    return Matrix{Tv}(undef,coo.m,coo.n)
end

Base.copyto!(dense::Matrix,coo::SparseMatrixCOO) = _copyto!(dense,coo.I,coo.J,coo.V)
function _copyto!(dense::Matrix{Tv},I,J,V) where Tv
    fill!(dense, zero(Tv))
    for i=1:length(I)
        @inbounds dense[I[i], J[i]] += V[i]
    end
    return dense
end

function get_cscsy_view(csc::SparseMatrixCSC{Tv,Ti},Ix;inds=collect(1:nnz(csc))) where {Tv,Ti<:Integer}
    cscind = SparseMatrixCSC{Int,Ti}(csc.m,csc.n,csc.colptr,csc.rowval,inds)
    cscindsub = cscind[Ix,Ix]
    return SparseMatrixCSC{Tv,Ti}(
        cscindsub.m,cscindsub.n,cscindsub.colptr,
        cscindsub.rowval,Vector{Tv}(undef,nnz(cscindsub))), view(csc.nzval,cscindsub.nzval)

end

function get_csc_view(csc::SparseMatrixCSC{Tv,Ti},Ix,Jx;inds=collect(1:nnz(csc))) where {Tv,Ti<:Integer}
    cscind = Symmetric(SparseMatrixCSC{Int,Ti}(csc.m,csc.n,csc.colptr,csc.rowval,inds),:L)
    cscindsub = cscind[Ix,Jx]
    resize!(cscindsub.rowval,cscindsub.colptr[end]-1)
    resize!(cscindsub.nzval,cscindsub.colptr[end]-1)
    return SparseMatrixCSC{Tv,Ti}(
        cscindsub.m,cscindsub.n,cscindsub.colptr,
        cscindsub.rowval,Vector{Tv}(undef,nnz(cscindsub))), view(csc.nzval,cscindsub.nzval)
end

function force_lower_triangular!(I,J)
    @simd for i=1:length(I)
        @inbounds if J[i] > I[i]
            tmp=J[i]
            J[i]=I[i]
            I[i]=tmp
        end
    end
end

