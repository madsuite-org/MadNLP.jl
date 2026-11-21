# Fill-reducing orderings of cuDSS, computed on the host.

"""
    ORDERING

Fill-reducing ordering passed to cuDSS (`cudss_ordering`). `DEFAULT_ORDERING` keeps the
ordering of cuDSS; `METIS_ORDERING` needs Metis loaded, `AMD_ORDERING`,
`SYMAMD_ORDERING` and `COLAMD_ORDERING` need AMD loaded; `USER_ORDERING` takes the
permutation `cudss_perm`.
"""
@enum ORDERING begin
    DEFAULT_ORDERING = 0
    METIS_ORDERING = 1
    AMD_ORDERING = 2
    USER_ORDERING = 3
    SYMAMD_ORDERING = 4
    COLAMD_ORDERING = 5
end

"""
    ordering_permutation(::Val{ordering}, A::SparseMatrixCSC)

Return the fill-reducing permutation `ordering` of the lower triangle `A` of a symmetric
matrix. The methods are in the extensions loaded by Metis (`METIS_ORDERING`) and AMD
(`AMD_ORDERING`, `SYMAMD_ORDERING`, `COLAMD_ORDERING`).
"""
function ordering_permutation(::Val{ordering}, A) where {ordering}
    pkg = ordering == METIS_ORDERING ? "Metis" :
        ordering in (AMD_ORDERING, SYMAMD_ORDERING, COLAMD_ORDERING) ? "AMD" : nothing
    pkg === nothing && throw(ArgumentError("The ordering $ordering has no permutation to compute."))
    return error("Load $pkg to use the ordering $ordering.")
end
