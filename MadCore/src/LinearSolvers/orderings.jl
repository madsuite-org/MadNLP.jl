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
    ordering_permutation(::Val{ordering}, A, perm)

Return the permutation that `ordering` passes to the solver for the lower triangle `A` of a
symmetric matrix, or `nothing` to keep the ordering of the solver. `perm` is the
permutation given by the user (`cudss_perm`), used by `USER_ORDERING`. The fill-reducing
orderings are computed on the host, in the extensions loaded by Metis (`METIS_ORDERING`)
and AMD (`AMD_ORDERING`, `SYMAMD_ORDERING`, `COLAMD_ORDERING`).
"""
ordering_permutation(::Val{DEFAULT_ORDERING}, A, perm) = nothing
function ordering_permutation(::Val{USER_ORDERING}, A, perm)
    (isempty(perm) || !isperm(perm)) && throw(ArgumentError("USER_ORDERING needs a valid permutation in cudss_perm."))
    return perm
end
ordering_permutation(o::Val, A, perm) = error("Load $(ordering_package(o)) to use the ordering $(_ordering(o)).")

ordering_package(::Val{METIS_ORDERING}) = "Metis"
ordering_package(::Union{Val{AMD_ORDERING}, Val{SYMAMD_ORDERING}, Val{COLAMD_ORDERING}}) = "AMD"
_ordering(::Val{o}) where {o} = o
