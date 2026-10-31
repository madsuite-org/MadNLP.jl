@inline function count_lu_bounds(lb::AbstractGPUVectorOrSubVector{T}, ub::AbstractGPUVectorOrSubVector{T}) where {T}
    num_lu_bounds = sum((lb .!= -Inf) .& (ub .!=  Inf))
    num_le_bounds = sum((lb .!= -Inf) .& (ub .==  Inf))
    num_ue_bounds = sum((ub .!=  Inf) .& (lb .== -Inf))
    return num_lu_bounds, num_le_bounds, num_ue_bounds
end
