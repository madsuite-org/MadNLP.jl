# Scaling sweep of MadCore's scaled factorization on a GPU: one work item per column of the
# pattern with both triangles.
@kernel function _scaling_sweep_kernel!(dnew, @Const(d), @Const(colptr), @Const(rowval), @Const(absnz))
    j = @index(Global)
    r = zero(eltype(d))
    for k in colptr[j]:(colptr[j + 1] - 1)
        r = max(r, absnz[k] * d[rowval[k]])
    end
    r *= d[j]
    dnew[j] = r > 0 ? d[j] / sqrt(r) : d[j]
end

function MadCore.scaling_sweep!(dnew::AbstractGPUVector, d, colptr, rowval, absnz)
    _scaling_sweep_kernel!(get_backend(dnew))(dnew, d, colptr, rowval, absnz; ndrange = length(dnew))
    return dnew
end
