# Scaling sweep of MadCore's scaled factorization on a GPU. Each column of the pattern with
# both triangles is shared by `lanes` work items, so that neighboring work items read
# neighboring nonzeros; the maximum over a column is reduced in local memory.
const SCALING_SWEEP_GROUP = 256

@kernel function _scaling_sweep_kernel!(dnew, @Const(d), @Const(colptr), @Const(rowval), @Const(absnz), lanes)
    i = @index(Local, Linear)
    g = @index(Group, Linear)
    rmax = @localmem eltype(d) (SCALING_SWEEP_GROUP,)
    lane = (i - 1) % lanes + 1
    j = (g - 1) * (SCALING_SWEEP_GROUP ÷ lanes) + (i - 1) ÷ lanes + 1
    r = zero(eltype(d))
    if j <= length(dnew)
        k = colptr[j] + lane - 1
        while k < colptr[j + 1]
            r = max(r, absnz[k] * d[rowval[k]])
            k += lanes
        end
    end
    rmax[i] = r
    @synchronize
    s = lanes ÷ 2
    while s > 0
        if lane <= s
            rmax[i] = max(rmax[i], rmax[i + s])
        end
        @synchronize
        s ÷= 2
    end
    if lane == 1 && j <= length(dnew)
        r = rmax[i] * d[j]
        dnew[j] = r > 0 ? d[j] / sqrt(r) : d[j]
    end
end

function MadCore.scaling_sweep!(dnew::AbstractGPUVector, d, colptr, rowval, absnz)
    n = length(dnew)
    # about one lane per nonzero of an average column, as a power of two
    lanes = clamp(prevpow(2, max(1, length(absnz) ÷ max(1, n))), 1, 32)
    _scaling_sweep_kernel!(get_backend(dnew), SCALING_SWEEP_GROUP)(
        dnew, d, colptr, rowval, absnz, lanes;
        ndrange = cld(n, SCALING_SWEEP_GROUP ÷ lanes) * SCALING_SWEEP_GROUP,
    )
    return dnew
end
