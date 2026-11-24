# Scaling sweep of MadCore's scaled factorization on a GPU. Each column of the pattern with
# both triangles is shared by `lanes` work items, so that neighboring work items read
# neighboring nonzeros, and the maximum over a column is reduced in local memory. A column
# with more than `MadCore.SCALING_LONG_COLUMN` nonzeros (in `long_cols`) gets a work group of
# its own instead, so that it does not hold up the rest.
const SCALING_SWEEP_GROUP = 256

@inline function _scaling_update(d, j, r)
    r *= d[j]
    return r > 0 ? d[j] / sqrt(r) : d[j]
end

@kernel function _scaling_sweep_kernel!(dnew, @Const(d), @Const(colptr), @Const(rowval), @Const(absnz), lanes, long)
    i = @index(Local, Linear)
    g = @index(Group, Linear)
    rmax = @localmem eltype(d) (SCALING_SWEEP_GROUP,)
    lane = (i - 1) % lanes + 1
    j = (g - 1) * (SCALING_SWEEP_GROUP ÷ lanes) + (i - 1) ÷ lanes + 1
    mine = j <= length(dnew) && colptr[j + 1] - colptr[j] <= long
    r = zero(eltype(d))
    if mine
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
    if lane == 1 && mine
        dnew[j] = _scaling_update(d, j, rmax[i])
    end
end

@kernel function _scaling_sweep_long_kernel!(dnew, @Const(d), @Const(colptr), @Const(rowval), @Const(absnz), @Const(long_cols))
    i = @index(Local, Linear)
    g = @index(Group, Linear)
    rmax = @localmem eltype(d) (SCALING_SWEEP_GROUP,)
    j = long_cols[g]
    r = zero(eltype(d))
    k = colptr[j] + i - 1
    while k < colptr[j + 1]
        r = max(r, absnz[k] * d[rowval[k]])
        k += SCALING_SWEEP_GROUP
    end
    rmax[i] = r
    @synchronize
    s = SCALING_SWEEP_GROUP ÷ 2
    while s > 0
        if i <= s
            rmax[i] = max(rmax[i], rmax[i + s])
        end
        @synchronize
        s ÷= 2
    end
    if i == 1
        dnew[j] = _scaling_update(d, j, rmax[1])
    end
end

function MadCore.scaling_sweep!(dnew::AbstractGPUVector, d, colptr, rowval, absnz, long_cols)
    n = length(dnew)
    backend = get_backend(dnew)
    # about one lane per nonzero of an average column, as a power of two
    lanes = clamp(prevpow(2, max(1, length(absnz) ÷ max(1, n))), 1, 32)
    _scaling_sweep_kernel!(backend, SCALING_SWEEP_GROUP)(
        dnew, d, colptr, rowval, absnz, lanes, MadCore.SCALING_LONG_COLUMN;
        ndrange = cld(n, SCALING_SWEEP_GROUP ÷ lanes) * SCALING_SWEEP_GROUP,
    )
    isempty(long_cols) || _scaling_sweep_long_kernel!(backend, SCALING_SWEEP_GROUP)(
        dnew, d, colptr, rowval, absnz, long_cols;
        ndrange = length(long_cols) * SCALING_SWEEP_GROUP,
    )
    return dnew
end
