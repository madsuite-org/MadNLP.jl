####################################################
##### GPU kernels for SparseCondensedKKTSystem #####
####################################################

#=
    _copy_to_map_kernel!
=#

@kernel function _copy_to_map_kernel!(y, p, x)
    i = @index(Global)
    @inbounds y[p[i]] = x[i]
end

#=
    _copy_from_map_kernel!
=#

@kernel function _copy_from_map_kernel!(y, x, p)
    i = @index(Global)
    @inbounds y[i] = x[p[i]]
end

#=
    _csc_to_dense_kernel!
=#

@kernel function _csc_to_dense_kernel!(y, @Const(colptr), @Const(rowval), @Const(nzval))
    col = @index(Global)
    @inbounds for ptr in colptr[col]:colptr[col+1]-1
        row = rowval[ptr]
        y[row, col] = nzval[ptr]
    end
end

#=
    MadNLP._set_colptr!
=#

@kernel function _set_colptr_kernel!(colptr, @Const(sym2), @Const(ptr2), @Const(guide))
    idx = @index(Global)
    @inbounds begin
        i = ptr2[idx+1]

        (~, prevcol) = sym2[i-1]
        (row, col) = sym2[i]
        g = guide[i]
        for j in prevcol+1:col
            colptr[j] = g
        end
    end
end

#=
    MadNLP.tril_to_full!
=#

@kernel function _tril_to_full_kernel!(dense)
    idx = @index(Global)
    n = size(dense, 1)
    i, j = getij(idx, n)
    @inbounds dense[j, i] = dense[i, j]
end

#=
    MadNLP.force_lower_triangular!
=#

@kernel function _force_lower_triangular_kernel!(I, J)
    i = @index(Global)

    @inbounds if J[i] > I[i]
        tmp = J[i]
        J[i] = I[i]
        I[i] = tmp
    end
end

#=
    MadNLP.coo_to_csc
=#

@kernel function _set_coo_to_colptr_kernel!(colptr, @Const(coord))
    index = @index(Global)

    @inbounds begin
        if index == 1
            ((i2, j2), k2) = coord[index]
            for k in 1:j2
                colptr[k] = 1
            end
            if index == length(coord)
                ip1 = index + 1
                for k in j2+1:length(colptr)
                    colptr[k] = ip1
                end
            end
        else
            ((i1, j1), k1) = coord[index-1]
            ((i2, j2), k2) = coord[index]
            if j1 != j2
                for k in j1+1:j2
                    colptr[k] = index
                end
            end
            if index == length(coord)
                ip1 = index + 1
                for k in j2+1:length(colptr)
                    colptr[k] = ip1
                end
            end
        end
    end
end

@kernel function _set_coo_to_csc_map_kernel!(cscmap, @Const(mapptr), @Const(coord))
    index = @index(Global)
    @inbounds for l in mapptr[index]:mapptr[index+1]-1
        ((i, j), k) = coord[l]
        cscmap[k] = index
    end
end

#=
    MadNLP.build_condensed_aug_coord!
=#

@kernel function _transfer_hessian_kernel!(y, @Const(ptr), @Const(x))
    index = @index(Global)
    @inbounds i, j = ptr[index]
    @inbounds y[i] += x[j]
end

@kernel function _transfer_jtsj_kernel!(y, @Const(ptr), @Const(ptrptr), @Const(x), @Const(s))
    index = @index(Global)
    @inbounds for index2 in ptrptr[index]:ptrptr[index+1]-1
        i, (j, k, l) = ptr[index2]
        y[i] += s[j] * x[k] * x[l]
    end
end

@kernel function _diag_operation_kernel!(
    y,
    @Const(A),
    @Const(x),
    @Const(alpha),
    @Const(idx_to),
    @Const(idx_fr)
)
    i = @index(Global)
    @inbounds begin
        to = idx_to[i]
        fr = idx_fr[i]
        y[to] -= alpha * A[fr] * x[to]
    end
end

# Subtract the double-counted diagonal of a lower-triangular CSC after a
# symmetric product computed as L*x + Lᵀ*x. For column j the diagonal entry is
# the first stored entry (row == j) when present.
@kernel function _sub_tril_diag_kernel!(
    wx,
    @Const(colptr),
    @Const(rowval),
    @Const(nzval),
    @Const(x),
    @Const(alpha),
)
    j = @index(Global)
    @inbounds begin
        for p in colptr[j]:(colptr[j + 1] - 1)
            if rowval[p] == j
                wx[j] -= alpha * nzval[p] * x[j]
                break
            end
        end
    end
end

#=
    MadNLP.compress_hessian! / MadNLP.compress_jacobian!
=#

@kernel function _transfer_to_csc_kernel!(y, @Const(ptr), @Const(ptrptr), @Const(x))
    index = @index(Global)
    @inbounds for index2 in ptrptr[index]:ptrptr[index+1]-1
        i, j = ptr[index2]
        y[i] += x[j]
    end
end

#=
    MadNLP._set_con_scale_sparse!
=#

@kernel function _set_con_scale_sparse_kernel!(
    con_scale,
    @Const(ptr),
    @Const(inds),
    @Const(jac_I),
    @Const(jac_buffer)
)
    index = @index(Global)

    @inbounds begin
        rng = ptr[index]:ptr[index+1]-1

        for k in rng
            (row, i) = inds[k]
            con_scale[row] = max(con_scale[row], abs(jac_buffer[i]))
        end
    end
end

#=
    MadNLP._build_condensed_aug_symbolic_hess
=#

@kernel function _build_condensed_aug_symbolic_hess_kernel!(
    sym,
    sym2,
    @Const(colptr),
    @Const(rowval)
)
    i = @index(Global)
    @inbounds for j in colptr[i]:colptr[i+1]-1
        c = rowval[j]
        sym[j] = (0, j, 0)
        sym2[j] = (c, i)
    end
end

#=
    MadNLP._build_condensed_aug_symbolic_jt
=#

@kernel function _build_condensed_aug_symbolic_jt_kernel!(
    sym,
    sym2,
    @Const(colptr),
    @Const(rowval),
    @Const(offsets)
)
    i = @index(Global)
    @inbounds begin
        cnt = if i == 1
            0
        else
            offsets[i-1]
        end
        for j in colptr[i]:colptr[i+1]-1
            c1 = rowval[j]
            for k in j:colptr[i+1]-1
                c2 = rowval[k]
                cnt += 1
                sym[cnt] = (i, j, k)
                sym2[cnt] = (c2, c1)
            end
        end
    end
end

#=
    MadNLP._build_scale_augmented_system_coo!
=#

@kernel function _scale_augmented_system_coo_kernel!(dest_V, @Const(src_I), @Const(src_J), @Const(src_V), @Const(scaling), @Const(n), @Const(m))
    k = @index(Global, Linear)
    i = src_I[k]
    j = src_J[k]

    # Primal regularization pr_diag
    if k <= n
        dest_V[k] = src_V[k]
    # Hessian block
    elseif i <= n && j <= n
        dest_V[k] = src_V[k] * scaling[i] * scaling[j]
    # Jacobian block
    elseif n + 1 <= i <= n + m && j <= n
        dest_V[k] = src_V[k] * scaling[j]
    # Dual regularization du_diag
    elseif (n + 1 <= i <= n + m) && (n + 1 <= j <= n + m)
        dest_V[k] = src_V[k]
    end
end
