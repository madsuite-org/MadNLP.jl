####################################################################
##### GPU kernels for DenseKKTSystem / DenseCondensedKKTSystem #####
####################################################################

#=
    MadCore.diag!
=#

@kernel function _copy_diag_kernel!(dest, src)
    i = @index(Global)
    @inbounds dest[i] = src[i, i]
end

#=
    MadCore.diag_add!
=#

@kernel function _add_diagonal_kernel!(dest, src1, src2)
    i = @index(Global)
    @inbounds dest[i, i] = src1[i] + src2[i]
end

#=
    MadCore._set_diag!
=#

@kernel function _set_diag_kernel!(A, inds, a)
    i = @index(Global)
    @inbounds begin
        index = inds[i]
        A[index, index] = a
    end
end

#=
    MadCore._build_dense_kkt_system!
=#

@kernel function _build_dense_kkt_system_kernel!(
        dest,
        hess,
        jac,
        pr_diag,
        du_diag,
        diag_hess,
        ind_ineq,
        n,
        m,
        ns,
    )
    i, j = @index(Global, NTuple)
    @inbounds if (i <= n)
        # Transfer Hessian
        if (i == j)
            dest[i, i] = pr_diag[i] + diag_hess[i]
        else
            dest[i, j] = hess[i, j]
        end
    elseif i <= n + ns
        # Transfer slack diagonal
        dest[i, i] = pr_diag[i]
        # Transfer Jacobian wrt slack
        js = i - n
        is = ind_ineq[js]
        dest[is + n + ns, is + n] = -1
        dest[is + n, is + n + ns] = -1
    elseif i <= n + ns + m
        # Transfer Jacobian wrt variable x
        i_ = i - n - ns
        dest[i, j] = jac[i_, j]
        dest[j, i] = jac[i_, j]
        # Transfer dual regularization
        dest[i, i] = du_diag[i_]
    end
end

#=
    MadCore._build_ineq_jac!
=#

@kernel function _build_jacobian_condensed_kernel!(dest, jac, diag_buffer, ind_ineq, m_ineq)
    i, j = @index(Global, NTuple)
    @inbounds is = ind_ineq[i]
    @inbounds dest[i, j] = jac[is, j] * sqrt(diag_buffer[i])
end

#=
    MadCore._build_condensed_kkt_system!
=#

@kernel function _build_condensed_kkt_system_kernel!(
        dest,
        hess,
        jac,
        pr_diag,
        du_diag,
        ind_eq,
        n,
        m_eq,
    )
    i, j = @index(Global, NTuple)

    # Transfer Hessian
    @inbounds if i <= n
        if i == j
            dest[i, i] += pr_diag[i] + hess[i, i]
        else
            dest[i, j] += hess[i, j]
        end
    elseif i <= n + m_eq
        i_ = i - n
        is = ind_eq[i_]
        # Jacobian / equality
        dest[i_ + n, j] = jac[is, j]
        dest[j, i_ + n] = jac[is, j]
        # Transfer dual regularization
        dest[i_ + n, i_ + n] = du_diag[is]
    end
end
