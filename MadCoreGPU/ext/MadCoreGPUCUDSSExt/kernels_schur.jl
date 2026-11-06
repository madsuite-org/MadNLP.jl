##################################################
##### GPU kernels for GPUSchurComplementKKT #####
##################################################
#
# Index convention: index maps for all scenarios are stored in flat arrays of
# length `ns * n_per_s`, concatenated in scenario order. Per-scenario A_kk
# blocks share row/col structure, so each scenario writes into its own slice
# of the batched nzval at offset `(k-1) * nnz_per_block`. The kernel global
# index `i` directly indexes the flat input arrays; the scenario index is
# `k = (i-1) ÷ n_per_s + 1`.

# Scatter src[src_idx[i]] into batched A_kk nzval at the matching scenario slot.
# Used for: Hessian diagonal entries, equality Jacobian entries, pr_diag, du_diag.
@kernel function _scatter_to_Akk_batched!(
    nzval,
    @Const(src),
    @Const(src_idx),
    @Const(nzpos_flat),
    @Const(nnz_per_block),
    @Const(n_per_s),
)
    i = @index(Global)
    offset = ((i - 1) ÷ n_per_s) * nnz_per_block
    @inbounds nzval[offset + nzpos_flat[i]] += src[src_idx[i]]
end

# Scatter src[src_idx[i]] into C_dk[v_idx, d_idx, k] (blk_size × nd × ns).
# Layout has scenario-var dim first so each C_dk[:, :, k] is a contiguous
# (blk × nd) matrix — directly usable as a cuDSS multi-RHS dense matrix.
# Used for: Hessian coupling entries, equality Jacobian coupling entries.
@kernel function _scatter_to_Cdk_batched!(
    C_dk,
    @Const(src),
    @Const(src_idx),
    @Const(v_idx_flat),       # scenario-var index (1..blk)
    @Const(d_idx_flat),       # design-var index (1..nd)
    @Const(n_per_s),
)
    i = @index(Global)
    k = (i - 1) ÷ n_per_s + 1
    @inbounds C_dk[v_idx_flat[i], d_idx_flat[i], k] += src[src_idx[i]]
end

# Inequality condensation → A_kk (lower triangle).
@kernel function _ineq_condense_Akk_kernel!(
    nzval,
    @Const(jac),
    @Const(diag_buffer),
    @Const(nzpos_flat),
    @Const(jcoo1_flat),
    @Const(jcoo2_flat),
    @Const(bufidx_flat),
    @Const(nnz_per_block),
    @Const(n_per_s),
)
    i = @index(Global)
    offset = ((i - 1) ÷ n_per_s) * nnz_per_block
    @inbounds nzval[offset + nzpos_flat[i]] += diag_buffer[bufidx_flat[i]] *
        jac[jcoo1_flat[i]] * jac[jcoo2_flat[i]]
end

# Inequality condensation → C_dk (blk_size × nd × ns; see _scatter_to_Cdk_batched!).
@kernel function _ineq_condense_Cdk_kernel!(
    C_dk,
    @Const(jac),
    @Const(diag_buffer),
    @Const(v_idx_flat),       # scenario-var index (1..blk)
    @Const(d_idx_flat),       # design-var index (1..nd)
    @Const(jcoo_d_flat),
    @Const(jcoo_v_flat),
    @Const(bufidx_flat),
    @Const(n_per_s),
)
    i = @index(Global)
    k = (i - 1) ÷ n_per_s + 1
    @inbounds C_dk[v_idx_flat[i], d_idx_flat[i], k] += diag_buffer[bufidx_flat[i]] *
        jac[jcoo_d_flat[i]] * jac[jcoo_v_flat[i]]
end

# Inequality condensation → S (atomic adds since multiple scenarios target the
# same nd × nd dense matrix).
@kernel function _ineq_condense_S_kernel!(
    S,
    @Const(jac),
    @Const(diag_buffer),
    @Const(row_flat),
    @Const(col_flat),
    @Const(jcoo1_flat),
    @Const(jcoo2_flat),
    @Const(bufidx_flat),
)
    i = @index(Global)
    @inbounds begin
        val = diag_buffer[bufidx_flat[i]] * jac[jcoo1_flat[i]] * jac[jcoo2_flat[i]]
        CUDACore.@atomic S[row_flat[i], col_flat[i]] += val
    end
end

# Initialize S from design-design Hessian
@kernel function _init_S_hess_kernel!(
    S,
    @Const(hess),
    @Const(coo_indices),
    @Const(row_indices),
    @Const(col_indices),
)
    idx = @index(Global)
    @inbounds S[row_indices[idx], col_indices[idx]] += hess[coo_indices[idx]]
end

# Add pr_diag_dd to S diagonal
@kernel function _init_S_diag_kernel!(
    S,
    @Const(pr_diag),
    @Const(diag_offset),
    @Const(nd),
)
    idx = @index(Global)
    if idx <= nd
        @inbounds S[idx, idx] += pr_diag[diag_offset + idx]
    end
end

# Extract per-scenario RHS from global wx/wy → rhs_k_batched (blk_size × ns)
@kernel function _extract_rhs_kernel!(
    rhs_k,
    @Const(wx),
    @Const(wy),
    @Const(eq_global_indices),
    @Const(nv),
    @Const(nc_eq),
    @Const(ns),
    @Const(blk_size),
)
    i = @index(Global)
    k = (i - 1) ÷ blk_size + 1
    local_idx = (i - 1) % blk_size + 1
    if k <= ns
        if local_idx <= nv
            gi = (k - 1) * nv + local_idx
            @inbounds rhs_k[local_idx, k] = wx[gi]
        else
            eq_idx = local_idx - nv
            flat_idx = (k - 1) * nc_eq + eq_idx
            @inbounds rhs_k[local_idx, k] = wy[eq_global_indices[flat_idx]]
        end
    end
end

# Write back per-scenario solution to global wx/wy
@kernel function _writeback_rhs_kernel!(
    wx,
    wy,
    @Const(rhs_k),
    @Const(eq_global_indices),
    @Const(nv),
    @Const(nc_eq),
    @Const(ns),
    @Const(blk_size),
)
    i = @index(Global)
    k = (i - 1) ÷ blk_size + 1
    local_idx = (i - 1) % blk_size + 1
    if k <= ns
        if local_idx <= nv
            gi = (k - 1) * nv + local_idx
            @inbounds wx[gi] = rhs_k[local_idx, k]
        else
            eq_idx = local_idx - nv
            flat_idx = (k - 1) * nc_eq + eq_idx
            @inbounds wy[eq_global_indices[flat_idx]] = rhs_k[local_idx, k]
        end
    end
end
