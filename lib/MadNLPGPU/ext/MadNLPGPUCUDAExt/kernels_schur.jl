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

# Inequality condensation → A_kk (lower triangle). ATOMIC: a scenario variable
# that appears in several inequalities produces several contributions to the SAME
# A_kk nzval slot (e.g. its diagonal), so concurrent threads must not race.
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
    @inbounds begin
        val = diag_buffer[bufidx_flat[i]] * jac[jcoo1_flat[i]] * jac[jcoo2_flat[i]]
        CUDACore.@atomic nzval[offset + nzpos_flat[i]] += val
    end
end

# Inequality condensation → C_dk (blk_size × nd × ns; see _scatter_to_Cdk_batched!).
# ATOMIC for the same reason as the A_kk kernel: a (design, scenario-var) pair can
# receive contributions from several inequalities targeting the same C_dk cell.
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
    @inbounds begin
        val = diag_buffer[bufidx_flat[i]] * jac[jcoo_d_flat[i]] * jac[jcoo_v_flat[i]]
        CUDACore.@atomic C_dk[v_idx_flat[i], d_idx_flat[i], k] += val
    end
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

# Add pr_diag_dd to S diagonal (design vars indexed by their global position —
# they are not necessarily the last nd entries).
@kernel function _init_S_diag_kernel!(
    S,
    @Const(pr_diag),
        @Const(design_var_global),
    @Const(nd),
)
    idx = @index(Global)
    if idx <= nd
        @inbounds S[idx, idx] += pr_diag[design_var_global[idx]]
    end
end

# Extract per-scenario RHS from global wx → rhs_k_batched (nv × ns). Under
# RelaxEquality the per-scenario block has no equality rows (blk_size == nv), so
# only scenario variables are gathered. Their global indices come from
# `scen_var_global` (flat, length ns*nv, scenario-major), since a scenario's
# variables need not be contiguous.
@kernel function _extract_rhs_kernel!(
    rhs_k,
    @Const(wx),
    @Const(scen_var_global),
    @Const(nv),
    @Const(ns),
)
    i = @index(Global)
    k = (i - 1) ÷ nv + 1
    local_idx = (i - 1) % nv + 1
    if k <= ns
        @inbounds gi = scen_var_global[(k - 1) * nv + local_idx]
        @inbounds rhs_k[local_idx, k] = wx[gi]
    end
end

# Write back per-scenario solution to global wx
@kernel function _writeback_rhs_kernel!(
    wx,
    @Const(rhs_k),
    @Const(scen_var_global),
    @Const(nv),
    @Const(ns),
)
    i = @index(Global)
    k = (i - 1) ÷ nv + 1
    local_idx = (i - 1) % nv + 1
    if k <= ns
        @inbounds gi = scen_var_global[(k - 1) * nv + local_idx]
        @inbounds wx[gi] = rhs_k[local_idx, k]
    end
end

# ===== Sparse-Schur (cuDSS) assembly kernels =====================================
# These scatter into the lower-triangular CSC `nzval` of the sparse Schur complement
# at precomputed nzval positions. Atomic because overlapping contributions (design
# Hessian + inequality condensation + Schur fill) can target the same slot.

# Scatter src[src_idx[i]] into nzval[nzpos[i]]. Used for design Hessian, pr_diag
# diagonal, design-eq border, and -Δ_eq diagonal.
@kernel function _scatter_to_csc_atomic!(
        nzval,
        @Const(src),
        @Const(src_idx),
        @Const(nzpos),
    )
    i = @index(Global)
    @inbounds CUDACore.@atomic nzval[nzpos[i]] += src[src_idx[i]]
end

# Inequality condensation → CSC nzval (scenario + design): atomic quad-add.
@kernel function _ineq_condense_csc_kernel!(
        nzval,
        @Const(jac),
        @Const(diag_buffer),
        @Const(nzpos),
        @Const(jcoo1),
        @Const(jcoo2),
        @Const(bufidx),
    )
    i = @index(Global)
    @inbounds begin
        val = diag_buffer[bufidx[i]] * jac[jcoo1[i]] * jac[jcoo2[i]]
        CUDACore.@atomic nzval[nzpos[i]] += val
    end
end

# Schur reduction block → CSC nzval: subtract -D[a,b,k] for the lower-or-diagonal
# half (a>=b) into the single lower-triangle slot fill_nzpos[a,b]. D is the
# per-scenario m×m symmetric block C_dk_red[:,:,k]' * tmp_red[:,:,k]; writing only
# a>=b avoids double-counting off-diagonals on the lower-only CSC.
@kernel function _scatter_schur_block!(
        nzval,
        @Const(D),
        @Const(fill_nzpos),
        @Const(m),
        @Const(ns),
    )
    i = @index(Global)
    a = (i - 1) % m + 1
    b = ((i - 1) ÷ m) % m + 1
    k = (i - 1) ÷ (m * m) + 1
    if k <= ns && a >= b
        @inbounds CUDACore.@atomic nzval[fill_nzpos[a, b]] += -D[a, b, k]
    end
end

# ===== Deterministic (atomic-free) condensation scatters ========================
# The atomic-add condensation scatters above are correct but their summation ORDER is
# nondeterministic. With the RelaxEquality condensation weights Σ ≈ 1e8, that reorders the
# accumulation enough to perturb the assembled blocks by ~1e-7 run-to-run; amplified by the
# (legitimate) KKT conditioning this tips the IPM step-acceptance residual across its
# threshold → nondeterministic convergence. These deterministic variants assign ONE thread
# per output slot, which sums that slot's contributions in a fixed (sorted) order with a plain
# `+=` (no two threads touch the same slot within a launch, and launches are sequential), so
# the assembled blocks are reproducible run-to-run — matching the deterministic CPU assembly.
#
# `segstart`/`segslot` describe, for one target array, the contiguous contribution ranges per
# slot in a stable-by-slot ordering: segment `s` owns sorted contributions
# `segstart[s] : segstart[s+1]-1`, all targeting linear index `segslot[s]`.

# Linear scatter value = src[src_idx] → any target nzval, deterministic (one thread per slot).
@kernel function _det_lin_scatter!(
        nzval,
        @Const(src),
        @Const(src_idx),
        @Const(segstart),
        @Const(segslot),
    )
    s = @index(Global)
    @inbounds begin
        acc = zero(eltype(nzval))
        for j in segstart[s]:(segstart[s + 1] - 1)
            acc += src[src_idx[j]]
        end
        nzval[segslot[s]] += acc
    end
end

# Inequality condensation (Σ-amplified) → any target nzval (A_kk / C_dk / S), deterministic.
@kernel function _det_quad_scatter!(
        nzval,
        @Const(jac),
        @Const(diag_buffer),
        @Const(jc1),
        @Const(jc2),
        @Const(buf),
        @Const(segstart),
        @Const(segslot),
    )
    s = @index(Global)
    @inbounds begin
        acc = zero(eltype(nzval))
        for j in segstart[s]:(segstart[s + 1] - 1)
            acc += diag_buffer[buf[j]] * jac[jc1[j]] * jac[jc2[j]]
        end
        nzval[segslot[s]] += acc
    end
end

# Schur reduction block → CSC nzval, deterministic: one thread per lower-triangle (a,b) slot
# sums -D[a,b,k] over the ns scenarios in fixed order.
@kernel function _det_schur_block!(
        nzval,
        @Const(D),
        @Const(fill_nzpos),
        @Const(m),
        @Const(ns),
    )
    i = @index(Global)
    a = (i - 1) % m + 1
    b = (i - 1) ÷ m + 1
    if a >= b
        @inbounds begin
            acc = zero(eltype(nzval))
            for k in 1:ns
                acc += -D[a, b, k]
            end
            nzval[fill_nzpos[a, b]] += acc
        end
    end
end
