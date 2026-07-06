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

# ===== Sparse-Schur (cuDSS) assembly kernel ======================================
# Scatters into the lower-triangular CSC `nzval` of the sparse Schur complement at a
# precomputed nzval position. Used for the design pr_diag diagonal, which targets distinct
# diagonal slots with no collision (so a plain atomic add is race-free and deterministic; the
# Σ-amplified / duplicate-prone contributions go through the deterministic scatters below).
@kernel function _scatter_to_csc_atomic!(
        nzval,
        @Const(src),
        @Const(src_idx),
        @Const(nzpos),
    )
    i = @index(Global)
    @inbounds CUDACore.@atomic nzval[nzpos[i]] += src[src_idx[i]]
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
