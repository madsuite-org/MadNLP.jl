
"""
    ScenarioBlockMap

Precomputed index mappings for one scenario — maps from global COO indices
to per-block positions in A_kk (sparse), C_dk (dense), and S (dense).
"""
struct ScenarioBlockMap
    # Hessian diagonal block: global hess COO index → A_kk nzval position
    hess_Akk_coo::Vector{Int}
    hess_Akk_nzpos::Vector{Int}

    # Hessian coupling block: global hess COO index → (C_dk row, C_dk col)
    hess_Cdk_coo::Vector{Int}
    hess_Cdk_row::Vector{Int}
    hess_Cdk_col::Vector{Int}

    # Equality Jacobian, scenario vars → A_kk lower triangle only
    jeq_Akk_coo::Vector{Int}
    jeq_Akk_nzpos::Vector{Int}

    # Equality Jacobian, design vars → C_dk
    jeq_Cdk_coo::Vector{Int}
    jeq_Cdk_row::Vector{Int}   # design var index (1:nd)
    jeq_Cdk_col::Vector{Int}   # nv + eq_local_idx

    # Inequality condensation → A_kk (lower triangle)
    ineq_Akk_nzpos::Vector{Int}
    ineq_Akk_jcoo1::Vector{Int}
    ineq_Akk_jcoo2::Vector{Int}
    ineq_Akk_bufidx::Vector{Int}

    # Inequality condensation → C_dk
    ineq_Cdk_row::Vector{Int}     # design var index (1:nd)
    ineq_Cdk_col::Vector{Int}     # scenario local var (1:nv)
    ineq_Cdk_jcoo_d::Vector{Int}  # jac COO for design var
    ineq_Cdk_jcoo_v::Vector{Int}  # jac COO for scenario var
    ineq_Cdk_bufidx::Vector{Int}

    # Inequality condensation → S (full, both triangles)
    ineq_S_row::Vector{Int}
    ineq_S_col::Vector{Int}
    ineq_S_jcoo1::Vector{Int}
    ineq_S_jcoo2::Vector{Int}
    ineq_S_bufidx::Vector{Int}

    # Diagonal positions in A_kk nzval for pr_diag (nv entries)
    pr_diag_global::Vector{Int}
    pr_diag_nzpos::Vector{Int}

    # Diagonal positions in A_kk nzval for du_diag (nc_eq entries)
    du_diag_global::Vector{Int}
    du_diag_nzpos::Vector{Int}
end

"""
    SchurComplementKKTSystem{T, VT, MT, QN, LS, LS2, VI} <: AbstractCondensedKKTSystem{T, VT, MT, QN}

KKT system exploiting block-arrowhead structure from two-stage stochastic programs
via Schur complement decomposition, using sparse COO/CSC storage for the global
Hessian and Jacobian and sparse per-scenario block solvers.

Variable layout: `[v_1, ..., v_ns, d]` where `v_k ∈ R^nv`, `d ∈ R^nd`.
Constraint layout: `[c_1, ..., c_ns]` where `c_k ∈ R^nc`.

The augmented per-scenario block `A_k` (size `blk_size × blk_size`) is stored
as a sparse lower-triangular `SparseMatrixCSC` and factored by a configurable
sparse solver (default `MumpsSolver` — each `A_k` is symmetric indefinite, not
SQD in general). The coupling blocks `C_dk` remain dense. The Schur complement
`S = aug_com` (size `nd × nd`) is dense.
"""
struct SchurComplementKKTSystem{
    T,
    VT <: AbstractVector{T},
    MT <: AbstractMatrix{T},
    QN,
    LS,
    LS2,
    VI <: AbstractVector{Int}
    } <: AbstractCondensedKKTSystem{T, VT, MT, QN}

    # COO value buffers (filled by NLP callback via jac_coord!/hess_coord!)
    hess::VT                        # length cb.nnzh
    jac::VT                         # length cb.nnzj

    # COO structures (share value buffers above)
    hess_raw::SparseMatrixCOO{T, Int32, VT, Vector{Int32}}
    jt_coo::SparseMatrixCOO{T, Int32, VT, Vector{Int32}}

    # CSC representations (for jtprod, mul, dual recovery)
    hess_csc::SparseMatrixCSC{T, Int32}
    hess_csc_map::Vector{Int}
    jt_csc::SparseMatrixCSC{T, Int32}
    jt_csc_map::Vector{Int}

    quasi_newton::QN

    # Standard MadNLP diagonal vectors
    reg::VT                         # n_total + n_ineq
    pr_diag::VT                     # n_total + n_ineq
    du_diag::VT                     # m_total
    l_diag::VT                      # nlb
    u_diag::VT                      # nub
    l_lower::VT                     # nlb
    u_lower::VT                     # nub

    # Two-stage dimensions
    ns::Int
    nv::Int
    nd::Int
    nc::Int
    nc_eq_per_s::Int
    nc_ineq_per_s::Int
    blk_size::Int                   # = nv + nc_eq_per_s

    # Per-scenario sparse augmented blocks (lower triangle only)
    A_kk::Vector{SparseMatrixCSC{T, Int32}}
    C_dk::Vector{MT}                # ns × (nd × blk_size) — dense

    # Schur complement (what the dense linear solver sees)
    aug_com::MT                     # nd × nd

    # Buffers
    diag_buffer::VT                 # n_ineq — condensing diagonal
    buffer::VT                      # m_total — general
    wy_eq_buf::VT                   # ns*nc_eq_per_s — preserves eq duals across J*Δx round-trip
    rhs_d::VT                       # nd — design RHS
    rhs_k::Vector{VT}              # ns × blk_size — scenario RHS buffers
    tmp_blk_nd::Vector{MT}         # ns × (blk_size × nd)
    solve_buffers::Vector{VT}      # ns × blk_size — per-scenario column-by-column solve buffers

    # Precomputed index maps
    block_maps::Vector{ScenarioBlockMap}
    hess_S_coo::Vector{Int}         # COO indices for design-design Hessian
    hess_S_row::Vector{Int}         # S row (1:nd)
    hess_S_col::Vector{Int}         # S col (1:nd)

    # Flat per-scenario equality indices (length ns * nc_eq_per_s).
    # Scenario k's indices live at (k-1)*nc_eq_per_s+1 : k*nc_eq_per_s.
    eq_global_indices::Vector{Int}

    # Inequality/equality/bound index info
    n_eq::Int
    ind_eq::VI
    n_ineq::Int
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI

    # Solvers
    scenario_solvers::Vector{LS2}
    linear_solver::LS               # for Schur complement S (dense)
end

"""
    _resolve_schur_dims(cb, n, m, schur_ns, schur_nv, schur_nd, schur_nc)
    -> (ns, nv, nd, nc)

Resolve two-stage stochastic dimensions for a SchurComplementKKTSystem.

If `schur_ns == 0`, attempt to auto-detect from `cb.nlp.tags::TwoStageTags`
(an ExaModel convention): `tags.var_scenario[i] == 0` flags design variables,
`== 1` flags scenario variables; same encoding for `con_scenario`.

Asserts that the resolved dimensions are consistent with `n` and `m`.
"""
function _resolve_schur_dims(cb, n, m, schur_ns, schur_nv, schur_nd, schur_nc)
    if schur_ns == 0 && hasproperty(cb.nlp, :tags)
        tags = cb.nlp.tags
        if hasproperty(tags, :ns) && hasproperty(tags, :var_scenario) && hasproperty(tags, :con_scenario)
            schur_ns = tags.ns
            var_scen = Array(tags.var_scenario)
            con_scen = Array(tags.con_scenario)

            # Single-pass histograms over the scenario tags. Index 1 is the design
            # bucket (tag 0); 1+k is scenario k. Cheaper than the previous three
            # `count(==(·))` passes and lets us validate per-scenario uniformity
            # in the same pass — important because a malformed model whose global
            # aggregates happen to satisfy n == ns*nv + nd would otherwise drive
            # downstream symbolic build into garbage territory.
            var_hist = zeros(Int, schur_ns + 1)
            for tag in var_scen
                t = Int(tag)
                (0 <= t <= schur_ns) || error(
                    "var_scenario tag $t out of range [0, $schur_ns]; " *
                    "0 = design, 1..$schur_ns = scenario index."
                )
                @inbounds var_hist[t + 1] += 1
            end
            con_hist = zeros(Int, schur_ns + 1)
            for tag in con_scen
                t = Int(tag)
                (0 <= t <= schur_ns) || error(
                    "con_scenario tag $t out of range [0, $schur_ns]; " *
                    "1..$schur_ns = scenario index."
                )
                @inbounds con_hist[t + 1] += 1
            end

            schur_nd = var_hist[1]
            schur_nv = var_hist[2]
            schur_nc = con_hist[2]

            # Reject design-only constraints — the Schur reduction has no slot
            # for them.
            con_hist[1] == 0 || error(
                "$(con_hist[1]) constraints have con_scenario tag 0; " *
                "design-only constraints are not supported by SchurComplementKKTSystem."
            )

            # Per-scenario uniformity. Cheap loop, fires before any symbolic
            # work and points at the offending scenario.
            for k in 2:schur_ns
                @inbounds nv_k = var_hist[k + 1]
                @inbounds nc_k = con_hist[k + 1]
                nv_k == schur_nv || error(
                    "Scenario $k has $nv_k variables; scenario 1 has $schur_nv. " *
                    "SchurComplementKKTSystem requires uniform per-scenario sizes."
                )
                nc_k == schur_nc || error(
                    "Scenario $k has $nc_k constraints; scenario 1 has $schur_nc. " *
                    "SchurComplementKKTSystem requires uniform per-scenario sizes."
                )
            end
        end
    end

    @assert schur_ns > 0 "schur_ns must be specified and positive (or use TwoStageTags for auto-detection)"
    @assert schur_nv > 0 "schur_nv must be specified and positive"
    @assert schur_nd > 0 "schur_nd must be specified and positive"
    @assert n == schur_ns * schur_nv + schur_nd "Variable count mismatch: n=$n != ns*nv+nd=$(schur_ns*schur_nv+schur_nd)"
    @assert m == schur_ns * schur_nc "Constraint count mismatch: m=$m != ns*nc=$(schur_ns*schur_nc)"

    return schur_ns, schur_nv, schur_nd, schur_nc
end

# --- Index-driven scatter helpers used by build_kkt! ---
# Each call site reads as a one-liner instead of an open-coded for loop.

@inline function _scatter_add!(dst::AbstractVector, src::AbstractVector, src_idx, dst_idx)
    @inbounds for i in eachindex(src_idx)
        dst[dst_idx[i]] += src[src_idx[i]]
    end
end

@inline function _scatter_add!(dst::AbstractMatrix, src::AbstractVector, src_idx, row, col)
    @inbounds for i in eachindex(src_idx)
        dst[row[i], col[i]] += src[src_idx[i]]
    end
end

@inline function _scatter_quad_add!(dst::AbstractVector, src::AbstractVector, diag, dst_idx, idx1, idx2, bufidx)
    @inbounds for i in eachindex(dst_idx)
        dst[dst_idx[i]] += diag[bufidx[i]] * src[idx1[i]] * src[idx2[i]]
    end
end

@inline function _scatter_quad_add!(dst::AbstractMatrix, src::AbstractVector, diag, row, col, idx1, idx2, bufidx)
    @inbounds for i in eachindex(row)
        dst[row[i], col[i]] += diag[bufidx[i]] * src[idx1[i]] * src[idx2[i]]
    end
end

"""
    _build_schur_symbolic(T, n, m, ns, nv, nd, nc,
                          hess_I, hess_J, jac_I, jac_J,
                          ind_eq, ind_ineq) -> NamedTuple

Pure-CPU symbolic construction shared by the CPU `SchurComplementKKTSystem`
and GPU `GPUSchurComplementKKTSystem` constructors. Inputs are CPU-resident
sparsity arrays (the GPU side downloads its sparsity before calling).

**Assumes uniform per-scenario structure** (same A_kk pattern for every k).
This matches typical two-stage stochastic models and is what the GPU batched
cuDSS path requires; CPU follows the same assumption.

Returns a NamedTuple with:
- `eq_per_scenario`, `ineq_per_scenario` — `Vector{Vector{Int}}` of global indices
- `nc_eq_per_s`, `nc_ineq_per_s`, `blk_size` — per-scenario sizes
- `block_maps::Vector{ScenarioBlockMap}` — per-scenario index maps
- `hess_S_coo`, `hess_S_row`, `hess_S_col` — design-design Hessian COO maps
- `akk_csc_template::SparseMatrixCSC{T, Int32}` — shared A_kk sparsity (zero values)
- `nnz_per_scenario::Int`
- `eq_global_flat::Vector{Int}` — flattened eq indices in scenario order (for GPU)
"""
function _build_schur_symbolic(
        ::Type{T},
        n::Int, m::Int, ns::Int, nv::Int, nd::Int, nc::Int,
        hess_I::AbstractVector{<:Integer}, hess_J::AbstractVector{<:Integer},
        jac_I::AbstractVector{<:Integer}, jac_J::AbstractVector{<:Integer},
        ind_eq::AbstractVector{<:Integer},
        ind_ineq::AbstractVector{<:Integer},
    ) where {T}

    n_hess = length(hess_I)
    n_jac = length(jac_I)

    # --- Classify constraints per scenario ---
    ind_eq_set = Set(ind_eq)
    ind_ineq_set = Set(ind_ineq)

    eq_per_scenario = Vector{Vector{Int}}(undef, ns)
    ineq_per_scenario = Vector{Vector{Int}}(undef, ns)

    for k in 1:ns
        cr = (k-1)*nc+1 : k*nc
        eq_per_scenario[k] = Int[]
        ineq_per_scenario[k] = Int[]
        for gi in cr
            if gi in ind_eq_set
                push!(eq_per_scenario[k], gi)
            end
            if gi in ind_ineq_set
                push!(ineq_per_scenario[k], gi)
            end
        end
    end

    # Scenario 1 sets the canonical per-scenario constraint count; reject any
    # scenario that disagrees, since downstream code (and the GPU batched layout)
    # assumes uniform per-scenario shape.
    nc_eq_per_s = length(eq_per_scenario[1])
    nc_ineq_per_s = length(ineq_per_scenario[1])
    for k in 2:ns
        n_eq_k = length(eq_per_scenario[k])
        n_in_k = length(ineq_per_scenario[k])
        if n_eq_k != nc_eq_per_s || n_in_k != nc_ineq_per_s
            error(
                "SchurComplementKKTSystem requires uniform per-scenario constraint counts. " *
                "Scenario 1 has (eq=$nc_eq_per_s, ineq=$nc_ineq_per_s); " *
                "scenario $k has (eq=$n_eq_k, ineq=$n_in_k)."
            )
        end
    end

    blk_size = nv + nc_eq_per_s

    # Lookup: global ineq index → diag_buffer index
    ineq_to_bufidx = Dict{Int,Int}()
    for idx in 1:length(ind_ineq)
        ineq_to_bufidx[Int(ind_ineq[idx])] = idx
    end

    # Lookup: constraint global row → list of (jac_coo_idx, col)
    jac_by_constraint = Dict{Int, Vector{Tuple{Int,Int}}}()
    for ci in 1:n_jac
        row = Int(jac_I[ci])
        col = Int(jac_J[ci])
        entries = get!(Vector{Tuple{Int,Int}}, jac_by_constraint, row)
        push!(entries, (ci, col))
    end

    d_start = ns * nv + 1
    d_end = ns * nv + nd

    # --- Classify Hessian COO entries ---
    hess_S_coo = Int[]
    hess_S_row = Int[]
    hess_S_col = Int[]

    hess_per_scenario_diag = [Tuple{Int,Int,Int}[] for _ in 1:ns]      # (coo, local_i, local_j)
    hess_per_scenario_coupling = [Tuple{Int,Int,Int}[] for _ in 1:ns]  # (coo, design_local, var_local)
    hess_classified = falses(n_hess)

    for ci in 1:n_hess
        ri = Int(hess_I[ci])
        rj = Int(hess_J[ci])  # lower triangle: ri >= rj

        # Both design vars → S entry (write both triangles for dense S)
        if ri >= d_start && ri <= d_end && rj >= d_start && rj <= d_end
            di = ri - d_start + 1
            dj = rj - d_start + 1
            push!(hess_S_coo, ci); push!(hess_S_row, di); push!(hess_S_col, dj)
            if di != dj
                push!(hess_S_coo, ci); push!(hess_S_row, dj); push!(hess_S_col, di)
            end
            hess_classified[ci] = true
            continue
        end

        # One design + one scenario var → coupling block
        if ri >= d_start && ri <= d_end && rj < d_start
            di = ri - d_start + 1
            k = div(rj - 1, nv) + 1
            if k >= 1 && k <= ns
                vj = rj - (k-1)*nv
                push!(hess_per_scenario_coupling[k], (ci, di, vj))
                hess_classified[ci] = true
            end
            continue
        end

        # Both within the same scenario → A_kk diagonal block
        if ri < d_start && rj < d_start
            ki = div(ri - 1, nv) + 1
            kj = div(rj - 1, nv) + 1
            if ki == kj && ki >= 1 && ki <= ns
                li = ri - (ki-1)*nv
                lj = rj - (ki-1)*nv
                push!(hess_per_scenario_diag[ki], (ci, li, lj))
                hess_classified[ci] = true
            end
        end
    end

    # Anything left unclassified is a Hessian entry that doesn't fit the
    # block-arrowhead pattern — typically cross-scenario coupling. Silently
    # dropping it would converge the IPM to a wrong optimum, so error loudly.
    n_bad_hess = count(!, hess_classified)
    if n_bad_hess > 0
        bad = findall(!, hess_classified)
        sample = first(bad, min(5, length(bad)))
        details = join(("(row=$(Int(hess_I[ci])), col=$(Int(hess_J[ci])))" for ci in sample), ", ")
        error(
            "$n_bad_hess Hessian COO entries do not fit the SchurComplementKKTSystem " *
            "block-arrowhead pattern (likely cross-scenario coupling). First few: " *
            details
        )
    end

    # --- Build shared A_kk template from scenario 1 ---
    eq_cons_1 = eq_per_scenario[1]
    ineq_cons_1 = ineq_per_scenario[1]
    eq_local_1 = Dict{Int,Int}()
    for (ci, gi) in enumerate(eq_cons_1)
        eq_local_1[gi] = ci
    end

    akk_entries = Dict{Tuple{Int,Int}, Nothing}()

    # Hessian diagonal
    for (_, li, lj) in hess_per_scenario_diag[1]
        akk_entries[(li, lj)] = nothing
    end
    # pr_diag (always nv diagonal entries)
    for i in 1:nv
        akk_entries[(i, i)] = nothing
    end
    # Equality Jacobian: row = nv + eq_local, col = local_var
    for gi in eq_cons_1
        for (_, col) in get(jac_by_constraint, gi, Tuple{Int,Int}[])
            if col >= 1 && col <= nv
                akk_entries[(nv + eq_local_1[gi], col)] = nothing
            end
        end
    end
    # du_diag for eq constraints
    for ci in 1:length(eq_cons_1)
        akk_entries[(nv + ci, nv + ci)] = nothing
    end
    # Inequality condensation fill-in (lower triangle pairs of scenario vars)
    for gi in ineq_cons_1
        local_vars = Int[]
        for (_, col) in get(jac_by_constraint, gi, Tuple{Int,Int}[])
            if col >= 1 && col <= nv
                push!(local_vars, col)
            end
        end
        for a in local_vars, b in local_vars
            if a >= b
                akk_entries[(a, b)] = nothing
            end
        end
    end

    akk_nnz = length(akk_entries)
    akk_I = Vector{Int32}(undef, akk_nnz)
    akk_J = Vector{Int32}(undef, akk_nnz)
    akk_V = zeros(T, akk_nnz)
    for (idx, ((ri, rj), _)) in enumerate(akk_entries)
        akk_I[idx] = Int32(ri)
        akk_J[idx] = Int32(rj)
    end
    akk_coo = SparseMatrixCOO(blk_size, blk_size, akk_I, akk_J, akk_V)
    akk_csc_template, _ = coo_to_csc(akk_coo)

    # nzval position lookup, shared across scenarios (uniform structure)
    akk_lookup = Dict{Tuple{Int,Int}, Int}()
    for col in 1:blk_size
        for p in akk_csc_template.colptr[col]:(akk_csc_template.colptr[col+1]-1)
            row = akk_csc_template.rowval[p]
            akk_lookup[(Int(row), Int(col))] = Int(p)
        end
    end

    # Wrap lookups so a missing key (scenario k has a sparsity entry absent from
    # scenario 1's template) surfaces as a meaningful error instead of KeyError.
    @inline akk_pos(key, k) = let p = get(akk_lookup, key, 0)
        p == 0 && error(
            "SchurComplementKKTSystem: scenario $k has an A_kk entry at local " *
            "(row=$(key[1]), col=$(key[2])) absent from scenario 1's template. " *
            "Per-scenario Hessian/Jacobian sparsity must be uniform."
        )
        p
    end

    nnz_per_scenario = length(akk_csc_template.nzval)

    # --- Build per-scenario ScenarioBlockMaps ---
    block_maps = Vector{ScenarioBlockMap}(undef, ns)
    eq_global_flat = Int[]
    jac_classified = falses(n_jac)

    for k in 1:ns
        vr_start = (k-1)*nv + 1
        eq_cons = eq_per_scenario[k]
        ineq_cons = ineq_per_scenario[k]

        eq_local = Dict{Int,Int}()
        for (ci, gi) in enumerate(eq_cons)
            eq_local[gi] = ci
        end

        # Hessian diagonal → A_kk
        hess_Akk_coo_vec = Int[]
        hess_Akk_nzpos_vec = Int[]
        for (ci, li, lj) in hess_per_scenario_diag[k]
            push!(hess_Akk_coo_vec, ci)
            push!(hess_Akk_nzpos_vec, akk_pos((li, lj), k))
        end

        # Hessian coupling → C_dk
        hess_Cdk_coo_vec = Int[]
        hess_Cdk_row_vec = Int[]
        hess_Cdk_col_vec = Int[]
        for (ci, di, vj) in hess_per_scenario_coupling[k]
            push!(hess_Cdk_coo_vec, ci)
            push!(hess_Cdk_row_vec, di)
            push!(hess_Cdk_col_vec, vj)
        end

        # Equality Jacobian → A_kk and C_dk
        jeq_Akk_coo_vec = Int[]
        jeq_Akk_nzpos_vec = Int[]
        jeq_Cdk_coo_vec = Int[]
        jeq_Cdk_row_vec = Int[]
        jeq_Cdk_col_vec = Int[]

        for gi in eq_cons
            local_eq = eq_local[gi]
            for (coo_idx, col) in get(jac_by_constraint, gi, Tuple{Int,Int}[])
                if col >= vr_start && col < vr_start + nv
                    local_var = col - vr_start + 1
                    push!(jeq_Akk_coo_vec, coo_idx)
                    push!(jeq_Akk_nzpos_vec, akk_pos((nv + local_eq, local_var), k))
                    jac_classified[coo_idx] = true
                elseif col >= d_start && col <= d_end
                    di = col - d_start + 1
                    push!(jeq_Cdk_coo_vec, coo_idx)
                    push!(jeq_Cdk_row_vec, di)
                    push!(jeq_Cdk_col_vec, nv + local_eq)
                    jac_classified[coo_idx] = true
                end
            end
        end

        # pr_diag → A_kk diagonal
        pr_diag_global_vec = Int[]
        pr_diag_nzpos_vec = Int[]
        for i in 1:nv
            push!(pr_diag_global_vec, vr_start + i - 1)
            push!(pr_diag_nzpos_vec, akk_pos((i, i), k))
        end

        # du_diag → A_kk diagonal
        du_diag_global_vec = Int[]
        du_diag_nzpos_vec = Int[]
        for (ci, gi) in enumerate(eq_cons)
            push!(du_diag_global_vec, gi)
            push!(du_diag_nzpos_vec, akk_pos((nv + ci, nv + ci), k))
        end

        # Inequality condensation
        ineq_Akk_nzpos_vec = Int[]
        ineq_Akk_jcoo1_vec = Int[]
        ineq_Akk_jcoo2_vec = Int[]
        ineq_Akk_bufidx_vec = Int[]

        ineq_Cdk_row_vec = Int[]
        ineq_Cdk_col_vec = Int[]
        ineq_Cdk_jcoo_d_vec = Int[]
        ineq_Cdk_jcoo_v_vec = Int[]
        ineq_Cdk_bufidx_vec = Int[]

        ineq_S_row_vec = Int[]
        ineq_S_col_vec = Int[]
        ineq_S_jcoo1_vec = Int[]
        ineq_S_jcoo2_vec = Int[]
        ineq_S_bufidx_vec = Int[]

        for gi in ineq_cons
            bidx = ineq_to_bufidx[gi]
            v_entries = Tuple{Int,Int}[]
            d_entries = Tuple{Int,Int}[]

            for (coo_idx, col) in get(jac_by_constraint, gi, Tuple{Int,Int}[])
                if col >= vr_start && col < vr_start + nv
                    push!(v_entries, (coo_idx, col - vr_start + 1))
                    jac_classified[coo_idx] = true
                elseif col >= d_start && col <= d_end
                    push!(d_entries, (coo_idx, col - d_start + 1))
                    jac_classified[coo_idx] = true
                end
            end

            # A_kk: lower-triangle pairs of scenario vars
            for (coo_a, la) in v_entries, (coo_b, lb) in v_entries
                if la >= lb
                    push!(ineq_Akk_nzpos_vec, akk_pos((la, lb), k))
                    push!(ineq_Akk_jcoo1_vec, coo_a)
                    push!(ineq_Akk_jcoo2_vec, coo_b)
                    push!(ineq_Akk_bufidx_vec, bidx)
                end
            end

            # C_dk: design × scenario
            for (coo_d, di) in d_entries, (coo_v, lv) in v_entries
                push!(ineq_Cdk_row_vec, di)
                push!(ineq_Cdk_col_vec, lv)
                push!(ineq_Cdk_jcoo_d_vec, coo_d)
                push!(ineq_Cdk_jcoo_v_vec, coo_v)
                push!(ineq_Cdk_bufidx_vec, bidx)
            end

            # S: design × design (full)
            for (coo_a, da) in d_entries, (coo_b, db) in d_entries
                push!(ineq_S_row_vec, da)
                push!(ineq_S_col_vec, db)
                push!(ineq_S_jcoo1_vec, coo_a)
                push!(ineq_S_jcoo2_vec, coo_b)
                push!(ineq_S_bufidx_vec, bidx)
            end
        end

        block_maps[k] = ScenarioBlockMap(
            hess_Akk_coo_vec, hess_Akk_nzpos_vec,
            hess_Cdk_coo_vec, hess_Cdk_row_vec, hess_Cdk_col_vec,
            jeq_Akk_coo_vec, jeq_Akk_nzpos_vec,
            jeq_Cdk_coo_vec, jeq_Cdk_row_vec, jeq_Cdk_col_vec,
            ineq_Akk_nzpos_vec, ineq_Akk_jcoo1_vec, ineq_Akk_jcoo2_vec, ineq_Akk_bufidx_vec,
            ineq_Cdk_row_vec, ineq_Cdk_col_vec, ineq_Cdk_jcoo_d_vec, ineq_Cdk_jcoo_v_vec, ineq_Cdk_bufidx_vec,
            ineq_S_row_vec, ineq_S_col_vec, ineq_S_jcoo1_vec, ineq_S_jcoo2_vec, ineq_S_bufidx_vec,
            pr_diag_global_vec, pr_diag_nzpos_vec,
            du_diag_global_vec, du_diag_nzpos_vec,
        )

        append!(eq_global_flat, eq_cons)
    end

    # Catch Jacobian entries whose column doesn't match the constraint's own
    # scenario stripe or the design stripe (cross-scenario coupling).
    n_bad_jac = count(!, jac_classified)
    if n_bad_jac > 0
        bad = findall(!, jac_classified)
        sample = first(bad, min(5, length(bad)))
        details = join(("(row=$(Int(jac_I[ci])), col=$(Int(jac_J[ci])))" for ci in sample), ", ")
        error(
            "$n_bad_jac Jacobian COO entries do not fit the SchurComplementKKTSystem " *
            "block-arrowhead pattern (column belongs to a different scenario). First few: " *
            details
        )
    end

    # Per-scenario field cardinalities must match scenario 1, otherwise the GPU
    # batched layout (which uses scenario-1 lengths as the per-scenario stride)
    # would silently drop entries on longer scenarios. This catches non-uniform
    # Hessian/Jacobian sparsity that still passes the aggregate count checks.
    bm1 = block_maps[1]
    fields_to_check = (
        :hess_Akk_coo, :hess_Cdk_coo,
        :jeq_Akk_coo, :jeq_Cdk_coo,
        :ineq_Akk_nzpos, :ineq_Cdk_row, :ineq_S_row,
        :pr_diag_global, :du_diag_global,
    )
    for k in 2:ns, f in fields_to_check
        n1 = length(getfield(bm1, f))
        nk = length(getfield(block_maps[k], f))
        if nk != n1
            error(
                "SchurComplementKKTSystem requires uniform per-scenario sparsity. " *
                "Scenario 1 has $n1 entries in $f; scenario $k has $nk."
            )
        end
    end

    return (
        eq_per_scenario = eq_per_scenario,
        ineq_per_scenario = ineq_per_scenario,
        nc_eq_per_s = nc_eq_per_s,
        nc_ineq_per_s = nc_ineq_per_s,
        blk_size = blk_size,
        block_maps = block_maps,
        hess_S_coo = hess_S_coo,
        hess_S_row = hess_S_row,
        hess_S_col = hess_S_col,
        akk_csc_template = akk_csc_template,
        nnz_per_scenario = nnz_per_scenario,
        eq_global_flat = eq_global_flat,
    )
end

"""
    _flatten_block_maps(block_maps) -> NamedTuple

Flatten `Vector{ScenarioBlockMap}` into the per-field concatenated CPU
`Vector{Int}`s the GPU constructor uploads to device. Also returns the
per-scenario count (`n_per_s_*`) for each map type, taken from scenario 1
under the uniform-structure assumption.
"""
function _flatten_block_maps(block_maps::Vector{ScenarioBlockMap})
    bm1 = block_maps[1]
    n_per_s_hess_Akk = length(bm1.hess_Akk_coo)
    n_per_s_hess_Cdk = length(bm1.hess_Cdk_coo)
    n_per_s_pr_diag  = length(bm1.pr_diag_global)
    n_per_s_du_diag  = length(bm1.du_diag_global)
    n_per_s_jeq_Akk  = length(bm1.jeq_Akk_coo)
    n_per_s_jeq_Cdk  = length(bm1.jeq_Cdk_coo)
    n_per_s_ineq_Akk = length(bm1.ineq_Akk_nzpos)
    n_per_s_ineq_Cdk = length(bm1.ineq_Cdk_row)
    n_per_s_ineq_S   = length(bm1.ineq_S_row)

    cat_int(get) = reduce(vcat, (get(bm) for bm in block_maps); init = Int[])

    return (
        n_per_s_hess_Akk = n_per_s_hess_Akk,
        all_hess_Akk_coo   = cat_int(bm -> bm.hess_Akk_coo),
        all_hess_Akk_nzpos = cat_int(bm -> bm.hess_Akk_nzpos),

        n_per_s_hess_Cdk = n_per_s_hess_Cdk,
        all_hess_Cdk_coo = cat_int(bm -> bm.hess_Cdk_coo),
        all_hess_Cdk_row = cat_int(bm -> bm.hess_Cdk_row),
        all_hess_Cdk_col = cat_int(bm -> bm.hess_Cdk_col),

        n_per_s_pr_diag    = n_per_s_pr_diag,
        all_pr_diag_global = cat_int(bm -> bm.pr_diag_global),
        all_pr_diag_nzpos  = cat_int(bm -> bm.pr_diag_nzpos),

        n_per_s_du_diag    = n_per_s_du_diag,
        all_du_diag_global = cat_int(bm -> bm.du_diag_global),
        all_du_diag_nzpos  = cat_int(bm -> bm.du_diag_nzpos),

        n_per_s_jeq_Akk   = n_per_s_jeq_Akk,
        all_jeq_Akk_coo   = cat_int(bm -> bm.jeq_Akk_coo),
        all_jeq_Akk_nzpos = cat_int(bm -> bm.jeq_Akk_nzpos),

        n_per_s_jeq_Cdk = n_per_s_jeq_Cdk,
        all_jeq_Cdk_coo = cat_int(bm -> bm.jeq_Cdk_coo),
        all_jeq_Cdk_row = cat_int(bm -> bm.jeq_Cdk_row),
        all_jeq_Cdk_col = cat_int(bm -> bm.jeq_Cdk_col),

        n_per_s_ineq_Akk    = n_per_s_ineq_Akk,
        all_ineq_Akk_nzpos  = cat_int(bm -> bm.ineq_Akk_nzpos),
        all_ineq_Akk_jcoo1  = cat_int(bm -> bm.ineq_Akk_jcoo1),
        all_ineq_Akk_jcoo2  = cat_int(bm -> bm.ineq_Akk_jcoo2),
        all_ineq_Akk_bufidx = cat_int(bm -> bm.ineq_Akk_bufidx),

        n_per_s_ineq_Cdk    = n_per_s_ineq_Cdk,
        all_ineq_Cdk_row    = cat_int(bm -> bm.ineq_Cdk_row),
        all_ineq_Cdk_col    = cat_int(bm -> bm.ineq_Cdk_col),
        all_ineq_Cdk_jcoo_d = cat_int(bm -> bm.ineq_Cdk_jcoo_d),
        all_ineq_Cdk_jcoo_v = cat_int(bm -> bm.ineq_Cdk_jcoo_v),
        all_ineq_Cdk_bufidx = cat_int(bm -> bm.ineq_Cdk_bufidx),

        n_per_s_ineq_S     = n_per_s_ineq_S,
        all_ineq_S_row     = cat_int(bm -> bm.ineq_S_row),
        all_ineq_S_col     = cat_int(bm -> bm.ineq_S_col),
        all_ineq_S_jcoo1   = cat_int(bm -> bm.ineq_S_jcoo1),
        all_ineq_S_jcoo2   = cat_int(bm -> bm.ineq_S_jcoo2),
        all_ineq_S_bufidx  = cat_int(bm -> bm.ineq_S_bufidx),
    )
end

function create_kkt_system(
    ::Type{SchurComplementKKTSystem},
    cb::SparseCallback{T,VT},
    linear_solver::Type;
    opt_linear_solver=default_options(linear_solver),
    hessian_approximation=ExactHessian,
    qn_options=QuasiNewtonOptions(),
    schur_ns::Int=0,
    schur_nv::Int=0,
    schur_nd::Int=0,
    schur_nc::Int=0,
    schur_scenario_linear_solver::Type=MumpsSolver,
) where {T, VT}

    n = cb.nvar
    m = cb.ncon
    ns_ineq = length(cb.ind_ineq)
    n_eq = m - ns_ineq
    nlb = length(cb.ind_lb)
    nub = length(cb.ind_ub)

    ns, nv, nd, nc = _resolve_schur_dims(cb, n, m, schur_ns, schur_nv, schur_nd, schur_nc)

    # --- Get sparsity patterns ---
    jac_sparsity_I = Vector{Int32}(undef, cb.nnzj)
    jac_sparsity_J = Vector{Int32}(undef, cb.nnzj)
    _jac_sparsity_wrapper!(cb, jac_sparsity_I, jac_sparsity_J)

    hess_sparsity_I, hess_sparsity_J = build_hessian_structure(cb, hessian_approximation)
    force_lower_triangular!(hess_sparsity_I, hess_sparsity_J)

    n_hess = length(hess_sparsity_I)
    n_jac = length(jac_sparsity_I)

    # --- COO value buffers ---
    hess = VT(undef, n_hess)
    jac = VT(undef, n_jac)
    fill!(hess, zero(T))
    fill!(jac, zero(T))

    # --- Build global COO + CSC ---
    hess_raw = SparseMatrixCOO(n, n, hess_sparsity_I, hess_sparsity_J, hess)
    jt_coo = SparseMatrixCOO(n, m, jac_sparsity_J, jac_sparsity_I, jac)  # transposed

    hess_csc, hess_csc_map = coo_to_csc(hess_raw)
    jt_csc, jt_csc_map = coo_to_csc(jt_coo)

    # --- Shared symbolic construction (constraints, Hessian classification, A_kk template, block_maps) ---
    sym = _build_schur_symbolic(
        T, n, m, ns, nv, nd, nc,
        hess_sparsity_I, hess_sparsity_J,
        jac_sparsity_I, jac_sparsity_J,
        cb.ind_eq, cb.ind_ineq,
    )
    nc_eq_per_s = sym.nc_eq_per_s
    nc_ineq_per_s = sym.nc_ineq_per_s
    blk_size = sym.blk_size
    block_maps = sym.block_maps

    # Per-scenario A_kk: independent copies of the shared template (same sparsity, fresh nzval).
    A_kk_vec = [copy(sym.akk_csc_template) for _ in 1:ns]

    # --- Dense matrices ---
    aug_com = Matrix{T}(undef, nd, nd)
    C_dk = [Matrix{T}(undef, nd, blk_size) for _ in 1:ns]
    tmp_blk_nd = [Matrix{T}(undef, blk_size, nd) for _ in 1:ns]

    # --- Diagonal vectors ---
    reg     = VT(undef, n + ns_ineq)
    pr_diag = VT(undef, n + ns_ineq)
    du_diag = VT(undef, m)
    l_diag  = fill!(VT(undef, nlb), one(T))
    u_diag  = fill!(VT(undef, nub), one(T))
    l_lower = fill!(VT(undef, nlb), zero(T))
    u_lower = fill!(VT(undef, nub), zero(T))

    # --- Buffers ---
    diag_buffer = VT(undef, ns_ineq)
    buffer      = VT(undef, m)
    # Size from the per-scenario invariant rather than the global eq count: the
    # uniform-scenario validation in `_build_schur_symbolic` guarantees these
    # are equal, but tying the buffer to `ns * nc_eq_per_s` keeps the
    # round-trip in `solve_kkt!` (gather/scatter via the eq index list) shape-
    # consistent by construction.
    wy_eq_buf   = VT(undef, ns * nc_eq_per_s)
    rhs_d       = VT(undef, nd)
    rhs_k       = [VT(undef, blk_size) for _ in 1:ns]
    solve_buffers = [VT(undef, blk_size) for _ in 1:ns]

    # --- Init ---
    fill!(aug_com, zero(T))
    fill!(pr_diag, zero(T))
    fill!(du_diag, zero(T))

    # --- Create solvers ---
    quasi_newton = create_quasi_newton(hessian_approximation, cb, n; options=qn_options)
    scenario_solvers = [schur_scenario_linear_solver(A_kk_vec[k]) for k in 1:ns]
    _linear_solver = linear_solver(aug_com; opt = opt_linear_solver)

    return SchurComplementKKTSystem(
        hess, jac,
        hess_raw, jt_coo,
        hess_csc, hess_csc_map, jt_csc, jt_csc_map,
        quasi_newton,
        reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower,
        ns, nv, nd, nc,
        nc_eq_per_s, nc_ineq_per_s, blk_size,
        A_kk_vec, C_dk,
        aug_com,
        diag_buffer, buffer, wy_eq_buf, rhs_d, rhs_k, tmp_blk_nd, solve_buffers,
        block_maps,
        sym.hess_S_coo, sym.hess_S_row, sym.hess_S_col,
        sym.eq_global_flat,
        n_eq, cb.ind_eq,
        ns_ineq, cb.ind_ineq, cb.ind_lb, cb.ind_ub,
        scenario_solvers,
        _linear_solver,
    )
end

num_variables(kkt::SchurComplementKKTSystem) = size(kkt.hess_csc, 1)

function get_slack_regularization(kkt::SchurComplementKKTSystem)
    n = num_variables(kkt)
    ns_ineq = kkt.n_ineq
    return view(kkt.pr_diag, n+1:n+ns_ineq)
end

function is_inertia_correct(kkt::SchurComplementKKTSystem, num_pos, num_zero, num_neg)
    return (num_zero == 0) && (num_pos == size(kkt.aug_com, 1))
end
# the Schur complement is definite: the duals are eliminated, not factorized
block_structure(kkt::SchurComplementKKTSystem) = (size(kkt.aug_com, 1), 0)

should_regularize_dual(kkt::SchurComplementKKTSystem, num_pos, num_zero, num_neg) = true

function jtprod!(y::AbstractVector, kkt::SchurComplementKKTSystem, x::AbstractVector)
    nx = num_variables(kkt)
    ns_ineq = kkt.n_ineq
    yx = view(y, 1:nx)
    ys = view(y, 1+nx:nx+ns_ineq)
    mul!(yx, kkt.jt_csc, x)
    ys .= -@view(x[kkt.ind_ineq])
    return
end

function compress_jacobian!(kkt::SchurComplementKKTSystem)
    transfer!(kkt.jt_csc, kkt.jt_coo, kkt.jt_csc_map)
end

function compress_hessian!(kkt::SchurComplementKKTSystem)
    transfer!(kkt.hess_csc, kkt.hess_raw, kkt.hess_csc_map)
end

nnz_jacobian(kkt::SchurComplementKKTSystem) = nnz(kkt.jt_coo)

function build_kkt!(kkt::SchurComplementKKTSystem{T, VT, MT}) where {T, VT, MT}
    ns = kkt.ns
    nv = kkt.nv
    nd = kkt.nd
    n = num_variables(kkt)
    blk = kkt.blk_size

    # Compute condensing diagonal for inequalities
    if kkt.n_ineq > 0
        Sigma_s = view(kkt.pr_diag, n+1:n+kkt.n_ineq)
        Sigma_d = @view(kkt.du_diag[kkt.ind_ineq])
        kkt.diag_buffer .= Sigma_s ./ (one(T) .- Sigma_d .* Sigma_s)
    end

    # Initialize Schur complement S = H_dd + diag(pr_diag_dd)
    S = kkt.aug_com
    fill!(S, zero(T))
    _scatter_add!(S, kkt.hess, kkt.hess_S_coo, kkt.hess_S_row, kkt.hess_S_col)
    @inbounds for i in 1:nd
        S[i, i] += kkt.pr_diag[ns*nv+i]
    end

    # Phase 1 (parallel): assemble per-scenario blocks, factorize, compute A_kk^{-1} * C_dk'.
    # `@blas_safe_threads` runs `Threads.@threads` over scenarios while pinning
    # BLAS to a single thread per task to avoid oversubscription with the
    # per-scenario `mul!` / `factorize!` calls inside the loop.
    @blas_safe_threads for k in 1:ns
        bm = kkt.block_maps[k]
        A_kk = kkt.A_kk[k]
        C_dk = kkt.C_dk[k]
        nz = A_kk.nzval

        fill!(nz, zero(T))
        fill!(C_dk, zero(T))

        _scatter_add!(nz,   kkt.hess,    bm.hess_Akk_coo,    bm.hess_Akk_nzpos)
        _scatter_add!(C_dk, kkt.hess,    bm.hess_Cdk_coo,    bm.hess_Cdk_row, bm.hess_Cdk_col)
        _scatter_add!(nz,   kkt.pr_diag, bm.pr_diag_global,  bm.pr_diag_nzpos)
        _scatter_add!(nz,   kkt.du_diag, bm.du_diag_global,  bm.du_diag_nzpos)
        _scatter_add!(nz,   kkt.jac,     bm.jeq_Akk_coo,     bm.jeq_Akk_nzpos)
        _scatter_add!(C_dk, kkt.jac,     bm.jeq_Cdk_coo,     bm.jeq_Cdk_row, bm.jeq_Cdk_col)
        _scatter_quad_add!(nz,   kkt.jac, kkt.diag_buffer,
                           bm.ineq_Akk_nzpos, bm.ineq_Akk_jcoo1, bm.ineq_Akk_jcoo2, bm.ineq_Akk_bufidx)
        _scatter_quad_add!(C_dk, kkt.jac, kkt.diag_buffer,
                           bm.ineq_Cdk_row, bm.ineq_Cdk_col,
                           bm.ineq_Cdk_jcoo_d, bm.ineq_Cdk_jcoo_v, bm.ineq_Cdk_bufidx)

        # Factor A_kk
        factorize!(kkt.scenario_solvers[k])

        # Compute tmp = A_kk^{-1} * C_dk'  (blk × nd)
        buf = kkt.solve_buffers[k]
        for j in 1:nd
            @inbounds for i in 1:blk
                buf[i] = C_dk[j, i]  # C_dk' column j
            end
            solve_linear_system!(kkt.scenario_solvers[k], buf)
            @inbounds for i in 1:blk
                kkt.tmp_blk_nd[k][i, j] = buf[i]
            end
        end
    end

    # Phase 2 (sequential): accumulate into shared Schur complement S
    for k in 1:ns
        bm = kkt.block_maps[k]
        _scatter_quad_add!(S, kkt.jac, kkt.diag_buffer,
                           bm.ineq_S_row, bm.ineq_S_col,
                           bm.ineq_S_jcoo1, bm.ineq_S_jcoo2, bm.ineq_S_bufidx)
        # S -= C_dk * A_kk^{-1} * C_dk'
        mul!(S, kkt.C_dk[k], kkt.tmp_blk_nd[k], -one(T), one(T))
    end

    return
end

function factorize_kkt!(kkt::SchurComplementKKTSystem)
    return factorize!(kkt.linear_solver)
end

function solve_kkt!(
    kkt::SchurComplementKKTSystem,
    w::AbstractKKTVector{T},
) where T

    ns = kkt.ns
    nv = kkt.nv
    nd = kkt.nd
    nc = kkt.nc
    n = num_variables(kkt)
    blk = kkt.blk_size

    wx = _madnlp_unsafe_wrap(full(w), n)
    ws = view(full(w), n+1:n+kkt.n_ineq)
    wy = dual(w)

    Sigma_s = get_slack_regularization(kkt)

    reduce_rhs!(kkt, w)

    # Step 1: condense inequality contributions
    fill!(kkt.buffer, zero(T))
    if kkt.n_ineq > 0
        kkt.buffer[kkt.ind_ineq] .= kkt.diag_buffer .* (wy[kkt.ind_ineq] .+ ws ./ Sigma_s)
        # J' * buffer → wx (using sparse jt_csc)
        mul!(wx, kkt.jt_csc, kkt.buffer, one(T), one(T))
    end

    # Step 2: Extract per-scenario RHS blocks
    # NOTE: writes to disjoint slices of wx/wy → safe to @blas_safe_threads,
    # but per-iteration work is just a few scalar copies; profile before threading.
    nc_eq = kkt.nc_eq_per_s
    @inbounds for k in 1:ns
        vr_start = (k-1)*nv
        rhs = kkt.rhs_k[k]
        for i in 1:nv
            rhs[i] = wx[vr_start + i]
        end
        eq_base = (k-1)*nc_eq
        for ci in 1:nc_eq
            rhs[nv+ci] = wy[kkt.eq_global_indices[eq_base + ci]]
        end
    end
    @inbounds for i in 1:nd
        kkt.rhs_d[i] = wx[ns*nv+i]
    end

    # Step 3: Forward elimination
    # Phase 1 (parallel): solve per-scenario systems
    @blas_safe_threads for k in 1:ns
        solve_linear_system!(kkt.scenario_solvers[k], kkt.rhs_k[k])
    end
    # Phase 2 (sequential): accumulate into shared rhs_d
    for k in 1:ns
        mul!(kkt.rhs_d, kkt.C_dk[k], kkt.rhs_k[k], -one(T), one(T))
    end

    # Step 4: Solve Schur complement
    solve_linear_system!(kkt.linear_solver, kkt.rhs_d)

    # Step 5: Back-substitution (parallel — reads shared rhs_d, writes per-scenario rhs_k)
    @blas_safe_threads for k in 1:ns
        mul!(kkt.rhs_k[k], kkt.tmp_blk_nd[k], kkt.rhs_d, -one(T), one(T))
    end

    # Step 6: Write back to w (same threading note as Step 2 above)
    @inbounds for k in 1:ns
        vr_start = (k-1)*nv
        rhs = kkt.rhs_k[k]
        for i in 1:nv
            wx[vr_start + i] = rhs[i]
        end
        eq_base = (k-1)*nc_eq
        for ci in 1:nc_eq
            wy[kkt.eq_global_indices[eq_base + ci]] = rhs[nv+ci]
        end
    end
    @inbounds for i in 1:nd
        wx[ns*nv+i] = kkt.rhs_d[i]
    end

    # Step 7: Recover inequality duals and slacks
    if kkt.n_ineq > 0
        # Stash eq duals; mul! below overwrites all of wy
        copyto!(kkt.wy_eq_buf, view(wy, kkt.ind_eq))

        # J * Δx via sparse: (jt_csc)' * wx
        mul!(wy, kkt.jt_csc', wx)

        # Restore equality duals
        view(wy, kkt.ind_eq) .= kkt.wy_eq_buf

        # Inequality dual recovery
        @inbounds for idx in 1:length(kkt.ind_ineq)
            gi = kkt.ind_ineq[idx]
            wy[gi] = kkt.diag_buffer[idx] * wy[gi] - kkt.buffer[gi]
        end
        ws .= (ws .+ view(wy, kkt.ind_ineq)) ./ Sigma_s
    end

    finish_aug_solve!(kkt, w)
    return w
end

# KKT matrix-vector product for iterative refinement
function mul!(w::AbstractKKTVector{T}, kkt::SchurComplementKKTSystem{T}, x::AbstractKKTVector, alpha = one(T), beta = zero(T)) where T
    n = num_variables(kkt)
    ns_ineq = kkt.n_ineq
    wx = @view(primal(w)[1:n])
    ws = @view(primal(w)[n+1:end])
    wy = dual(w)

    xx = @view(primal(x)[1:n])
    xs = @view(primal(x)[n+1:end])
    xy = dual(x)
    xz = @view(dual(x)[kkt.ind_ineq])

    # H * xx → wx (using sparse symmetric Hessian)
    wx .= beta .* wx
    mul!(wx, Symmetric(kkt.hess_csc, :L), xx, alpha, one(T))

    m = size(kkt.jt_csc, 2)
    if m > 0
        mul!(wx, kkt.jt_csc, dual(x), alpha, one(T))       # J' * xy
        mul!(wy, kkt.jt_csc', xx, alpha, beta)              # J * xx
    else
        wy .= beta .* wy
    end
    ws .= beta.*ws .- alpha.* xz
    @view(dual(w)[kkt.ind_ineq]) .-= alpha.* xs
    _kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

function mul_hess_blk!(wx, kkt::SchurComplementKKTSystem, t)
    n = num_variables(kkt)
    mul!(@view(wx[1:n]), Symmetric(kkt.hess_csc, :L), @view(t[1:n]))
    fill!(@view(wx[n+1:end]), 0)
    wx .+= t .* kkt.pr_diag
end
