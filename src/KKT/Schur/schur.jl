
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
end

"""
    SchurComplementCondensedKKTSystem{T, VT, MT, QN, LS, LS2, VI} <: AbstractCondensedKKTSystem{T, VT, MT, QN}

KKT system exploiting block-arrowhead structure from two-stage stochastic programs
via Schur complement decomposition, using sparse COO/CSC storage for the global
Hessian and Jacobian and sparse per-scenario block solvers.

Variable layout: `[v_1, ..., v_ns, d]` where `v_k ∈ R^nv`, `d ∈ R^nd`.
Constraint layout: `[c_1, ..., c_ns]` where `c_k ∈ R^nc`.

All equality constraints are relaxed to inequalities (`RelaxEquality`) and
condensed, so the per-scenario block `A_k` (size `nv × nv`) and the first-stage
Schur complement `schur_csc` (size `nd × nd`) are both symmetric *positive
definite* — there is no bordered equality saddle. `A_k` is a sparse
lower-triangular `SparseMatrixCSC` (`= H_k + pr_diag + Σ_i J_i' D_i J_i`) factored
by a configurable sparse solver (default `MumpsSolver`). The coupling blocks
`C_dk` remain dense. The reduction `Σ_k C_dk A_kk⁻¹ C_dk'` fills only the `m×m`
coupled-design block of `schur_csc`, factored by a sparse symmetric solver
(`MumpsSolver` by default; any `:csc` symmetric solver that reports inertia,
e.g. `Ma57Solver`/`LDLSolver`, works).
"""
struct SchurComplementCondensedKKTSystem{
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
    nc_ineq_per_s::Int
    blk_size::Int                   # = nv (per-scenario condensed block size)

    # Per-scenario sparse augmented blocks (lower triangle only)
    A_kk::Vector{SparseMatrixCSC{T, Int32}}
    C_dk::Vector{MT}                # ns × (m × blk_size) — dense, reduced to the m COUPLED
                                    # design rows (design vars that never couple are exactly zero)

    # Sparse first-stage Schur complement (lower-triangular CSC, size nd × nd): the
    # design block S_dd plus the condensed design inequalities. SPD. Only `m` design
    # vars couple to a scenario, so the reduction `Σ_k C_dk A_kk⁻¹ C_dk'` fills only
    # the `m×m` coupled-design block.
    schur_csc::SparseMatrixCSC{T, Int32}
    m::Int                                # coupled design vars (Schur fill width)
    coupled_design_local::Vector{Int}     # length m — design-local indices that couple
    schur_fill_nzpos::Matrix{Int}         # (m, m) — nzval positions of the Schur-fill block
    schur_block::MT                       # (m, m) buffer — per-scenario C_dk_red' A_kk⁻¹ C_dk_red'

    # Buffers
    diag_buffer::VT                 # n_ineq — condensing diagonal
    buffer::VT                      # m_total — general
    rhs_d::VT                       # nd — design vars
    rhs_d_red::VT                   # m — reduced (coupled) design RHS accumulator
    rhs_k::Vector{VT}              # ns × blk_size — scenario RHS buffers
    tmp_blk_nd::Vector{MT}         # ns × (blk_size × m) — A_kk⁻¹ C_dk_red' (reduced coupled cols)
    solve_buffers::Vector{VT}      # ns × blk_size — per-scenario column-by-column solve buffers

    block_maps::Vector{ScenarioBlockMap}

    # Sparse-Schur scatter maps: scatter into `schur_csc.nzval` at precomputed nzval
    # positions (lower-triangle; the symmetric solver reads the lower triangle). Lists
    # are lower-only so each slot is hit once.
    schur_hess_coo::Vector{Int}           # design Hessian: COO index into `hess`
    schur_hess_nzpos::Vector{Int}         # → nzval slot
    schur_diag_nzpos::Vector{Int}         # pr_diag design diagonal (paired with design_var_global)
    schur_ineq_S_nzpos::Vector{Int}       # scenario ineq → S (flat over all scenarios)
    schur_ineq_S_jcoo1::Vector{Int}
    schur_ineq_S_jcoo2::Vector{Int}
    schur_ineq_S_bufidx::Vector{Int}
    schur_design_ineq_S_nzpos::Vector{Int}    # design ineq → S
    schur_design_ineq_S_jcoo1::Vector{Int}
    schur_design_ineq_S_jcoo2::Vector{Int}
    schur_design_ineq_S_bufidx::Vector{Int}

    # Tag-driven global index lists (design/scenario vars need not be contiguous).
    design_var_global::Vector{Int}        # length nd — global index of each design var
    scen_var_global::Vector{Vector{Int}}  # ns × nv — global index of each scenario var

    # Inequality/equality/bound index info (n_eq == 0 / ind_eq empty under RelaxEquality,
    # kept for the generic KKT interface).
    n_eq::Int
    ind_eq::VI
    n_ineq::Int
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI

    # Aggregated inertia of the per-scenario blocks A_k, refreshed by `build_kkt!`.
    # Haynsworth additivity gives In(K) = Σ_k In(A_k) + In(S), but the IPM only ever
    # queries the linear solver for In(S) (`solver.jl`: `inertia(kkt.linear_solver)`).
    # `is_inertia_correct` reads these so the test covers all of In(K).
    block_num_neg::Base.RefValue{Int}     # Σ_k #negative eigenvalues of A_k
    block_num_zero::Base.RefValue{Int}    # Σ_k #zero eigenvalues of A_k
    block_num_indef::Base.RefValue{Int}   # #{k : A_k not positive definite} — diagnostics

    # Solvers
    scenario_solvers::Vector{LS2}
    linear_solver::LS               # for Schur complement S (dense)
end

"""
    _schur_tags_from_callback(cb) -> (ns, var_scen, con_scen) | nothing

Read per-variable/per-constraint scenario-assignment vectors from the NLP model
behind `cb`, if it exposes a two-stage tag. Supports the ExaModels convention
(`cb.nlp.tag::TwoStageExaModelTag` with fields `nscen`, `var_scen`, `con_scen`)
and a legacy/test interface (`cb.nlp.tags` with `ns`, `var_scenario`,
`con_scenario`). Returns `nothing` when no recognizable tag is present.

In both conventions `var_scen[i] == 0` flags a design variable and `== k` flags
scenario `k`; same encoding for constraints.
"""
function _schur_tags_from_callback(cb)
    nlp = cb.nlp
    if hasproperty(nlp, :tag)
        tag = nlp.tag
        if hasproperty(tag, :nscen) && hasproperty(tag, :var_scen) && hasproperty(tag, :con_scen)
            return Int(tag.nscen), Vector{Int}(Array(tag.var_scen)), Vector{Int}(Array(tag.con_scen))
        end
    end
    if hasproperty(nlp, :tags)
        tags = nlp.tags
        if hasproperty(tags, :ns) && hasproperty(tags, :var_scenario) && hasproperty(tags, :con_scenario)
            return Int(tags.ns), Vector{Int}(Array(tags.var_scenario)), Vector{Int}(Array(tags.con_scenario))
        end
    end
    return nothing
end

"""
    _resolve_schur_dims(cb, n, m, schur_ns, schur_nv, schur_nd, schur_nc,
                        schur_var_scen=nothing, schur_con_scen=nothing)
    -> (; ns, nv, nd, nc, nc_design, var_scen, con_scen)

Resolve two-stage stochastic dimensions AND the per-variable/per-constraint
scenario-assignment vectors for a `SchurComplementCondensedKKTSystem`. The vectors encode
`var_scen[i] ∈ {0..ns}` (0 = design) and `con_scen[j] ∈ {0..ns}`; the symbolic
build partitions the Hessian/Jacobian by these tags rather than by index
arithmetic, so design variables and a scenario's variables need NOT be contiguous
in the global ordering.

Resolution priority:
1. explicit `schur_var_scen` / `schur_con_scen` (passed via `kkt_options`);
2. else a two-stage tag on the model (see [`_schur_tags_from_callback`](@ref));
3. else synthesize the contiguous layout `[v_1..v_ns, d]` / `[c_1..c_ns]` from
   `schur_ns/nv/nd/nc` (backward-compatible with hand-built two-stage models).

Design-only constraints (`con_scen == 0`) are supported; `nc_design` counts them.
Asserts consistency with `n` and `m`, and per-scenario uniformity.
"""
function _resolve_schur_dims(
        cb, n, m, schur_ns, schur_nv, schur_nd, schur_nc,
        schur_var_scen = nothing, schur_con_scen = nothing,
    )
    var_scen = nothing
    con_scen = nothing
    ns = schur_ns

    if schur_var_scen !== nothing && schur_con_scen !== nothing
        var_scen = Vector{Int}(Array(schur_var_scen))
        con_scen = Vector{Int}(Array(schur_con_scen))
        ns = schur_ns > 0 ? schur_ns : maximum(var_scen)
    elseif schur_ns == 0
        tag_info = _schur_tags_from_callback(cb)
        if tag_info !== nothing
            ns, var_scen, con_scen = tag_info
        end
    end

    if var_scen === nothing
        # No tags/vectors available: synthesize the contiguous layout from the
        # explicit per-stage dimensions. Reproduces the legacy index-arithmetic
        # behaviour exactly (design variables last, scenario stripes contiguous).
        @assert schur_ns > 0 "schur_ns must be specified and positive (or pass schur_var_scen/schur_con_scen, or use a two-stage tag for auto-detection)"
        @assert schur_nv > 0 "schur_nv must be specified and positive"
        @assert schur_nd > 0 "schur_nd must be specified and positive"
        ns = schur_ns
        var_scen = vcat((fill(k, schur_nv) for k in 1:ns)..., fill(0, schur_nd))
        con_scen = vcat((fill(k, schur_nc) for k in 1:ns)...)
    end

    @assert ns > 0 "resolved scenario count must be positive"
    length(var_scen) == n || error("var_scen has length $(length(var_scen)); expected n=$n")
    length(con_scen) == m || error("con_scen has length $(length(con_scen)); expected m=$m")

    # Single-pass histograms over the scenario tags. Index 1 is the design bucket
    # (tag 0); 1+k is scenario k. Validates per-scenario uniformity in the same
    # pass — a malformed model whose global aggregates happen to satisfy
    # n == ns*nv + nd would otherwise drive the symbolic build into garbage.
    var_hist = zeros(Int, ns + 1)
    for t in var_scen
        (0 <= t <= ns) || error("var_scen tag $t out of range [0, $ns]; 0 = design, 1..$ns = scenario index.")
        @inbounds var_hist[t + 1] += 1
    end
    con_hist = zeros(Int, ns + 1)
    for t in con_scen
        (0 <= t <= ns) || error("con_scen tag $t out of range [0, $ns]; 0 = design, 1..$ns = scenario index.")
        @inbounds con_hist[t + 1] += 1
    end

    nd = var_hist[1]
    nv = var_hist[2]
    nc = con_hist[2]
    nc_design = con_hist[1]

    for k in 2:ns
        @inbounds nv_k = var_hist[k + 1]
        @inbounds nc_k = con_hist[k + 1]
        nv_k == nv || error(
            "Scenario $k has $nv_k variables; scenario 1 has $nv. " *
                "SchurComplementCondensedKKTSystem requires uniform per-scenario sizes."
        )
        nc_k == nc || error(
            "Scenario $k has $nc_k constraints; scenario 1 has $nc. " *
                "SchurComplementCondensedKKTSystem requires uniform per-scenario sizes."
        )
    end

    @assert nv > 0 "resolved per-scenario variable count nv must be positive"
    @assert nd > 0 "resolved design variable count nd must be positive"
    @assert n == ns * nv + nd "Variable count mismatch: n=$n != ns*nv+nd=$(ns * nv + nd)"
    @assert m == ns * nc + nc_design "Constraint count mismatch: m=$m != ns*nc+nc_design=$(ns * nc + nc_design)"

    return (; ns, nv, nd, nc, nc_design, var_scen, con_scen)
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

Pure-CPU symbolic construction shared by the CPU `SchurComplementCondensedKKTSystem`
and GPU `GPUSchurComplementCondensedKKTSystem` constructors. Inputs are CPU-resident
sparsity arrays (the GPU side downloads its sparsity before calling).

**Assumes uniform per-scenario structure** (same A_kk pattern for every k).
This matches typical two-stage stochastic models and is what the GPU batched
cuDSS path requires; CPU follows the same assumption.

RelaxEquality-only: `ind_eq` must be empty (all constraints are inequalities),
so the per-scenario blocks and the first-stage Schur complement are condensed/SPD.

Returns a NamedTuple with:
- `ineq_per_scenario` — `Vector{Vector{Int}}` of global inequality indices
- `nc_ineq_per_s`, `blk_size` — per-scenario sizes (`blk_size == nv`)
- `block_maps::Vector{ScenarioBlockMap}` — per-scenario index maps
- `hess_S_coo`, `hess_S_row`, `hess_S_col` — design-design Hessian COO maps
- `akk_csc_template::SparseMatrixCSC{T, Int32}` — shared A_kk sparsity (zero values)
- `nnz_per_scenario::Int`
- the sparse-Schur pattern (`schur_csc_*`, `schur_*_nzpos`, `coupled_design_local`, …)
"""
function _build_schur_symbolic(
        ::Type{T},
        n::Int, m::Int, ns::Int, nv::Int, nd::Int, nc::Int,
        hess_I::AbstractVector{<:Integer}, hess_J::AbstractVector{<:Integer},
        jac_I::AbstractVector{<:Integer}, jac_J::AbstractVector{<:Integer},
        ind_eq::AbstractVector{<:Integer},
        ind_ineq::AbstractVector{<:Integer},
        var_scen = nothing,
        con_scen = nothing,
    ) where {T}

    n_hess = length(hess_I)
    n_jac = length(jac_I)

    # Synthesize the contiguous tag layout (`[v_1..v_ns, d]`, `[c_1..c_ns]`) when
    # no tags are supplied. This reproduces the legacy index-arithmetic behaviour
    # exactly, so hand-built two-stage models and the symbolic unit tests keep
    # working unchanged.
    if var_scen === nothing
        var_scen = vcat((fill(k, nv) for k in 1:ns)..., fill(0, nd))
    end
    if con_scen === nothing
        con_scen = vcat((fill(k, nc) for k in 1:ns)...)
    end
    var_scen = Vector{Int}(Array(var_scen))
    con_scen = Vector{Int}(Array(con_scen))

    # --- Tag-driven global index lists & local maps ---
    # Partition variables/constraints by their scenario tag (0 = design) instead
    # of by contiguous index arithmetic, so design variables and a scenario's
    # variables need NOT be contiguous in the global ordering. `var_local[i]` is
    # the position of variable `i` within its own block (1:nd for design,
    # 1:nv for its scenario), assigned in increasing global-index order so it is
    # monotone within a block — which keeps the lower-triangular A_kk layout valid.
    design_var_global = Int[]                       # length nd
    scen_var_global = [Int[] for _ in 1:ns]         # each length nv
    var_local = zeros(Int, n)
    for i in 1:n
        o = var_scen[i]
        if o == 0
            push!(design_var_global, i)
            var_local[i] = length(design_var_global)
        else
            push!(scen_var_global[o], i)
            var_local[i] = length(scen_var_global[o])
        end
    end

    # --- Classify constraints per scenario / design (tag-driven) ---
    # RelaxEquality-only: every constraint is an inequality (ind_eq must be empty),
    # so the per-scenario blocks and the first-stage Schur complement are condensed
    # and SPD — there is no equality saddle / bordered block.
    isempty(ind_eq) || error(
        "SchurComplementCondensedKKTSystem is RelaxEquality-only, but got $(length(ind_eq)) " *
        "constraint(s) kept as equalities. The bordered EnforceEquality saddle that the old " *
        "`SchurComplementKKTSystem` used was removed: the first-stage Schur complement is now SPD " *
        "and requires every constraint to be relaxed into the barrier. Pass " *
        "`equality_treatment=MadNLP.RelaxEquality` (the default when " *
        "`kkt_system=SchurComplementCondensedKKTSystem`) — do not override it with `EnforceEquality`."
    )
    ind_ineq_set = Set(Int.(ind_ineq))

    ineq_per_scenario = [Int[] for _ in 1:ns]
    design_ineq = Int[]     # design-only inequality constraint global rows

    for gi in 1:m
        (gi in ind_ineq_set) || continue
        o = con_scen[gi]
        if o == 0
            push!(design_ineq, gi)
        else
            push!(ineq_per_scenario[o], gi)
        end
    end

    # Scenario 1 sets the canonical per-scenario constraint count; reject any
    # scenario that disagrees, since downstream code (and the GPU batched layout)
    # assumes uniform per-scenario shape.
    nc_ineq_per_s = length(ineq_per_scenario[1])
    for k in 2:ns
        n_in_k = length(ineq_per_scenario[k])
        if n_in_k != nc_ineq_per_s
            error(
                "SchurComplementCondensedKKTSystem requires uniform per-scenario constraint counts. " *
                "Scenario 1 has ineq=$nc_ineq_per_s; scenario $k has ineq=$n_in_k."
            )
        end
    end

    blk_size = nv

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

    # --- Classify Hessian COO entries (tag-driven) ---
    hess_S_coo = Int[]
    hess_S_row = Int[]
    hess_S_col = Int[]

    hess_per_scenario_diag = [Tuple{Int,Int,Int}[] for _ in 1:ns]      # (coo, local_i, local_j)
    hess_per_scenario_coupling = [Tuple{Int,Int,Int}[] for _ in 1:ns]  # (coo, design_local, var_local)
    hess_classified = falses(n_hess)

    for ci in 1:n_hess
        ri = Int(hess_I[ci])
        rj = Int(hess_J[ci])  # global lower triangle: ri >= rj
        oi = var_scen[ri]
        oj = var_scen[rj]

        if oi == 0 && oj == 0
            # Both design vars → S entry (write both triangles for dense S).
            di = var_local[ri]
            dj = var_local[rj]
            push!(hess_S_coo, ci); push!(hess_S_row, di); push!(hess_S_col, dj)
            if di != dj
                push!(hess_S_coo, ci); push!(hess_S_row, dj); push!(hess_S_col, di)
            end
            hess_classified[ci] = true
        elseif (oi == 0) != (oj == 0)
            # One design + one scenario var → coupling block. Either global index
            # may be the design one now (design vars are not necessarily last).
            if oi == 0
                di = var_local[ri]; k = oj; vj = var_local[rj]
            else
                di = var_local[rj]; k = oi; vj = var_local[ri]
            end
            push!(hess_per_scenario_coupling[k], (ci, di, vj))
            hess_classified[ci] = true
        elseif oi == oj
            # Both within the same scenario → A_kk diagonal block. `var_local` is
            # monotone in the global index within a scenario, so li >= lj here;
            # normalize defensively to keep the lower-triangular layout.
            k = oi
            li = var_local[ri]
            lj = var_local[rj]
            if li < lj
                li, lj = lj, li
            end
            push!(hess_per_scenario_diag[k], (ci, li, lj))
            hess_classified[ci] = true
        end
        # else: both scenario but oi != oj → cross-scenario coupling, left
        # unclassified and reported below.
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
            "$n_bad_hess Hessian COO entries do not fit the SchurComplementCondensedKKTSystem " *
            "block-arrowhead pattern (likely cross-scenario coupling). First few: " *
            details
        )
    end

    # --- Build shared A_kk template from scenario 1 ---
    ineq_cons_1 = ineq_per_scenario[1]

    akk_entries = Dict{Tuple{Int,Int}, Nothing}()

    # Hessian diagonal
    for (_, li, lj) in hess_per_scenario_diag[1]
        akk_entries[(li, lj)] = nothing
    end
    # pr_diag (always nv diagonal entries)
    for i in 1:nv
        akk_entries[(i, i)] = nothing
    end
    # Inequality condensation fill-in (lower triangle pairs of scenario vars)
    for gi in ineq_cons_1
        local_vars = Int[]
        for (_, col) in get(jac_by_constraint, gi, Tuple{Int,Int}[])
            if var_scen[col] == 1
                push!(local_vars, var_local[col])
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
            "SchurComplementCondensedKKTSystem: scenario $k has an A_kk entry at local " *
            "(row=$(key[1]), col=$(key[2])) absent from scenario 1's template. " *
            "Per-scenario Hessian/Jacobian sparsity must be uniform."
        )
        p
    end

    nnz_per_scenario = length(akk_csc_template.nzval)

    # --- Build per-scenario ScenarioBlockMaps ---
    block_maps = Vector{ScenarioBlockMap}(undef, ns)
    jac_classified = falses(n_jac)

    for k in 1:ns
        ineq_cons = ineq_per_scenario[k]

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

        # pr_diag → A_kk diagonal
        pr_diag_global_vec = Int[]
        pr_diag_nzpos_vec = Int[]
        for i in 1:nv
            push!(pr_diag_global_vec, scen_var_global[k][i])
            push!(pr_diag_nzpos_vec, akk_pos((i, i), k))
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
                oc = var_scen[col]
                if oc == k
                    push!(v_entries, (coo_idx, var_local[col]))
                    jac_classified[coo_idx] = true
                elseif oc == 0
                    push!(d_entries, (coo_idx, var_local[col]))
                    jac_classified[coo_idx] = true
                end
                # else: cross-scenario column, left unclassified and reported below.
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
            ineq_Akk_nzpos_vec, ineq_Akk_jcoo1_vec, ineq_Akk_jcoo2_vec, ineq_Akk_bufidx_vec,
            ineq_Cdk_row_vec, ineq_Cdk_col_vec, ineq_Cdk_jcoo_d_vec, ineq_Cdk_jcoo_v_vec, ineq_Cdk_bufidx_vec,
            ineq_S_row_vec, ineq_S_col_vec, ineq_S_jcoo1_vec, ineq_S_jcoo2_vec, ineq_S_bufidx_vec,
            pr_diag_global_vec, pr_diag_nzpos_vec,
        )
    end

    # --- Design-only constraint maps ---
    # Design-only constraints touch ONLY design variables; they are condensed into
    # S_dd just like scenario inequalities. Referencing a scenario variable is
    # unrepresentable.
    nc_design_ineq = length(design_ineq)

    # Design inequality condensation → S_dd (design × design, full — both triangles).
    design_ineq_S_row = Int[]
    design_ineq_S_col = Int[]
    design_ineq_S_jcoo1 = Int[]
    design_ineq_S_jcoo2 = Int[]
    design_ineq_S_bufidx = Int[]
    for gi in design_ineq
        bidx = ineq_to_bufidx[gi]
        d_entries = Tuple{Int, Int}[]
        for (coo_idx, col) in get(jac_by_constraint, gi, Tuple{Int, Int}[])
            var_scen[col] == 0 || error(
                "Design-only inequality (row $gi) references scenario variable (col $col)."
            )
            push!(d_entries, (coo_idx, var_local[col]))
            jac_classified[coo_idx] = true
        end
        for (coo_a, da) in d_entries, (coo_b, db) in d_entries
            push!(design_ineq_S_row, da)
            push!(design_ineq_S_col, db)
            push!(design_ineq_S_jcoo1, coo_a)
            push!(design_ineq_S_jcoo2, coo_b)
            push!(design_ineq_S_bufidx, bidx)
        end
    end

    # Catch Jacobian entries whose column doesn't match the constraint's own
    # scenario stripe or the design stripe (cross-scenario coupling).
    n_bad_jac = count(!, jac_classified)
    if n_bad_jac > 0
        bad = findall(!, jac_classified)
        sample = first(bad, min(5, length(bad)))
        details = join(("(row=$(Int(jac_I[ci])), col=$(Int(jac_J[ci])))" for ci in sample), ", ")
        error(
            "$n_bad_jac Jacobian COO entries do not fit the SchurComplementCondensedKKTSystem " *
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
        :ineq_Akk_nzpos, :ineq_Cdk_row, :ineq_S_row,
        :pr_diag_global,
    )
    for k in 2:ns, f in fields_to_check
        n1 = length(getfield(bm1, f))
        nk = length(getfield(block_maps[k], f))
        if nk != n1
            error(
                "SchurComplementCondensedKKTSystem requires uniform per-scenario sparsity. " *
                "Scenario 1 has $n1 entries in $f; scenario $k has $nk."
            )
        end
    end

    # ===== Static sparse-Schur pattern (shared CPU/GPU) =============================
    # The first-stage Schur complement `schur_csc` (size nd × nd) is sparse: the
    # reduction Σ_k C_dk A_kk⁻¹ C_dk' only fills the coupled-design × coupled-design
    # block (design vars that actually couple to a scenario). Compute, once: the set of
    # coupled design vars (uniform across scenarios), the lower-triangular sparsity
    # pattern, and per-contribution nzval-position maps used to assemble and factorize
    # `schur_csc` (the GPU additionally uploads them to build a CuSparseMatrixCSC).

    # (1) Coupled design vars per scenario (design-local indices appearing in C_dk).
    coupled_per_s = [Set{Int}() for _ in 1:ns]
    for k in 1:ns
        for (_, di, _) in hess_per_scenario_coupling[k]
            push!(coupled_per_s[k], di)
        end
        for di in block_maps[k].ineq_Cdk_row
            push!(coupled_per_s[k], di)
        end
    end
    for k in 2:ns
        coupled_per_s[k] == coupled_per_s[1] || error(
            "SchurComplementCondensedKKTSystem (sparse): scenario $k couples a different set of design " *
                "variables than scenario 1; the sparse Schur path requires uniform coupling."
        )
    end
    coupled_design_local = sort!(collect(coupled_per_s[1]))   # length m
    m_coupled = length(coupled_design_local)
    coupled_inv = zeros(Int, nd)                              # design-local → compact col (1:m) or 0
    for (c, di) in enumerate(coupled_design_local)
        coupled_inv[di] = c
    end

    # (2) Lower-triangular sparsity pattern: union of every contribution, folded to (max,min).
    _lo(r, c) = r >= c ? (r, c) : (c, r)
    schur_set = Dict{Tuple{Int, Int}, Nothing}()
    for t in eachindex(hess_S_row)                                  # design Hessian
        schur_set[_lo(hess_S_row[t], hess_S_col[t])] = nothing
    end
    for i in 1:nd                                                  # design pr_diag diagonal
        schur_set[(i, i)] = nothing
    end
    for k in 1:ns                                                  # scenario ineq → S (positions uniform)
        bm = block_maps[k]
        for t in eachindex(bm.ineq_S_row)
            schur_set[_lo(bm.ineq_S_row[t], bm.ineq_S_col[t])] = nothing
        end
    end
    for t in eachindex(design_ineq_S_row)                          # design ineq → S
        schur_set[_lo(design_ineq_S_row[t], design_ineq_S_col[t])] = nothing
    end
    for a in coupled_design_local, b in coupled_design_local       # Schur fill block
        schur_set[_lo(a, b)] = nothing
    end

    schur_nnz = length(schur_set)
    schur_I = Vector{Int32}(undef, schur_nnz)
    schur_J = Vector{Int32}(undef, schur_nnz)
    let t = 0
        for ((r, c), _) in schur_set
            @assert r >= c "sparse Schur pattern entry ($r,$c) is not lower-triangular"
            t += 1
            schur_I[t] = Int32(r)
            schur_J[t] = Int32(c)
        end
    end
    schur_csc_template, schur_coo_map = coo_to_csc(
        SparseMatrixCOO(nd, nd, schur_I, schur_J, zeros(T, schur_nnz))
    )
    # (row,col) → nzval position, from the canonical (column-major) CSC ordering.
    schur_pos = Dict{Tuple{Int, Int}, Int}()
    for t in 1:schur_nnz
        schur_pos[(Int(schur_I[t]), Int(schur_J[t]))] = Int(schur_coo_map[t])
    end
    _nzpos(r, c) = schur_pos[_lo(r, c)]

    # (3) Per-contribution nzpos maps, lower-triangle only (each lower slot hit once;
    #     cuDSS reads the lower triangle and symmetrizes).
    schur_hess_coo = Int[]                                         # design Hessian
    schur_hess_nzpos = Int[]
    for t in eachindex(hess_S_row)
        hess_S_row[t] >= hess_S_col[t] || continue                 # keep lower only
        push!(schur_hess_coo, hess_S_coo[t])
        push!(schur_hess_nzpos, _nzpos(hess_S_row[t], hess_S_col[t]))
    end
    schur_diag_nzpos = Int[_nzpos(i, i) for i in 1:nd]             # pr_diag (with design_var_global)

    schur_ineq_S_nzpos = Int[]                                     # scenario ineq → S (flat, lower only)
    schur_ineq_S_jcoo1 = Int[]
    schur_ineq_S_jcoo2 = Int[]
    schur_ineq_S_bufidx = Int[]
    for k in 1:ns
        bm = block_maps[k]
        for t in eachindex(bm.ineq_S_row)
            bm.ineq_S_row[t] >= bm.ineq_S_col[t] || continue
            push!(schur_ineq_S_nzpos, _nzpos(bm.ineq_S_row[t], bm.ineq_S_col[t]))
            push!(schur_ineq_S_jcoo1, bm.ineq_S_jcoo1[t])
            push!(schur_ineq_S_jcoo2, bm.ineq_S_jcoo2[t])
            push!(schur_ineq_S_bufidx, bm.ineq_S_bufidx[t])
        end
    end

    schur_design_ineq_S_nzpos = Int[]                             # design ineq → S (lower only)
    schur_design_ineq_S_jcoo1 = Int[]
    schur_design_ineq_S_jcoo2 = Int[]
    schur_design_ineq_S_bufidx = Int[]
    for t in eachindex(design_ineq_S_row)
        design_ineq_S_row[t] >= design_ineq_S_col[t] || continue
        push!(schur_design_ineq_S_nzpos, _nzpos(design_ineq_S_row[t], design_ineq_S_col[t]))
        push!(schur_design_ineq_S_jcoo1, design_ineq_S_jcoo1[t])
        push!(schur_design_ineq_S_jcoo2, design_ineq_S_jcoo2[t])
        push!(schur_design_ineq_S_bufidx, design_ineq_S_bufidx[t])
    end

    # (4) Schur-fill nzpos (m×m); column-major flat, lower-or-diagonal entries valid.
    schur_fill_nzpos = zeros(Int, m_coupled, m_coupled)
    for a in 1:m_coupled, b in 1:m_coupled
        schur_fill_nzpos[a, b] = _nzpos(coupled_design_local[a], coupled_design_local[b])
    end

    return (
        ineq_per_scenario = ineq_per_scenario,
        nc_ineq_per_s = nc_ineq_per_s,
        blk_size = blk_size,
        block_maps = block_maps,
        hess_S_coo = hess_S_coo,
        hess_S_row = hess_S_row,
        hess_S_col = hess_S_col,
        akk_csc_template = akk_csc_template,
        nnz_per_scenario = nnz_per_scenario,
        # Tag-driven global index lists (replace the old contiguous arithmetic).
        var_scen = var_scen,
        con_scen = con_scen,
        design_var_global = design_var_global,
        scen_var_global = scen_var_global,
        # Design-only constraint block.
        nc_design_ineq = nc_design_ineq,
        design_ineq_S_row = design_ineq_S_row,
        design_ineq_S_col = design_ineq_S_col,
        design_ineq_S_jcoo1 = design_ineq_S_jcoo1,
        design_ineq_S_jcoo2 = design_ineq_S_jcoo2,
        design_ineq_S_bufidx = design_ineq_S_bufidx,
        # Sparse-Schur (GPU cuDSS) pattern + nzpos maps.
        m_coupled = m_coupled,
        coupled_design_local = coupled_design_local,
        coupled_inv = coupled_inv,
        schur_csc_colptr = schur_csc_template.colptr,
        schur_csc_rowval = schur_csc_template.rowval,
        schur_nnz = length(schur_csc_template.nzval),
        schur_hess_coo = schur_hess_coo,
        schur_hess_nzpos = schur_hess_nzpos,
        schur_diag_nzpos = schur_diag_nzpos,
        schur_ineq_S_nzpos = schur_ineq_S_nzpos,
        schur_ineq_S_jcoo1 = schur_ineq_S_jcoo1,
        schur_ineq_S_jcoo2 = schur_ineq_S_jcoo2,
        schur_ineq_S_bufidx = schur_ineq_S_bufidx,
        schur_design_ineq_S_nzpos = schur_design_ineq_S_nzpos,
        schur_design_ineq_S_jcoo1 = schur_design_ineq_S_jcoo1,
        schur_design_ineq_S_jcoo2 = schur_design_ineq_S_jcoo2,
        schur_design_ineq_S_bufidx = schur_design_ineq_S_bufidx,
        schur_fill_nzpos = schur_fill_nzpos,
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
    ::Type{SchurComplementCondensedKKTSystem},
    cb::SparseCallback{T,VT},
    linear_solver::Type;
    opt_linear_solver=default_options(linear_solver),
    hessian_approximation=ExactHessian,
    qn_options=QuasiNewtonOptions(),
    schur_ns::Int=0,
    schur_nv::Int=0,
    schur_nd::Int=0,
    schur_nc::Int=0,
        schur_var_scen = nothing,
        schur_con_scen = nothing,
    schur_scenario_linear_solver::Type=MumpsSolver,
    schur_scenario_opt_linear_solver=default_options(schur_scenario_linear_solver),
    schur_opt_linear_solver=nothing,
    kwargs...,
) where {T, VT}

    isempty(kwargs) || Base.@warn(
        "SchurComplementCondensedKKTSystem (CPU) ignores unsupported kkt_options: " *
            join(string.(keys(kwargs)), ", ")
    )
    schur_opt_linear_solver === nothing || Base.@warn(
        "SchurComplementCondensedKKTSystem (CPU): `schur_opt_linear_solver` is a GPU-only option " *
            "(first-stage cuDSS options) and is ignored on CPU; the first-stage Schur complement " *
            "is solved by `linear_solver` with `opt_linear_solver`."
    )

    n = cb.nvar
    m = cb.ncon
    ns_ineq = length(cb.ind_ineq)
    n_eq = m - ns_ineq
    nlb = length(cb.ind_lb)
    nub = length(cb.ind_ub)

    dims = _resolve_schur_dims(cb, n, m, schur_ns, schur_nv, schur_nd, schur_nc, schur_var_scen, schur_con_scen)
    ns, nv, nd, nc = dims.ns, dims.nv, dims.nd, dims.nc

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
        Array(cb.ind_eq), Array(cb.ind_ineq),
        dims.var_scen, dims.con_scen,
    )
    nc_ineq_per_s = sym.nc_ineq_per_s
    blk_size = sym.blk_size
    m_coupled = sym.m_coupled
    coupled_inv = sym.coupled_inv               # design-local → compact coupled col (1:m) or 0

    # Reduce the per-scenario coupling maps to the m COUPLED design rows: remap each C_dk row
    # (a design-local index in 1:nd) to its compact column in 1:m. Every C_dk row is coupled by
    # construction, so `coupled_inv` is nonzero there. Mirrors the GPU reduction — the coupling
    # blocks and their solves are width m, not nd, which is what makes SCOPF-shaped problems
    # (nd ≫ m) tractable on CPU.
    block_maps = [
        let bm = sym.block_maps[k]
            ScenarioBlockMap(
                bm.hess_Akk_coo, bm.hess_Akk_nzpos,
                bm.hess_Cdk_coo, coupled_inv[bm.hess_Cdk_row], bm.hess_Cdk_col,
                bm.ineq_Akk_nzpos, bm.ineq_Akk_jcoo1, bm.ineq_Akk_jcoo2, bm.ineq_Akk_bufidx,
                coupled_inv[bm.ineq_Cdk_row], bm.ineq_Cdk_col, bm.ineq_Cdk_jcoo_d, bm.ineq_Cdk_jcoo_v, bm.ineq_Cdk_bufidx,
                bm.ineq_S_row, bm.ineq_S_col, bm.ineq_S_jcoo1, bm.ineq_S_jcoo2, bm.ineq_S_bufidx,
                bm.pr_diag_global, bm.pr_diag_nzpos,
            )
        end for k in 1:ns
    ]

    # Per-scenario A_kk: independent copies of the shared template (same sparsity, fresh nzval).
    A_kk_vec = [copy(sym.akk_csc_template) for _ in 1:ns]

    # --- Sparse Schur complement + reduced dense coupling blocks ---
    # schur_csc is the first-stage block (lower-triangular, size nd). The reduction
    # `Σ_k C_dk A_kk⁻¹ C_dk'` fills only the m×m coupled block, so C_dk is stored reduced to its
    # m coupled design rows (m × blk_size) and tmp_blk_nd = A_kk⁻¹ C_dk_red' is (blk_size × m).
    schur_csc = SparseMatrixCSC{T, Int32}(
        nd, nd, sym.schur_csc_colptr, sym.schur_csc_rowval, zeros(T, sym.schur_nnz),
    )
    schur_block = Matrix{T}(undef, m_coupled, m_coupled)
    C_dk = [Matrix{T}(undef, m_coupled, blk_size) for _ in 1:ns]
    tmp_blk_nd = [Matrix{T}(undef, blk_size, m_coupled) for _ in 1:ns]

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
    rhs_d = VT(undef, nd)
    rhs_d_red = VT(undef, m_coupled)
    rhs_k       = [VT(undef, blk_size) for _ in 1:ns]
    solve_buffers = [VT(undef, blk_size) for _ in 1:ns]

    # --- Init ---
    fill!(pr_diag, zero(T))
    fill!(du_diag, zero(T))

    # --- Create solvers ---
    quasi_newton = create_quasi_newton(hessian_approximation, cb, n; options=qn_options)
    scenario_solvers = [schur_scenario_linear_solver(A_kk_vec[k]; opt = schur_scenario_opt_linear_solver) for k in 1:ns]
    _linear_solver = linear_solver(schur_csc; opt = opt_linear_solver)

    return SchurComplementCondensedKKTSystem(
        hess, jac,
        hess_raw, jt_coo,
        hess_csc, hess_csc_map, jt_csc, jt_csc_map,
        quasi_newton,
        reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower,
        ns, nv, nd, nc,
        nc_ineq_per_s, blk_size,
        A_kk_vec, C_dk,
        schur_csc, m_coupled, sym.coupled_design_local, sym.schur_fill_nzpos, schur_block,
        diag_buffer, buffer, rhs_d, rhs_d_red, rhs_k, tmp_blk_nd, solve_buffers,
        block_maps,
        sym.schur_hess_coo, sym.schur_hess_nzpos,
        sym.schur_diag_nzpos,
        sym.schur_ineq_S_nzpos, sym.schur_ineq_S_jcoo1, sym.schur_ineq_S_jcoo2, sym.schur_ineq_S_bufidx,
        sym.schur_design_ineq_S_nzpos, sym.schur_design_ineq_S_jcoo1, sym.schur_design_ineq_S_jcoo2, sym.schur_design_ineq_S_bufidx,
        sym.design_var_global, sym.scen_var_global,
        n_eq, cb.ind_eq,
        ns_ineq, cb.ind_ineq, cb.ind_lb, cb.ind_ub,
        Ref(0), Ref(0), Ref(0),
        scenario_solvers,
        _linear_solver,
    )
end

num_variables(kkt::SchurComplementCondensedKKTSystem) = size(kkt.hess_csc, 1)

function get_slack_regularization(kkt::SchurComplementCondensedKKTSystem)
    n = num_variables(kkt)
    ns_ineq = kkt.n_ineq
    return view(kkt.pr_diag, n+1:n+ns_ineq)
end

"""
    is_inertia(kkt::SchurComplementCondensedKKTSystem)

The Schur path can only certify In(K) if *every* factorization contributing to the
Haynsworth sum reports inertia — the complement solver and all `ns` block solvers.
"""
is_inertia(kkt::SchurComplementCondensedKKTSystem) =
    is_inertia(kkt.linear_solver) && all(is_inertia, kkt.scenario_solvers)

function is_inertia_correct(kkt::SchurComplementCondensedKKTSystem, num_pos, num_zero, num_neg)
    # `(num_pos, num_zero, num_neg)` is In(S) alone — the IPM reads it off
    # `kkt.linear_solver`, which factorizes only the nd×nd first-stage complement. By
    # Haynsworth additivity In(K) = Σ_k In(A_k) + In(S), so In(S) == (nd, 0, 0) certifies
    # the whole condensed KKT matrix ONLY if every block A_k is positive definite too.
    #
    # This is not a conservative omission: an indefinite A_k makes its contribution
    # −C_k A_k⁻¹ C_kᵀ to S a *positive* shift, which can leave S positive definite. S then
    # passes, δ_w is never raised, and the IPM accepts a direction with wrong curvature.
    # The counts below are refreshed by `build_kkt!`, which factorizes the blocks; the IPM
    # calls `factorize_wrapper!` (→ `build_kkt!`) before every inertia query, so they are
    # always in sync with the In(S) passed in.
    is_inertia(kkt) || throw(InertiaException())
    return (num_zero == 0) && (num_pos == kkt.nd) && (num_neg == 0) &&
           (kkt.block_num_neg[] == 0) && (kkt.block_num_zero[] == 0)
end

"""
    _accumulate_block_inertia!(kkt::SchurComplementCondensedKKTSystem)

Sum the per-scenario block inertias into `kkt.block_num_neg` / `block_num_zero` /
`block_num_indef`. Called by `build_kkt!` right after the blocks are factorized; the
counts are a by-product of that factorization (MUMPS stores them in INFOG(12)), so this
only reads them back. Solvers that do not report inertia are skipped — `is_inertia(kkt)`
is then false and `is_inertia_correct` refuses to certify anything.

Set `MADNLP_SCHUR_BLOCK_INERTIA` in the environment to log the counts per factorization.
"""
function _accumulate_block_inertia!(kkt::SchurComplementCondensedKKTSystem)
    num_neg = 0
    num_zero = 0
    num_indef = 0
    worst = 0
    for s in kkt.scenario_solvers
        is_inertia(s) || continue
        (_, z, neg) = inertia(s)
        num_neg += neg
        num_zero += z
        (neg > 0 || z > 0) && (num_indef += 1)
        worst = max(worst, neg)
    end
    kkt.block_num_neg[] = num_neg
    kkt.block_num_zero[] = num_zero
    kkt.block_num_indef[] = num_indef
    if haskey(ENV, "MADNLP_SCHUR_BLOCK_INERTIA")
        Base.@info "schur block inertia" blocks_indefinite = num_indef total_neg_pivots = num_neg total_zero_pivots = num_zero max_neg_pivots = worst ns = kkt.ns
    end
    return
end

should_regularize_dual(kkt::SchurComplementCondensedKKTSystem, num_pos, num_zero, num_neg) = true

function jtprod!(y::AbstractVector, kkt::SchurComplementCondensedKKTSystem, x::AbstractVector)
    nx = num_variables(kkt)
    ns_ineq = kkt.n_ineq
    yx = view(y, 1:nx)
    ys = view(y, 1+nx:nx+ns_ineq)
    mul!(yx, kkt.jt_csc, x)
    ys .= -@view(x[kkt.ind_ineq])
    return
end

function compress_jacobian!(kkt::SchurComplementCondensedKKTSystem)
    transfer!(kkt.jt_csc, kkt.jt_coo, kkt.jt_csc_map)
end

function compress_hessian!(kkt::SchurComplementCondensedKKTSystem)
    transfer!(kkt.hess_csc, kkt.hess_raw, kkt.hess_csc_map)
end

nnz_jacobian(kkt::SchurComplementCondensedKKTSystem) = nnz(kkt.jt_coo)

function build_kkt!(kkt::SchurComplementCondensedKKTSystem{T, VT, MT}) where {T, VT, MT}
    ns = kkt.ns
    nv = kkt.nv
    nd = kkt.nd
    m = kkt.m
    n = num_variables(kkt)
    blk = kkt.blk_size

    # Compute condensing diagonal for inequalities
    if kkt.n_ineq > 0
        Sigma_s = view(kkt.pr_diag, n+1:n+kkt.n_ineq)
        Sigma_d = @view(kkt.du_diag[kkt.ind_ineq])
        kkt.diag_buffer .= Sigma_s ./ (one(T) .- Sigma_d .* Sigma_s)
    end

    # Initialize the sparse first-stage bordered block: scatter the design Hessian and
    # the design pr_diag diagonal into the lower-triangular CSC `nz` by precomputed
    # position (design variables need not be the last nd global indices).
    nz = kkt.schur_csc.nzval
    fill!(nz, zero(T))
    _scatter_add!(nz, kkt.hess, kkt.schur_hess_coo, kkt.schur_hess_nzpos)
    _scatter_add!(nz, kkt.pr_diag, kkt.design_var_global, kkt.schur_diag_nzpos)

    # Phase 1 (parallel): assemble per-scenario blocks, factorize, compute A_kk^{-1} * C_dk'.
    # `@blas_safe_threads` runs `Threads.@threads` over scenarios while pinning
    # BLAS to a single thread per task to avoid oversubscription with the
    # per-scenario `mul!` / `factorize!` calls inside the loop. Only A_kk[k]/C_dk[k]/
    # tmp[k] are written here — the shared sparse `nz` is touched sequentially below.
    @blas_safe_threads for k in 1:ns
        bm = kkt.block_maps[k]
        A_kk = kkt.A_kk[k]
        C_dk = kkt.C_dk[k]
        aknz = A_kk.nzval

        fill!(aknz, zero(T))
        fill!(C_dk, zero(T))

        _scatter_add!(aknz, kkt.hess, bm.hess_Akk_coo, bm.hess_Akk_nzpos)
        _scatter_add!(C_dk, kkt.hess,    bm.hess_Cdk_coo,    bm.hess_Cdk_row, bm.hess_Cdk_col)
        _scatter_add!(aknz, kkt.pr_diag, bm.pr_diag_global, bm.pr_diag_nzpos)
        _scatter_quad_add!(
            aknz, kkt.jac, kkt.diag_buffer,
                           bm.ineq_Akk_nzpos, bm.ineq_Akk_jcoo1, bm.ineq_Akk_jcoo2, bm.ineq_Akk_bufidx)
        _scatter_quad_add!(C_dk, kkt.jac, kkt.diag_buffer,
                           bm.ineq_Cdk_row, bm.ineq_Cdk_col,
                           bm.ineq_Cdk_jcoo_d, bm.ineq_Cdk_jcoo_v, bm.ineq_Cdk_bufidx)

        # Factor A_kk
        factorize!(kkt.scenario_solvers[k])

        # Compute tmp_red = A_kk^{-1} * C_dk_red'  (blk × m): only the m coupled columns —
        # the other nd-m design columns of C_dk are exactly zero, so we never solve them.
        buf = kkt.solve_buffers[k]
        tmp = kkt.tmp_blk_nd[k]
        for j in 1:m
            @inbounds for i in 1:blk
                buf[i] = C_dk[j, i]  # C_dk_red' column j = row j of the reduced C_dk
            end
            solve_linear_system!(kkt.scenario_solvers[k], buf)
            @inbounds for i in 1:blk
                tmp[i, j] = buf[i]
            end
        end
    end

    # Aggregate the block inertias for `is_inertia_correct`. Each solver computed this as a
    # by-product of the `factorize!` above (MUMPS: INFOG(12)), so this is a read, not work.
    _accumulate_block_inertia!(kkt)

    # Phase 2 (sequential): scatter the scenario inequality condensation into `nz`
    # (flattened over scenarios), then the Schur reduction. The reduction
    # `Σ_k C_dk A_kk⁻¹ C_dk'` is nonzero only on the m coupled design vars, so it is
    # formed from the coupled rows/cols of the dense C_dk/tmp as an `m×m` symmetric
    # block and the lower half scattered as `-D` into `nz`.
    _scatter_quad_add!(
        nz, kkt.jac, kkt.diag_buffer,
        kkt.schur_ineq_S_nzpos, kkt.schur_ineq_S_jcoo1,
        kkt.schur_ineq_S_jcoo2, kkt.schur_ineq_S_bufidx
    )
    if m > 0
        for k in 1:ns
            # C_dk_red (m×blk) and tmp_red (blk×m) are already reduced to the coupled block, so
            # this is a plain contiguous GEMM (BLAS) — no Vector{Int}-indexed views, unlike the
            # old full-width nd path which fell back to the generic (non-BLAS) mul!.
            mul!(kkt.schur_block, kkt.C_dk[k], kkt.tmp_blk_nd[k])
            @inbounds for b in 1:m, a in b:m
                nz[kkt.schur_fill_nzpos[a, b]] -= kkt.schur_block[a, b]
            end
        end
    end

    # Design-only inequalities: condense into S_dd at their precomputed
    # (lower-triangle) positions.
    _scatter_quad_add!(
        nz, kkt.jac, kkt.diag_buffer,
        kkt.schur_design_ineq_S_nzpos, kkt.schur_design_ineq_S_jcoo1,
        kkt.schur_design_ineq_S_jcoo2, kkt.schur_design_ineq_S_bufidx
    )
    return
end

function factorize_kkt!(kkt::SchurComplementCondensedKKTSystem)
    return factorize!(kkt.linear_solver)
end

function solve_kkt!(
    kkt::SchurComplementCondensedKKTSystem,
    w::AbstractKKTVector{T},
) where T

    ns = kkt.ns
    nv = kkt.nv
    nd = kkt.nd
    nc = kkt.nc
    m = kkt.m
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

    # Step 2: Extract per-scenario RHS blocks via the tag-driven global index lists.
    @inbounds for k in 1:ns
        sv = kkt.scen_var_global[k]
        rhs = kkt.rhs_k[k]
        for i in 1:nv
            rhs[i] = wx[sv[i]]
        end
    end
    @inbounds for i in 1:nd
        kkt.rhs_d[i] = wx[kkt.design_var_global[i]]
    end

    # Step 3: Forward elimination
    # Phase 1 (parallel): solve per-scenario systems
    @blas_safe_threads for k in 1:ns
        solve_linear_system!(kkt.scenario_solvers[k], kkt.rhs_k[k])
    end
    # Phase 2 (sequential): rhs_d[coupled] -= Σ_k C_dk_red[k] * rhs_k[k]. Only the m coupled
    # design rows receive a contribution (the reduced C_dk holds exactly those rows). Accumulate
    # into the contiguous reduced buffer (BLAS GEMV), then scatter-subtract into rhs_d.
    if m > 0
        fill!(kkt.rhs_d_red, zero(T))
        for k in 1:ns
            mul!(kkt.rhs_d_red, kkt.C_dk[k], kkt.rhs_k[k], one(T), one(T))
        end
        @views kkt.rhs_d[kkt.coupled_design_local] .-= kkt.rhs_d_red
    end

    # Step 4: Solve the first-stage Schur complement system (size nd, SPD)
    solve_linear_system!(kkt.linear_solver, kkt.rhs_d)

    # Step 5: Back-substitution. rhs_k[k] -= tmp_red[k] * rhs_d[coupled] per scenario (only the
    # coupled design entries feed back; tmp_red is the reduced blk×m block).
    if m > 0
        @views kkt.rhs_d_red .= kkt.rhs_d[kkt.coupled_design_local]
        @blas_safe_threads for k in 1:ns
            mul!(kkt.rhs_k[k], kkt.tmp_blk_nd[k], kkt.rhs_d_red, -one(T), one(T))
        end
    end

    # Step 6: Write back to w
    @inbounds for k in 1:ns
        sv = kkt.scen_var_global[k]
        rhs = kkt.rhs_k[k]
        for i in 1:nv
            wx[sv[i]] = rhs[i]
        end
    end
    @inbounds for i in 1:nd
        wx[kkt.design_var_global[i]] = kkt.rhs_d[i]
    end

    # Step 7: Recover inequality duals and slacks (all constraints are inequalities
    # under RelaxEquality, so there are no equality duals to preserve).
    if kkt.n_ineq > 0
        # J * Δx via sparse: (jt_csc)' * wx  (overwrites all of wy)
        mul!(wy, kkt.jt_csc', wx)

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
function mul!(w::AbstractKKTVector{T}, kkt::SchurComplementCondensedKKTSystem{T}, x::AbstractKKTVector, alpha = one(T), beta = zero(T)) where T
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

function mul_hess_blk!(wx, kkt::SchurComplementCondensedKKTSystem, t)
    n = num_variables(kkt)
    mul!(@view(wx[1:n]), Symmetric(kkt.hess_csc, :L), @view(t[1:n]))
    fill!(@view(wx[n+1:end]), 0)
    wx .+= t .* kkt.pr_diag
end
