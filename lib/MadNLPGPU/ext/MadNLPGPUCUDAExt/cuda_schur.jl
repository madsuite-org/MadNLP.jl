###########################################################
##### CUDA wrappers for SchurComplementCondensedKKTSystem ##########
###########################################################

"""
    GPUSchurComplementCondensedKKTSystem

GPU-native Schur complement KKT system, the CUDA counterpart of CPU
[`MadNLP.SchurComplementCondensedKKTSystem`](@ref). Uses a single batched
`CuSparseMatrixCSC` holding all `ns` scenario blocks (factored by `CUDSSSolver`
with uniform batching) and CUBLAS strided batched GEMM for the Schur reduction.

The first-stage Schur complement `S` is itself assembled as a **sparse**
lower-triangular `CuSparseMatrixCSC` (the reduction `Σ_k C_dk A_kk⁻¹ C_dk'` fills
only the coupled-design × coupled-design block) and factored by a second
`CUDSSSolver` (`nbatch=1`, which also reports inertia). The coupling block `C_dk`
is stored reduced to its `m` coupled design columns. The sparsity pattern is
static across the IPM loop: cuDSS analysis runs once at construction; each
iteration only refreshes `S.nzVal` and refactorizes.

Variable layout: `[v_1, ..., v_ns, d]`, `v_k ∈ R^nv`, `d ∈ R^nd`.
Constraint layout: `[c_1, ..., c_ns]`, `c_k ∈ R^nc`.

The CPU version's `Vector{ScenarioBlockMap}` is flattened here into a set of
device vectors, one per (block-target × source-tensor) pair, with all `ns`
scenarios concatenated in scenario order. Each `n_per_s_*` field stores the
per-scenario length; the matching `gpu_*` vector then has length `ns * n_per_s_*`
and is indexed directly by the kernel global index.
"""
struct GPUSchurComplementCondensedKKTSystem{
    T,
    VT <: AbstractVector{T},
    MT <: AbstractMatrix{T},
    QN,
        LS,   # linear solver for the sparse Schur complement S (CUDSSSolver, nbatch=1)
        LS2,  # batched scenario solver (CUDSSSolver, nbatch=ns)
    COO_T,   # SparseMatrixCOO type for hess_raw
    COO_JT,  # SparseMatrixCOO type for jt_coo
    CSC_T,   # CuSparseMatrixCSC type
    VI <: AbstractVector{Int},
} <: MadNLP.AbstractCondensedKKTSystem{T, VT, MT, QN}

    # COO value buffers on GPU
    hess::VT
    jac::VT

    # COO structures
    hess_raw::COO_T
    jt_coo::COO_JT

    # CSC representations on GPU
    hess_csc::CSC_T
    hess_csc_map::VI
    jt_csc::CSC_T
    jt_csc_map::VI

    quasi_newton::QN

    # Standard diagonals on GPU
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT

    # Two-stage dimensions
    ns::Int
    nv::Int
    nd::Int
    nc::Int
    nc_ineq_per_s::Int
    blk_size::Int                   # == nv (per-scenario condensed block size)
    m::Int                          # number of coupled design vars (Schur fill width)

    # Batched scenario block
    A_kk_batched::CSC_T
    nnz_per_scenario::Int

    # Reduced coupling blocks: only the m design columns that couple to a scenario.
    # `(blk_size, m, ns)`, scenario-var dim first, so each slice `C_dk[:, :, k]` is
    # the cuDSS multi-RHS input for `A_kk \ C_dk_red'`.
    C_dk_batched::CuArray{T, 3}
    schur_csc::CSC_T                # lower-triangular sparse Schur complement S (nd × nd, SPD)
    schur_block_batched::CuArray{T, 3}   # (m, m, ns) — per-scenario C_dk_red' A_kk⁻¹ C_dk_red'
    coupled_design_local::VI        # length m — design-local indices that couple to scenarios
    schur_fill_nzpos::CuMatrix{Int} # (m, m) — nzval position of the Schur-fill block entries

    # Buffers
    diag_buffer::VT
    buffer::VT
    rhs_d::VT                           # nd
    rhs_d_red::VT                       # m — reduced design coupling RHS
    rhs_d_red_batched::CuArray{T, 3}    # (m, 1, ns) — strided-batched reduction GEMM output
    rhs_d_red_mat::MT                   # (m, ns) — rhs_d_red broadcast across scenarios (back-sub)
    rhs_k_batched::MT                   # (blk_size, ns)
    tmp_blk_nd_batched::CuArray{T, 3}   # (blk_size, m, ns)
    solve_buffer::VT                    # blk_size * ns

    # Flattened GPU index maps (all scenarios concatenated in scenario order).
    # See struct docstring for the layout convention.
    n_per_s_hess_Akk::Int
    gpu_hess_Akk_coo::VI
    gpu_hess_Akk_nzpos::VI

    n_per_s_hess_Cdk::Int
    gpu_hess_Cdk_coo::VI
    gpu_hess_Cdk_row::VI
    gpu_hess_Cdk_col::VI

    n_per_s_pr_diag::Int
    gpu_pr_diag_global::VI
    gpu_pr_diag_nzpos::VI

    n_per_s_ineq_Akk::Int
    gpu_ineq_Akk_nzpos::VI
    gpu_ineq_Akk_jcoo1::VI
    gpu_ineq_Akk_jcoo2::VI
    gpu_ineq_Akk_bufidx::VI

    n_per_s_ineq_Cdk::Int
    gpu_ineq_Cdk_row::VI
    gpu_ineq_Cdk_col::VI
    gpu_ineq_Cdk_jcoo_d::VI
    gpu_ineq_Cdk_jcoo_v::VI
    gpu_ineq_Cdk_bufidx::VI

    # Sparse-Schur scatter maps: scatter into `schur_csc.nzVal` at precomputed nzval
    # positions (lower-triangle CSC; cuDSS symmetrizes). Lists are lower-only so each
    # slot is hit once.
    schur_hess_coo::VI                    # design Hessian: COO index into `hess`
    schur_hess_nzpos::VI                  # → nzval slot
    schur_diag_nzpos::VI                  # pr_diag design diagonal (paired with design_var_global)
    schur_ineq_S_nzpos::VI                # scenario ineq → S (flat over all scenarios)
    schur_ineq_S_jcoo1::VI
    schur_ineq_S_jcoo2::VI
    schur_ineq_S_bufidx::VI
    schur_design_ineq_S_nzpos::VI         # design ineq → S
    schur_design_ineq_S_jcoo1::VI
    schur_design_ineq_S_jcoo2::VI
    schur_design_ineq_S_bufidx::VI

    # Tag-driven global index lists (design/scenario vars need not be contiguous).
    design_var_global::VI                 # length nd — global index of each design var
    scen_var_global::VI                   # length ns*nv (scenario-major) — scenario var globals

    # Inequality/equality/bound indices (n_eq == 0 / ind_eq empty under RelaxEquality,
    # kept for the generic KKT interface).
    n_eq::Int
    ind_eq::VI
    n_ineq::Int
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI

    # Solvers
    scenario_solver::LS2                  # batched per-scenario blocks (CUDSSSolver, nbatch=ns)
    # Schur complement (CUDSSSolver, nbatch=1); the IPM queries this field for inertia/factorize.
    linear_solver::LS

    # Lazily-built deterministic condensation-scatter structure (NamedTuple of _DetScatter /
    # _DetLinScatter), memoized here so it is owned by — and freed with — the kkt.
    det_cache::Base.RefValue{Any}
end

# cuDSS sparse-LDL is less accurate than a direct factorization. The relaxed Schur
# path solves each (SPD) per-scenario block with it and accumulates A_kk⁻¹ into the
# first-stage complement, so per-block error compounds — iterative refinement is the
# remedy, applied to both the batched per-scenario blocks and the nbatch=1 complement.
# Empirically on case9 two-stage SCOPF: ir=1 diverges to NaN, ir=3 reaches the optimum
# neighbourhood but stalls (inf_du ~1e-1), ir=5 converges cleanly to tol=1e-4 (obj
# matches the CPU / `:single` solve). Default to 5; stiffer/larger problems may need
# more — override via `--cudss-ir` / the `schur_*_opt_linear_solver` kwargs.
#
# This inner cuDSS IR is COMPLEMENTARY to — not a replacement for — the outer full-KKT
# `RichardsonIterator` (src/LinearSolvers/backsolve.jl), which already refines every Newton
# step against the true KKT residual. The outer IR's approximate inverse is one Schur
# reduction, whose accuracy depends on these per-scenario solves, so accurate inner solves
# are a prerequisite for the outer IR to converge (cf. Petra–Schenk–Lubin 2014). When the
# inner solve genuinely fails (e.g. IR on a numerically singular block), the cuDSS wrappers
# raise a Solve/FactorizationException → ERROR_IN_STEP_COMPUTATION rather than crashing.
const SCHUR_DEFAULT_CUDSS_IR = 5
function _default_schur_cudss_options()
    opt = MadNLP.default_options(CUDSSSolver)
    opt.cudss_ir = SCHUR_DEFAULT_CUDSS_IR
    opt.cudss_ir_tol = 0.0  # disarm cuDSS 0.8's IR_FAILED gate; keep the refinement steps
    return opt
end

# --- Deterministic condensation scatter: sorted-by-slot segment structure -------------
# The Σ-amplified inequality-condensation scatters were atomic-add (nondeterministic
# summation order). For a given target array, group the contributions by their linear target
# slot so each slot can be summed by a single thread in a fixed order (see _det_quad_scatter!
# in kernels_schur.jl). Built once per kkt from the static index maps and cached.
struct _DetScatter{VI}      # quadratic condensation: value = diag_buffer*jac*jac
    jc1::VI
    jc2::VI
    buf::VI
    segstart::VI
    segslot::VI
end
struct _DetLinScatter{VI}   # linear scatter: value = src[src_idx]
    src_idx::VI
    segstart::VI
    segslot::VI
end

# Build the contiguous per-slot segment boundaries for a stable sort of `slots`.
function _segments(slots::Vector{Int})
    perm = sortperm(slots)
    s = slots[perm]
    segslot = Int[]; segstart = Int[]
    @inbounds for t in eachindex(s)
        if t == 1 || s[t] != s[t - 1]
            push!(segslot, s[t]); push!(segstart, t)
        end
    end
    push!(segstart, length(s) + 1)
    return perm, segstart, segslot
end

function _segment_scatter(slots::Vector{Int}, jc1::Vector{Int}, jc2::Vector{Int}, buf::Vector{Int})
    if isempty(slots)
        z = CuVector{Int}(undef, 0)
        return _DetScatter(z, z, z, CuVector{Int}([1]), z)
    end
    perm, segstart, segslot = _segments(slots)
    return _DetScatter(
        CuVector{Int}(jc1[perm]), CuVector{Int}(jc2[perm]), CuVector{Int}(buf[perm]),
        CuVector{Int}(segstart), CuVector{Int}(segslot),
    )
end

function _segment_lin_scatter(slots::Vector{Int}, src_idx::Vector{Int})
    if isempty(slots)
        z = CuVector{Int}(undef, 0)
        return _DetLinScatter(z, CuVector{Int}([1]), z)
    end
    perm, segstart, segslot = _segments(slots)
    return _DetLinScatter(CuVector{Int}(src_idx[perm]), CuVector{Int}(segstart), CuVector{Int}(segslot))
end

# Build the (A_kk, C_dk, S) deterministic scatter structures for `kkt` from its static index
# maps. Computed once and memoized in `kkt.det_cache` (a Ref field owned by the kkt, so it is
# freed with the kkt — no global cache, no leak).
function _compute_det_scatter(kkt::GPUSchurComplementCondensedKKTSystem)
    ns = kkt.ns; nnzb = kkt.nnz_per_scenario; blk = kkt.blk_size; m = kkt.m
    # ineq → A_kk: linear slot into the batched nzVal = (k-1)*nnzb + nzpos
    n_iA = kkt.n_per_s_ineq_Akk
    nz = Array(kkt.gpu_ineq_Akk_nzpos)
    slotA = n_iA > 0 ? Int[((i - 1) ÷ n_iA) * nnzb + nz[i] for i in eachindex(nz)] : Int[]
    Akk = _segment_scatter(slotA,
        Array(kkt.gpu_ineq_Akk_jcoo1), Array(kkt.gpu_ineq_Akk_jcoo2), Array(kkt.gpu_ineq_Akk_bufidx))
    # ineq → C_dk: linear slot into (blk × m × ns) = v + (d-1)*blk + (k-1)*blk*m
    n_iC = kkt.n_per_s_ineq_Cdk
    vv = Array(kkt.gpu_ineq_Cdk_col); dd = Array(kkt.gpu_ineq_Cdk_row)
    slotC = n_iC > 0 ? Int[vv[i] + (dd[i] - 1) * blk + ((i - 1) ÷ n_iC) * blk * m for i in eachindex(vv)] : Int[]
    Cdk = _segment_scatter(slotC,
        Array(kkt.gpu_ineq_Cdk_jcoo_d), Array(kkt.gpu_ineq_Cdk_jcoo_v), Array(kkt.gpu_ineq_Cdk_bufidx))
    # ineq → S: merge scenario + design-only contributions, slot = nzpos
    slotS = vcat(Array(kkt.schur_ineq_S_nzpos), Array(kkt.schur_design_ineq_S_nzpos))
    j1S = vcat(Array(kkt.schur_ineq_S_jcoo1), Array(kkt.schur_design_ineq_S_jcoo1))
    j2S = vcat(Array(kkt.schur_ineq_S_jcoo2), Array(kkt.schur_design_ineq_S_jcoo2))
    bfS = vcat(Array(kkt.schur_ineq_S_bufidx), Array(kkt.schur_design_ineq_S_bufidx))
    S = _segment_scatter(slotS, j1S, j2S, bfS)

    # Hessian scatters are ALSO large-valued (constraint Hessian × the relaxed-equality duals
    # λ) and the Hessian COO has many duplicates → the non-atomic `+=` races. Make them
    # deterministic linear scatters too. (pr_diag scatters target distinct diagonal slots with
    # no collision, so they are already deterministic and left as-is.)
    n_hA = kkt.n_per_s_hess_Akk
    hnz = Array(kkt.gpu_hess_Akk_nzpos)
    slotHA = n_hA > 0 ? Int[((i - 1) ÷ n_hA) * nnzb + hnz[i] for i in eachindex(hnz)] : Int[]
    hessAkk = _segment_lin_scatter(slotHA, Array(kkt.gpu_hess_Akk_coo))
    n_hC = kkt.n_per_s_hess_Cdk
    hv = Array(kkt.gpu_hess_Cdk_col); hd = Array(kkt.gpu_hess_Cdk_row)
    slotHC = n_hC > 0 ? Int[hv[i] + (hd[i] - 1) * blk + ((i - 1) ÷ n_hC) * blk * m for i in eachindex(hv)] : Int[]
    hessCdk = _segment_lin_scatter(slotHC, Array(kkt.gpu_hess_Cdk_coo))
    hessS = _segment_lin_scatter(Array(kkt.schur_hess_nzpos), Array(kkt.schur_hess_coo))

    # --- Correct Hessian for the iterative-refinement mul! -----------------------------
    # The refinement matvec needs the TRUE full symmetric Hessian H. Two GPU pitfalls make
    # the naive `mul!(.., Symmetric(hess_csc,:L), ..)` wrong (it only poisons mul!, not the
    # Schur step — but mul! is the Richardson reference operator, so a wrong mul! stalls the
    # outer refinement and inflates the IPM iteration count ~10x):
    #   (1) the generic GPU `transfer!` is a non-summing scatter `view(nzVal,map) .= V`, so
    #       duplicate Hessian COO entries (very common from AD) are dropped, not summed;
    #   (2) CUSPARSE ignores the `Symmetric(.,:L)` wrapper and multiplies only the stored
    #       lower triangle, dropping the strict-upper contribution.
    # Fix both here: (1) a deterministic segmented SUM that fills hess_csc.nzVal correctly,
    # and (2) a full-symmetric CSC `hess_full` so a plain general SpMV computes the true H*x.
    nnzc = length(kkt.hess)
    hess_lin = _segment_lin_scatter(Array(kkt.hess_csc_map), collect(1:nnzc))
    hess_full, hess_full_perm = MadNLP.get_tril_to_full(kkt.hess_csc)
    fill!(hess_full.nzVal, zero(eltype(hess_full.nzVal)))

    # --- Correct Jacobian assembly for compress_jacobian! -------------------------------
    # The generic GPU `transfer!` (`view(jt_csc.nzVal, map) .= jac`) is a NON-summing scatter, so
    # duplicate Jacobian COO entries (same (constraint, variable) coordinate — common with
    # ExaModels AD) are dropped last-write-wins. The condensation kernels and the Hessian scatter
    # SUM duplicates, so jt_csc must too, else jtprod!/mul!/dual recovery use a *different*
    # Jacobian than the assembled A_kk/C_dk/S blocks. Build a deterministic segmented SUM keyed by
    # jt_csc nzval slot (source = the jac COO values).
    njac = length(kkt.jac)
    jt_lin = _segment_lin_scatter(Array(kkt.jt_csc_map), collect(1:njac))

    return (Akk = Akk, Cdk = Cdk, S = S, hessAkk = hessAkk, hessCdk = hessCdk, hessS = hessS,
            hess_lin = hess_lin, hess_full = hess_full, hess_full_perm = hess_full_perm,
            jt_lin = jt_lin)
end

function _get_det_scatter(kkt::GPUSchurComplementCondensedKKTSystem)
    kkt.det_cache[] === nothing && (kkt.det_cache[] = _compute_det_scatter(kkt))
    return kkt.det_cache[]
end

# --- Dispatch: GPU path when callback uses CuVector ---
function MadNLP.create_kkt_system(
    ::Type{MadNLP.SchurComplementCondensedKKTSystem},
    cb::MadNLP.SparseCallback{T, VT},
    linear_solver::Type;
    opt_linear_solver=MadNLP.default_options(linear_solver),
    hessian_approximation=MadNLP.ExactHessian,
    qn_options=MadNLP.QuasiNewtonOptions(),
    schur_ns::Int=0,
    schur_nv::Int=0,
    schur_nd::Int=0,
    schur_nc::Int=0,
        schur_var_scen = nothing,
        schur_con_scen = nothing,
        schur_scenario_linear_solver::Type = CUDSSSolver,
        schur_scenario_opt_linear_solver = _default_schur_cudss_options(),
        schur_opt_linear_solver = _default_schur_cudss_options(),
        kwargs...,
) where {T, VT <: CuVector{T}}

    isempty(kwargs) || Base.@warn(
        "GPUSchurComplementCondensedKKTSystem ignores unsupported kkt_options: " *
            join(string.(keys(kwargs)), ", ")
    )
    schur_scenario_linear_solver === CUDSSSolver || Base.@warn(
        "GPUSchurComplementCondensedKKTSystem always factorizes the per-scenario blocks with a " *
            "batched cuDSS solver; the requested `schur_scenario_linear_solver=" *
            "$(schur_scenario_linear_solver)` is ignored. Pass `schur_scenario_opt_linear_solver` " *
            "to configure the cuDSS scenario solver."
    )
    # The first-stage Schur complement is also cuDSS (nbatch=1, so it reports inertia); the
    # positional `linear_solver`/`opt_linear_solver` chosen by the IPM are NOT used here. Warn
    # loudly when the user overrode `linear_solver` (e.g. LapackCUDASolver or a custom
    # AbstractLinearSolver) instead of silently substituting cuDSS. The first-stage cuDSS options
    # are configured via `schur_opt_linear_solver`.
    linear_solver === CUDSSSolver || Base.@warn(
        "GPUSchurComplementCondensedKKTSystem factorizes the first-stage Schur complement with " *
            "cuDSS; the requested `linear_solver=$(linear_solver)` (and its `opt_linear_solver`) " *
            "is ignored. Configure the first-stage cuDSS solver via `schur_opt_linear_solver`."
    )

    n = cb.nvar
    m = cb.ncon
    ns_ineq = length(cb.ind_ineq)
    n_eq_total = m - ns_ineq
    nlb = length(cb.ind_lb)
    nub = length(cb.ind_ub)

    dims = MadNLP._resolve_schur_dims(cb, n, m, schur_ns, schur_nv, schur_nd, schur_nc, schur_var_scen, schur_con_scen)
    ns, nv, nd, nc = dims.ns, dims.nv, dims.nd, dims.nc

    # --- Get sparsity patterns on CPU ---
    jac_sparsity_I = Vector{Int32}(undef, cb.nnzj)
    jac_sparsity_J = Vector{Int32}(undef, cb.nnzj)
    MadNLP._jac_sparsity_wrapper!(cb, jac_sparsity_I, jac_sparsity_J)

    hess_sparsity_I_gpu, hess_sparsity_J_gpu = MadNLP.build_hessian_structure(cb, hessian_approximation)
    # Convert to CPU for construction-time classification loops
    hess_sparsity_I = Vector{Int32}(Array(hess_sparsity_I_gpu))
    hess_sparsity_J = Vector{Int32}(Array(hess_sparsity_J_gpu))
    MadNLP.force_lower_triangular!(hess_sparsity_I, hess_sparsity_J)

    n_hess = length(hess_sparsity_I)
    n_jac = length(jac_sparsity_I)

    # --- COO value buffers on GPU ---
    hess = CuVector{T}(undef, n_hess)
    jac = CuVector{T}(undef, n_jac)
    fill!(hess, zero(T))
    fill!(jac, zero(T))

    # --- Build global COO + CSC on GPU ---
    hess_I_gpu = CuVector{Int32}(hess_sparsity_I)
    hess_J_gpu = CuVector{Int32}(hess_sparsity_J)
    jac_I_gpu = CuVector{Int32}(jac_sparsity_I)
    jac_J_gpu = CuVector{Int32}(jac_sparsity_J)

    hess_raw = MadNLP.SparseMatrixCOO(n, n, hess_I_gpu, hess_J_gpu, hess)
    jt_coo = MadNLP.SparseMatrixCOO(n, m, jac_J_gpu, jac_I_gpu, jac)

    hess_csc, hess_csc_map = MadNLP.coo_to_csc(hess_raw)
    jt_csc, jt_csc_map = MadNLP.coo_to_csc(jt_coo)

    # --- Classify constraints per scenario and build per-scenario maps (CPU) ---
    cpu_ind_eq = Array(cb.ind_eq)
    cpu_ind_ineq = Array(cb.ind_ineq)
    cpu_ind_lb = Array(cb.ind_lb)
    cpu_ind_ub = Array(cb.ind_ub)

    sym = MadNLP._build_schur_symbolic(
        T, n, m, ns, nv, nd, nc,
        hess_sparsity_I, hess_sparsity_J,
        jac_sparsity_I, jac_sparsity_J,
        cpu_ind_eq, cpu_ind_ineq,
        dims.var_scen, dims.con_scen,
    )
    nc_ineq_per_s = sym.nc_ineq_per_s
    blk_size = sym.blk_size
    nnz_per_scenario = sym.nnz_per_scenario
    akk_csc_cpu = sym.akk_csc_template
    m_coupled = sym.m_coupled
    coupled_inv = sym.coupled_inv

    # Flatten the per-scenario variable global indices (scenario-major) for the
    # extract/writeback kernels, which can no longer assume contiguous stripes.
    scen_var_global_flat = reduce(vcat, sym.scen_var_global; init = Int[])

    # Flatten per-scenario block_maps into the per-field concatenated CPU vectors
    # the GPU struct stores (one upload per field below).
    flat = MadNLP._flatten_block_maps(sym.block_maps)

    # --- Create batched CSC on GPU (shared colptr/rowval, per-scenario nzval slabs) ---
    batched_colPtr = CuVector{Cint}(Vector{Cint}(akk_csc_cpu.colptr))
    batched_rowVal = CuVector{Cint}(Vector{Cint}(akk_csc_cpu.rowval))
    batched_nzVal = CUDACore.fill(zero(T), ns * nnz_per_scenario)

    A_kk_batched = cuSPARSE.CuSparseMatrixCSC{T, Cint}(
        batched_colPtr, batched_rowVal, batched_nzVal, (blk_size, blk_size),
    )

    # --- Sparse Schur complement (lower-triangular CSC) + reduced coupling buffers ---
    schur_colPtr = CuVector{Cint}(Vector{Cint}(sym.schur_csc_colptr))
    schur_rowVal = CuVector{Cint}(Vector{Cint}(sym.schur_csc_rowval))
    schur_nzVal = CUDACore.fill(zero(T), sym.schur_nnz)
    schur_csc = cuSPARSE.CuSparseMatrixCSC{T, Cint}(
        schur_colPtr, schur_rowVal, schur_nzVal, (nd, nd),
    )
    # Reduced coupling: only the m design columns that couple to a scenario.
    C_dk_batched = CUDACore.fill(zero(T), blk_size, m_coupled, ns)
    tmp_blk_nd_batched = CuArray{T, 3}(undef, blk_size, m_coupled, ns)
    schur_block_batched = CuArray{T, 3}(undef, m_coupled, m_coupled, ns)
    gpu_schur_fill_nzpos = CuMatrix{Int}(sym.schur_fill_nzpos)
    gpu_coupled_design_local = CuVector{Int}(sym.coupled_design_local)

    # --- Diagonal vectors on GPU ---
    reg     = CuVector{T}(undef, n + ns_ineq)
    pr_diag = CuVector{T}(undef, n + ns_ineq)
    du_diag = CuVector{T}(undef, m)
    l_diag  = CUDACore.fill(one(T), nlb)
    u_diag  = CUDACore.fill(one(T), nub)
    l_lower = CUDACore.fill(zero(T), nlb)
    u_lower = CUDACore.fill(zero(T), nub)

    fill!(pr_diag, zero(T))
    fill!(du_diag, zero(T))

    # --- Buffers ---
    diag_buffer = CuVector{T}(undef, max(ns_ineq, 1))
    buffer      = CuVector{T}(undef, m)
    rhs_d = CuVector{T}(undef, nd)
    rhs_d_red = CuVector{T}(undef, m_coupled)
    rhs_d_red_batched = CuArray{T, 3}(undef, m_coupled, 1, ns)
    rhs_d_red_mat = CuMatrix{T}(undef, m_coupled, ns)
    rhs_k_batched = CuMatrix{T}(undef, blk_size, ns)
    solve_buffer = CuVector{T}(undef, blk_size * ns)

    # --- Transfer flattened index maps to GPU ---
    gpu_hess_Akk_coo    = CuVector{Int}(flat.all_hess_Akk_coo)
    gpu_hess_Akk_nzpos  = CuVector{Int}(flat.all_hess_Akk_nzpos)
    gpu_hess_Cdk_coo    = CuVector{Int}(flat.all_hess_Cdk_coo)
    # Reduced C_dk: remap the design-column targets (1:nd) to compact columns (1:m).
    gpu_hess_Cdk_row = CuVector{Int}(coupled_inv[flat.all_hess_Cdk_row])
    gpu_hess_Cdk_col    = CuVector{Int}(flat.all_hess_Cdk_col)
    gpu_pr_diag_global  = CuVector{Int}(flat.all_pr_diag_global)
    gpu_pr_diag_nzpos   = CuVector{Int}(flat.all_pr_diag_nzpos)
    gpu_ineq_Akk_nzpos  = CuVector{Int}(flat.all_ineq_Akk_nzpos)
    gpu_ineq_Akk_jcoo1  = CuVector{Int}(flat.all_ineq_Akk_jcoo1)
    gpu_ineq_Akk_jcoo2  = CuVector{Int}(flat.all_ineq_Akk_jcoo2)
    gpu_ineq_Akk_bufidx = CuVector{Int}(flat.all_ineq_Akk_bufidx)
    gpu_ineq_Cdk_row = CuVector{Int}(coupled_inv[flat.all_ineq_Cdk_row])
    gpu_ineq_Cdk_col    = CuVector{Int}(flat.all_ineq_Cdk_col)
    gpu_ineq_Cdk_jcoo_d = CuVector{Int}(flat.all_ineq_Cdk_jcoo_d)
    gpu_ineq_Cdk_jcoo_v = CuVector{Int}(flat.all_ineq_Cdk_jcoo_v)
    gpu_ineq_Cdk_bufidx = CuVector{Int}(flat.all_ineq_Cdk_bufidx)

    # Sparse-Schur nzpos scatter maps (lower-only) + their value sources.
    gpu_schur_hess_coo = CuVector{Int}(sym.schur_hess_coo)
    gpu_schur_hess_nzpos = CuVector{Int}(sym.schur_hess_nzpos)
    gpu_schur_diag_nzpos = CuVector{Int}(sym.schur_diag_nzpos)
    gpu_schur_ineq_S_nzpos = CuVector{Int}(sym.schur_ineq_S_nzpos)
    gpu_schur_ineq_S_jcoo1 = CuVector{Int}(sym.schur_ineq_S_jcoo1)
    gpu_schur_ineq_S_jcoo2 = CuVector{Int}(sym.schur_ineq_S_jcoo2)
    gpu_schur_ineq_S_bufidx = CuVector{Int}(sym.schur_ineq_S_bufidx)
    gpu_schur_design_ineq_S_nzpos = CuVector{Int}(sym.schur_design_ineq_S_nzpos)
    gpu_schur_design_ineq_S_jcoo1 = CuVector{Int}(sym.schur_design_ineq_S_jcoo1)
    gpu_schur_design_ineq_S_jcoo2 = CuVector{Int}(sym.schur_design_ineq_S_jcoo2)
    gpu_schur_design_ineq_S_bufidx = CuVector{Int}(sym.schur_design_ineq_S_bufidx)

    # Tag-driven global index lists.
    gpu_design_var_global = CuVector{Int}(sym.design_var_global)
    gpu_scen_var_global = CuVector{Int}(scen_var_global_flat)

    # --- Create solvers ---
    quasi_newton = MadNLP.create_quasi_newton(hessian_approximation, cb, n; options=qn_options)
    # Both the per-scenario blocks and the Schur complement are factorized by cuDSS.
    # The per-scenario blocks are batched (nbatch=ns); the Schur complement is a single
    # lower-triangular CSC (nbatch=1), so cuDSS reports its inertia. The dense cuSOLVER
    # `sytrf` path is replaced. `linear_solver`/`opt_linear_solver` are unused on GPU.
    # The batched scenario solver is analyzed once for its single-RHS shape (nbatch=ns) inside
    # the CUDSSSolver constructor, and every scenario solve in build_kkt!/solve_kkt! is
    # single-RHS (the multi-RHS ubatch solve is broken above nbatch≈14, see build_kkt!), so no
    # extra multi-RHS descriptors/analysis are needed.
    scenario_solver = CUDSSSolver(A_kk_batched; opt=schur_scenario_opt_linear_solver)
    schur_solver = CUDSSSolver(schur_csc; opt = schur_opt_linear_solver)  # analysis runs once

    return GPUSchurComplementCondensedKKTSystem(
        hess, jac,
        hess_raw, jt_coo,
        hess_csc, hess_csc_map, jt_csc, jt_csc_map,
        quasi_newton,
        reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower,
        ns, nv, nd, nc, nc_ineq_per_s, blk_size, m_coupled,
        A_kk_batched, nnz_per_scenario,
        C_dk_batched, schur_csc, schur_block_batched, gpu_coupled_design_local, gpu_schur_fill_nzpos,
        diag_buffer, buffer, rhs_d, rhs_d_red, rhs_d_red_batched, rhs_d_red_mat, rhs_k_batched, tmp_blk_nd_batched, solve_buffer,
        flat.n_per_s_hess_Akk, gpu_hess_Akk_coo, gpu_hess_Akk_nzpos,
        flat.n_per_s_hess_Cdk, gpu_hess_Cdk_coo, gpu_hess_Cdk_row, gpu_hess_Cdk_col,
        flat.n_per_s_pr_diag, gpu_pr_diag_global, gpu_pr_diag_nzpos,
        flat.n_per_s_ineq_Akk, gpu_ineq_Akk_nzpos, gpu_ineq_Akk_jcoo1, gpu_ineq_Akk_jcoo2, gpu_ineq_Akk_bufidx,
        flat.n_per_s_ineq_Cdk, gpu_ineq_Cdk_row, gpu_ineq_Cdk_col, gpu_ineq_Cdk_jcoo_d, gpu_ineq_Cdk_jcoo_v, gpu_ineq_Cdk_bufidx,
        gpu_schur_hess_coo, gpu_schur_hess_nzpos,
        gpu_schur_diag_nzpos,
        gpu_schur_ineq_S_nzpos, gpu_schur_ineq_S_jcoo1, gpu_schur_ineq_S_jcoo2, gpu_schur_ineq_S_bufidx,
        gpu_schur_design_ineq_S_nzpos, gpu_schur_design_ineq_S_jcoo1, gpu_schur_design_ineq_S_jcoo2, gpu_schur_design_ineq_S_bufidx,
        gpu_design_var_global, gpu_scen_var_global,
        n_eq_total, CuVector{Int}(cpu_ind_eq),
        ns_ineq, CuVector{Int}(cpu_ind_ineq), CuVector{Int}(cpu_ind_lb), CuVector{Int}(cpu_ind_ub),
        scenario_solver, schur_solver,
        Base.RefValue{Any}(nothing),
    )
end

# --- Trivial accessors ---
MadNLP.num_variables(kkt::GPUSchurComplementCondensedKKTSystem) = size(kkt.hess_csc, 1)

function MadNLP.get_slack_regularization(kkt::GPUSchurComplementCondensedKKTSystem)
    n = MadNLP.num_variables(kkt)
    return view(kkt.pr_diag, n+1:n+kkt.n_ineq)
end

function MadNLP.is_inertia_correct(kkt::GPUSchurComplementCondensedKKTSystem, num_pos, num_zero, num_neg)
    # RelaxEquality-only: the first-stage Schur complement is SPD (nd positive
    # eigenvalues, no negative or zero ones).
    return (num_zero == 0) && (num_pos == kkt.nd) && (num_neg == 0)
end

MadNLP.should_regularize_dual(kkt::GPUSchurComplementCondensedKKTSystem, num_pos, num_zero, num_neg) = true

MadNLP.nnz_jacobian(kkt::GPUSchurComplementCondensedKKTSystem) = MadNLP.nnz(kkt.jt_coo)

function MadNLP.jtprod!(y::VT, kkt::GPUSchurComplementCondensedKKTSystem, x::VT) where {VT <: CuVector}
    nx = MadNLP.num_variables(kkt)
    ns_ineq = kkt.n_ineq
    yx = view(y, 1:nx)
    ys = view(y, 1+nx:nx+ns_ineq)
    mul!(yx, kkt.jt_csc, x)
    ys .= -@view(x[kkt.ind_ineq])
    return
end

MadNLP.compress_jacobian!(kkt::GPUSchurComplementCondensedKKTSystem) =
    _schur_compress_jacobian!(kkt, _get_det_scatter(kkt))   # function barrier (see build_kkt!)
function _schur_compress_jacobian!(kkt::GPUSchurComplementCondensedKKTSystem, det)
    # NOT the generic non-summing `transfer!` (`view(jt_csc.nzVal, map) .= jac`), which drops
    # duplicate Jacobian COO entries (last-write-wins). Deterministically SUM duplicates into
    # jt_csc.nzVal so jtprod!/mul!/dual recovery use the SAME Jacobian as the assembled
    # A_kk/C_dk/S blocks (which also sum duplicates). See the `jt_lin` comment in
    # `_compute_det_scatter`. Matches the CPU `_transfer!`, which sums with `+=`.
    backend = CUDABackend()
    nz = kkt.jt_csc.nzVal
    fill!(nz, zero(eltype(nz)))
    if length(det.jt_lin.segslot) > 0
        _det_lin_scatter!(backend)(
            nz, kkt.jac, det.jt_lin.src_idx, det.jt_lin.segstart, det.jt_lin.segslot;
            ndrange = length(det.jt_lin.segslot),
        )
    end
    return
end

MadNLP.compress_hessian!(kkt::GPUSchurComplementCondensedKKTSystem) =
    _schur_compress_hessian!(kkt, _get_det_scatter(kkt))   # function barrier (see build_kkt!)
function _schur_compress_hessian!(kkt::GPUSchurComplementCondensedKKTSystem, det)
    # NOT the generic non-summing `transfer!`: deterministically SUM duplicate Hessian COO
    # entries into hess_csc.nzVal, then materialize the full symmetric `hess_full` used by the
    # refinement mul!. (See the hess_full comment in `_compute_det_scatter`.)
    backend = CUDABackend()
    nz = kkt.hess_csc.nzVal
    fill!(nz, zero(eltype(nz)))
    if length(det.hess_lin.segslot) > 0
        _det_lin_scatter!(backend)(
            nz, kkt.hess, det.hess_lin.src_idx, det.hess_lin.segstart, det.hess_lin.segslot;
            ndrange = length(det.hess_lin.segslot),
        )
    end
    copyto!(det.hess_full.nzVal, det.hess_full_perm)
    return
end

# --- build_kkt! ---
# Function barrier: `_get_det_scatter` returns the memoized scatter as `::Any` (its concrete
# NamedTuple type is huge), so accessing `det.X` in-body would dynamic-dispatch every IPM
# iteration. Passing `det` to a specialized inner method makes every `det.X` access (and the
# kernel launches taking them) type-stable after a single dynamic dispatch at the call.
MadNLP.build_kkt!(kkt::GPUSchurComplementCondensedKKTSystem) = _schur_build_kkt!(kkt, _get_det_scatter(kkt))
function _schur_build_kkt!(kkt::GPUSchurComplementCondensedKKTSystem{T}, det) where T
    ns = kkt.ns
    nv = kkt.nv
    nd = kkt.nd
    m = kkt.m
    n = MadNLP.num_variables(kkt)
    blk = kkt.blk_size
    backend = CUDABackend()
    nzval = kkt.A_kk_batched.nzVal
    nnz_s = kkt.nnz_per_scenario
    Snz = kkt.schur_csc.nzVal

    # Compute condensing diagonal for inequalities
    if kkt.n_ineq > 0
        Sigma_s = view(kkt.pr_diag, n+1:n+kkt.n_ineq)
        Sigma_d = @view(kkt.du_diag[kkt.ind_ineq])
        kkt.diag_buffer .= Sigma_s ./ (one(T) .- Sigma_d .* Sigma_s)
    end

    # Zero out assembly targets (sparse Schur nzval + scenario block + reduced coupling)
    fill!(Snz, zero(T))
    fill!(nzval, zero(T))
    fill!(kkt.C_dk_batched, zero(T))

    # Scatter design-design Hessian (deterministic) and pr_diag diagonal into the sparse
    # Schur nzval. Design Hessian is large-valued + has duplicates → deterministic linear
    # scatter; pr_diag targets distinct diagonal slots → already race-free, left atomic.
    if length(det.hessS.segslot) > 0
        _det_lin_scatter!(backend)(
            Snz, kkt.hess, det.hessS.src_idx, det.hessS.segstart, det.hessS.segslot;
            ndrange = length(det.hessS.segslot),
        )
    end
    if nd > 0
        _scatter_to_csc_atomic!(backend)(
            Snz, kkt.pr_diag, kkt.design_var_global, kkt.schur_diag_nzpos; ndrange = nd,
        )
    end

    # Scatter Hessian diagonal → A_kk (deterministic linear scatter)
    if length(det.hessAkk.segslot) > 0
        _det_lin_scatter!(backend)(
            nzval, kkt.hess, det.hessAkk.src_idx, det.hessAkk.segstart, det.hessAkk.segslot;
            ndrange = length(det.hessAkk.segslot),
        )
    end

    # Scatter Hessian coupling → C_dk (deterministic linear scatter)
    if length(det.hessCdk.segslot) > 0
        _det_lin_scatter!(backend)(
            reshape(kkt.C_dk_batched, :), kkt.hess, det.hessCdk.src_idx, det.hessCdk.segstart, det.hessCdk.segslot;
            ndrange = length(det.hessCdk.segslot),
        )
    end

    # Scatter pr_diag → A_kk diagonal
    n_pr = kkt.n_per_s_pr_diag
    if n_pr > 0
        _scatter_to_Akk_batched!(backend)(
            nzval, kkt.pr_diag, kkt.gpu_pr_diag_global, kkt.gpu_pr_diag_nzpos,
            nnz_s, n_pr;
            ndrange=ns * n_pr,
        )
    end

    # Inequality condensation (Σ-amplified) → A_kk / C_dk / S, DETERMINISTIC (one thread per
    # output slot, fixed-order sum). Replaces the atomic-add scatters whose nondeterministic
    # summation order perturbed the blocks ~1e-7 run-to-run and made convergence a roulette.
    if length(det.Akk.segslot) > 0
        _det_quad_scatter!(backend)(
            nzval, kkt.jac, kkt.diag_buffer,
            det.Akk.jc1, det.Akk.jc2, det.Akk.buf, det.Akk.segstart, det.Akk.segslot;
            ndrange = length(det.Akk.segslot),
        )
    end
    if length(det.Cdk.segslot) > 0
        _det_quad_scatter!(backend)(
            reshape(kkt.C_dk_batched, :), kkt.jac, kkt.diag_buffer,
            det.Cdk.jc1, det.Cdk.jc2, det.Cdk.buf, det.Cdk.segstart, det.Cdk.segslot;
            ndrange = length(det.Cdk.segslot),
        )
    end
    # Scenario + design-only inequalities → S, merged into one deterministic scatter.
    if length(det.S.segslot) > 0
        _det_quad_scatter!(backend)(
            Snz, kkt.jac, kkt.diag_buffer,
            det.S.jc1, det.S.jc2, det.S.buf, det.S.segstart, det.S.segslot;
            ndrange = length(det.S.segslot),
        )
    end

    # Factorize all scenario blocks in one batched cuDSS call
    MadNLP.factorize!(kkt.scenario_solver)

    # Schur reduction restricted to the coupled-design block. Only the m design
    # columns that couple to a scenario contribute (the rest of C_dk is exactly
    # zero), so the reduction fills only the coupled × coupled sub-block.
    if m > 0
        # tmp_red = A_kk⁻¹ * C_dk_red'  (m right-hand sides per scenario).
        #
        # cuDSS's uniform-batch ("ubatch") solve is BROKEN for MULTI-RHS (nrhs > 1) once the
        # batch count nbatch ≳ 14: it returns garbage (off by ~1e13). Verified on case118
        # (ns=176) and reproduced in PURE CUDSS with a synthetic uniform batch — single-RHS
        # (nrhs=1) is always correct, multi-RHS fails above the threshold regardless of the
        # matrix (a synthetic pure-CUDSS batch reproduces it). cuDSS's own batched tests only
        # cover single-RHS, so this path was never exercised upstream.
        # So solve the m coupled columns one at a time with the (correct) single-RHS batched
        # solve — column j across all ns scenarios — mirroring the CPU path's column-by-column
        # `A_kk \ C_dk'`. Slower than one multi-RHS call, but correct. (Remove this loop and
        # restore the batched `scenario_*_multi` solve once cuDSS fixes multi-RHS ubatch.)
        for j in 1:m
            copyto!(reshape(kkt.solve_buffer, blk, ns), view(kkt.C_dk_batched, :, j, :))
            # `solve_linear_system!` already converts a cuDSS failure into a SolveException (after
            # logging the status), which the IPM maps to ERROR_IN_STEP_COMPUTATION. The old outer
            # try/catch that re-wrapped it as a FactorizationException was redundant double-handling
            # (and its `CUDSS.CUDSSError` branch was dead — never reached the outer scope).
            MadNLP.solve_linear_system!(kkt.scenario_solver, kkt.solve_buffer)
            copyto!(view(kkt.tmp_blk_nd_batched, :, j, :), reshape(kkt.solve_buffer, blk, ns))
        end

        # D[:,:,k] = C_dk_red[:,:,k]' * tmp_red[:,:,k] (m×m) for all k in ONE batched
        # GEMM (each batch slice is independent, so this is the correct batched form —
        # NOT the fused-reshape trick that scrambles d/k for ns>1). Then scatter the
        # lower-or-diagonal half as `S -= D` into the Schur nzval at the precomputed
        # fill positions.
        cuBLAS.gemm_strided_batched!(
            'T', 'N', one(T), kkt.C_dk_batched, kkt.tmp_blk_nd_batched,
            zero(T), kkt.schur_block_batched,
        )
        # Deterministic: one thread per lower-triangle (a,b) slot sums -D[a,b,k] over k.
        _det_schur_block!(backend)(
            Snz, kkt.schur_block_batched, kkt.schur_fill_nzpos, m, ns; ndrange = m * m,
        )
    end

    return
end

# --- factorize_kkt! ---
function MadNLP.factorize_kkt!(kkt::GPUSchurComplementCondensedKKTSystem)
    # cuDSS reads `nonzeros(schur_solver.tril)` (=== schur_csc.nzVal, populated by
    # build_kkt!) and runs factorization (first iter) / refactorization (later).
    return MadNLP.factorize!(kkt.linear_solver)
end

# --- solve_kkt! ---
function MadNLP.solve_kkt!(
    kkt::GPUSchurComplementCondensedKKTSystem{T},
    w::MadNLP.AbstractKKTVector{T},
) where T

    ns = kkt.ns
    nv = kkt.nv
    nd = kkt.nd
    m = kkt.m
    n = MadNLP.num_variables(kkt)
    blk = kkt.blk_size
    backend = CUDABackend()

    wx = view(MadNLP.full(w), 1:n)
    ws = view(MadNLP.full(w), n+1:n+kkt.n_ineq)
    wy = MadNLP.dual(w)

    Sigma_s = MadNLP.get_slack_regularization(kkt)

    MadNLP.reduce_rhs!(kkt, w)

    # Step 1: condense inequality contributions
    fill!(kkt.buffer, zero(T))
    if kkt.n_ineq > 0
        kkt.buffer[kkt.ind_ineq] .= kkt.diag_buffer .* (wy[kkt.ind_ineq] .+ ws ./ Sigma_s)
        mul!(wx, kkt.jt_csc, kkt.buffer, one(T), one(T))
    end

    # Step 2: Extract per-scenario RHS blocks via kernel.
    if blk * ns > 0
        _extract_rhs_kernel!(backend)(
            kkt.rhs_k_batched, wx, kkt.scen_var_global, nv, ns;
            ndrange=blk * ns,
        )
    end
    @views kkt.rhs_d[1:nd] .= wx[kkt.design_var_global]

    # Step 3: Forward elimination — batched solve
    copyto!(kkt.solve_buffer, vec(kkt.rhs_k_batched))
    MadNLP.solve_linear_system!(kkt.scenario_solver, kkt.solve_buffer)
    copyto!(vec(kkt.rhs_k_batched), kkt.solve_buffer)

    # rhs_d[coupled] -= Σ_k C_dk_red[:,:,k]' * rhs_k[:,k]. Only the coupled design rows receive a
    # contribution (non-coupled C_dk columns are exactly zero). One strided-batched GEMM computes
    # the per-scenario (m×1) products C_dk_red[:,:,k]' rhs_k[:,k]; sum over the batch dim gives the
    # accumulated reduced RHS, which is then scatter-subtracted. (Replaces the per-scenario `mul!`
    # loop = ns kernel launches per solve.)
    if m > 0
        cuBLAS.gemm_strided_batched!(
            'T', 'N', one(T), kkt.C_dk_batched, reshape(kkt.rhs_k_batched, blk, 1, ns),
            zero(T), kkt.rhs_d_red_batched,
        )
        sum!(reshape(kkt.rhs_d_red, m, 1, 1), kkt.rhs_d_red_batched)   # reduce over scenarios
        @views kkt.rhs_d[kkt.coupled_design_local] .-= kkt.rhs_d_red
    end

    # Step 4: Solve the first-stage Schur complement system (size nd, SPD) with cuDSS.
    MadNLP.solve_linear_system!(kkt.linear_solver, kkt.rhs_d)

    # Step 5: Back-substitution. rhs_k[:,k] -= tmp_red[:,:,k] * rhs_d[coupled] per scenario. The
    # reduced design update is shared across scenarios, so broadcast it into an (m×ns) buffer and
    # do the whole back-substitution as one strided-batched GEMM.
    if m > 0
        @views kkt.rhs_d_red .= kkt.rhs_d[kkt.coupled_design_local]
        kkt.rhs_d_red_mat .= reshape(kkt.rhs_d_red, m, 1)
        cuBLAS.gemm_strided_batched!(
            'N', 'N', -one(T), kkt.tmp_blk_nd_batched, reshape(kkt.rhs_d_red_mat, m, 1, ns),
            one(T), reshape(kkt.rhs_k_batched, blk, 1, ns),
        )
    end

    # Step 6: Write back to w via kernel
    if blk * ns > 0
        _writeback_rhs_kernel!(backend)(
            wx, kkt.rhs_k_batched, kkt.scen_var_global, nv, ns;
            ndrange=blk * ns,
        )
    end
    @views wx[kkt.design_var_global] .= kkt.rhs_d[1:nd]

    # Step 7: Recover inequality duals and slacks (all constraints are inequalities
    # under RelaxEquality, so there are no equality duals to preserve).
    if kkt.n_ineq > 0
        # J * Δx  (overwrites all of wy)
        mul!(wy, kkt.jt_csc', wx)

        # Inequality dual recovery
        wy[kkt.ind_ineq] .= kkt.diag_buffer .* wy[kkt.ind_ineq] .- kkt.buffer[kkt.ind_ineq]
        ws .= (ws .+ view(wy, kkt.ind_ineq)) ./ Sigma_s
    end

    MadNLP.finish_aug_solve!(kkt, w)
    return w
end

# --- mul! for iterative refinement ---
# Function barrier: hoist the `::Any` det access (its `hess_full` is the SpMV reference operator,
# hit many times per Richardson step) out of the hot body so the SpMV dispatches statically.
function MadNLP.mul!(
    w::MadNLP.AbstractKKTVector{T, VT},
    kkt::GPUSchurComplementCondensedKKTSystem{T},
    x::MadNLP.AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where {T, VT <: CuVector{T}}
    return _schur_mul!(w, kkt, x, alpha, beta, _get_det_scatter(kkt).hess_full)
end
function _schur_mul!(
    w::MadNLP.AbstractKKTVector{T, VT},
    kkt::GPUSchurComplementCondensedKKTSystem{T},
    x::MadNLP.AbstractKKTVector,
    alpha,
    beta,
    hess_full,
) where {T, VT <: CuVector{T}}
    n = MadNLP.num_variables(kkt)

    wx = @view(MadNLP.primal(w)[1:n])
    ws = @view(MadNLP.primal(w)[n+1:end])
    wy = MadNLP.dual(w)

    xx = @view(MadNLP.primal(x)[1:n])
    xs = @view(MadNLP.primal(x)[n+1:end])
    xz = @view(MadNLP.dual(x)[kkt.ind_ineq])

    wx .= beta .* wx
    # Use the correctly-summed full-symmetric Hessian (general SpMV) — NOT
    # Symmetric(hess_csc,:L), which CUSPARSE multiplies as lower-triangle-only.
    mul!(wx, hess_full, xx, alpha, one(T))

    m = size(kkt.jt_csc, 2)
    if m > 0
        mul!(wx, kkt.jt_csc, MadNLP.dual(x), alpha, one(T))
        mul!(wy, kkt.jt_csc', xx, alpha, beta)
    else
        wy .= beta .* wy
    end
    ws .= beta .* ws .- alpha .* xz
    @view(MadNLP.dual(w)[kkt.ind_ineq]) .-= alpha .* xs
    MadNLP._kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

MadNLP.mul_hess_blk!(wx::VT, kkt::GPUSchurComplementCondensedKKTSystem{T}, t) where {T, VT <: CuVector{T}} =
    _schur_mul_hess_blk!(wx, kkt, t, _get_det_scatter(kkt).hess_full)   # function barrier (see mul!)
function _schur_mul_hess_blk!(wx::VT, kkt::GPUSchurComplementCondensedKKTSystem{T}, t, hess_full) where {T, VT <: CuVector{T}}
    n = MadNLP.num_variables(kkt)
    mul!(@view(wx[1:n]), hess_full, @view(t[1:n]))
    fill!(@view(wx[n+1:end]), 0)
    wx .+= t .* kkt.pr_diag
end
