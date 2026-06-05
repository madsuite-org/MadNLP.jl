###########################################################
##### CUDA wrappers for SchurComplementKKTSystem ##########
###########################################################

"""
    GPUSchurComplementKKTSystem

GPU-native Schur complement KKT system, the CUDA counterpart of CPU
[`MadNLP.SchurComplementKKTSystem`](@ref). Uses a single batched
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
struct GPUSchurComplementKKTSystem{
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

    # Multi-RHS descriptors over `C_dk_batched` / `tmp_blk_nd_batched`, used to solve
    # A_kk * tmp[:, :, k] = C_dk_red[:, :, k] for all k in one cuDSS call (m right-hand
    # sides per scenario, ns batch). Zero-copy onto the `(blk, m, ns)` buffers.
    scenario_x_multi::CUDSS.CudssMatrix{T}
    scenario_b_multi::CUDSS.CudssMatrix{T}
end

# cuDSS sparse-LDL needs a step of iterative refinement to match the accuracy the
# dense cuSOLVER (`sytrf`) path had on the bordered Schur complement. Without it the
# complement solve is only ~1e-5 even at modest conditioning (cond≈3e7), while ONE IR
# step restores ~1e-12 (dense-level); 2-3 steps add nothing. So the GPU Schur path's
# cuDSS solvers — both the batched per-scenario blocks and the nbatch=1 complement —
# default to one IR step. Override via the `schur_*_opt_linear_solver` kwargs.
const SCHUR_DEFAULT_CUDSS_IR = 1
function _default_schur_cudss_options()
    opt = MadNLP.default_options(CUDSSSolver)
    opt.cudss_ir = SCHUR_DEFAULT_CUDSS_IR
    return opt
end

# --- Dispatch: GPU path when callback uses CuVector ---
function MadNLP.create_kkt_system(
    ::Type{MadNLP.SchurComplementKKTSystem},
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
        schur_scenario_opt_linear_solver = _default_schur_cudss_options(),
        schur_opt_linear_solver = _default_schur_cudss_options(),
) where {T, VT <: CuVector{T}}

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
    nd_aug = sym.nd_aug
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
        schur_colPtr, schur_rowVal, schur_nzVal, (nd_aug, nd_aug),
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
    rhs_d = CuVector{T}(undef, nd_aug)
    rhs_d_red = CuVector{T}(undef, m_coupled)
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
    scenario_solver = CUDSSSolver(A_kk_batched; opt=schur_scenario_opt_linear_solver)
    schur_solver = CUDSSSolver(schur_csc; opt = schur_opt_linear_solver)  # analysis runs once

    # --- Multi-RHS cuDSS descriptors for batched A_kk \ C_dk_red' (m RHS per scenario) ---
    mdesc = max(m_coupled, 1)
    scenario_b_multi = CUDSS.CudssMatrix(T, blk_size, mdesc; nbatch = ns)
    scenario_x_multi = CUDSS.CudssMatrix(T, blk_size, mdesc; nbatch = ns)
    # Re-analyze for the multi-RHS shape so cuDSS plans enough workspace.
    # This is the LARGEST RHS shape we will ever solve with on this handle;
    # the later single-RHS solve at `solve_kkt!` step 3 reuses the same handle
    # with a (blk × 1) × ns descriptor, which relies on the invariant that
    # "analysis planned for a larger RHS accepts a smaller RHS at solve time".
    # If that breaks on a future cuDSS version (e.g. a strict shape check or
    # per-column IR state), the single-RHS path would need its own handle or
    # padding. The `schur_cudss_ir` test exercises `cudss_ir > 0` — the config
    # most likely to surface such a regression — as a tripwire.
    CUDSS.cudss(
        "analysis", scenario_solver.inner,
        scenario_x_multi, scenario_b_multi;
        asynchronous=scenario_solver.opt.cudss_asynchronous,
    )

    return GPUSchurComplementKKTSystem(
        hess, jac,
        hess_raw, jt_coo,
        hess_csc, hess_csc_map, jt_csc, jt_csc_map,
        quasi_newton,
        reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower,
        ns, nv, nd, nc, nc_ineq_per_s, blk_size, m_coupled,
        A_kk_batched, nnz_per_scenario,
        C_dk_batched, schur_csc, schur_block_batched, gpu_coupled_design_local, gpu_schur_fill_nzpos,
        diag_buffer, buffer, rhs_d, rhs_d_red, rhs_k_batched, tmp_blk_nd_batched, solve_buffer,
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
        scenario_x_multi, scenario_b_multi,
    )
end

# --- Trivial accessors ---
MadNLP.num_variables(kkt::GPUSchurComplementKKTSystem) = size(kkt.hess_csc, 1)

function MadNLP.get_slack_regularization(kkt::GPUSchurComplementKKTSystem)
    n = MadNLP.num_variables(kkt)
    return view(kkt.pr_diag, n+1:n+kkt.n_ineq)
end

function MadNLP.is_inertia_correct(kkt::GPUSchurComplementKKTSystem, num_pos, num_zero, num_neg)
    # RelaxEquality-only: the first-stage Schur complement is SPD (nd positive
    # eigenvalues, no negative or zero ones).
    return (num_zero == 0) && (num_pos == kkt.nd) && (num_neg == 0)
end
# the Schur complement is definite: the duals are eliminated, not factorized
MadNLP.block_structure(kkt::GPUSchurComplementKKTSystem) = (size(kkt.aug_com, 1), 0)

MadNLP.should_regularize_dual(kkt::GPUSchurComplementKKTSystem, num_pos, num_zero, num_neg) = true

MadNLP.nnz_jacobian(kkt::GPUSchurComplementKKTSystem) = MadNLP.nnz(kkt.jt_coo)

function MadNLP.jtprod!(y::VT, kkt::GPUSchurComplementKKTSystem, x::VT) where {VT <: CuVector}
    nx = MadNLP.num_variables(kkt)
    ns_ineq = kkt.n_ineq
    yx = view(y, 1:nx)
    ys = view(y, 1+nx:nx+ns_ineq)
    mul!(yx, kkt.jt_csc, x)
    ys .= -@view(x[kkt.ind_ineq])
    return
end

function MadNLP.compress_jacobian!(kkt::GPUSchurComplementKKTSystem)
    MadNLP.transfer!(kkt.jt_csc, kkt.jt_coo, kkt.jt_csc_map)
end

function MadNLP.compress_hessian!(kkt::GPUSchurComplementKKTSystem)
    MadNLP.transfer!(kkt.hess_csc, kkt.hess_raw, kkt.hess_csc_map)
end

# --- build_kkt! ---
function MadNLP.build_kkt!(kkt::GPUSchurComplementKKTSystem{T}) where T
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

    # Scatter design-design Hessian (lower-only) and pr_diag diagonal into the
    # sparse Schur nzval. Atomic: overlapping contributions can share an nzval slot.
    n_hess_S = length(kkt.schur_hess_coo)
    if n_hess_S > 0
        _scatter_to_csc_atomic!(backend)(
            Snz, kkt.hess, kkt.schur_hess_coo, kkt.schur_hess_nzpos; ndrange = n_hess_S,
        )
    end
    if nd > 0
        _scatter_to_csc_atomic!(backend)(
            Snz, kkt.pr_diag, kkt.design_var_global, kkt.schur_diag_nzpos; ndrange = nd,
        )
    end

    # Scatter Hessian diagonal → A_kk (flattened maps)
    n_hA = kkt.n_per_s_hess_Akk
    if n_hA > 0
        _scatter_to_Akk_batched!(backend)(
            nzval, kkt.hess, kkt.gpu_hess_Akk_coo, kkt.gpu_hess_Akk_nzpos,
            nnz_s, n_hA;
            ndrange=ns * n_hA,
        )
    end

    # Scatter Hessian coupling → C_dk
    n_hC = kkt.n_per_s_hess_Cdk
    if n_hC > 0
        _scatter_to_Cdk_batched!(backend)(
            kkt.C_dk_batched, kkt.hess, kkt.gpu_hess_Cdk_coo,
            kkt.gpu_hess_Cdk_col, kkt.gpu_hess_Cdk_row, n_hC;
            ndrange=ns * n_hC,
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

    # Inequality condensation → A_kk
    n_iA = kkt.n_per_s_ineq_Akk
    if n_iA > 0
        _ineq_condense_Akk_kernel!(backend)(
            nzval, kkt.jac, kkt.diag_buffer,
            kkt.gpu_ineq_Akk_nzpos, kkt.gpu_ineq_Akk_jcoo1,
            kkt.gpu_ineq_Akk_jcoo2, kkt.gpu_ineq_Akk_bufidx,
            nnz_s, n_iA;
            ndrange=ns * n_iA,
        )
    end

    # Inequality condensation → C_dk
    n_iC = kkt.n_per_s_ineq_Cdk
    if n_iC > 0
        _ineq_condense_Cdk_kernel!(backend)(
            kkt.C_dk_batched, kkt.jac, kkt.diag_buffer,
            kkt.gpu_ineq_Cdk_col, kkt.gpu_ineq_Cdk_row,
            kkt.gpu_ineq_Cdk_jcoo_d, kkt.gpu_ineq_Cdk_jcoo_v,
            kkt.gpu_ineq_Cdk_bufidx, n_iC;
            ndrange=ns * n_iC,
        )
    end

    # Inequality condensation → Schur nzval (scenario, flattened lower-only over all scenarios).
    n_iS = length(kkt.schur_ineq_S_nzpos)
    if n_iS > 0
        _ineq_condense_csc_kernel!(backend)(
            Snz, kkt.jac, kkt.diag_buffer,
            kkt.schur_ineq_S_nzpos, kkt.schur_ineq_S_jcoo1,
            kkt.schur_ineq_S_jcoo2, kkt.schur_ineq_S_bufidx;
            ndrange = n_iS,
        )
    end

    # Design-only inequalities: condense into S (lower-only) in the sparse nzval.
    n_diS = length(kkt.schur_design_ineq_S_nzpos)
    if n_diS > 0
        _ineq_condense_csc_kernel!(backend)(
            Snz, kkt.jac, kkt.diag_buffer,
            kkt.schur_design_ineq_S_nzpos, kkt.schur_design_ineq_S_jcoo1,
            kkt.schur_design_ineq_S_jcoo2, kkt.schur_design_ineq_S_bufidx;
            ndrange = n_diS,
        )
    end

    # Factorize all scenario blocks in one batched cuDSS call
    MadNLP.factorize!(kkt.scenario_solver)

    # Schur reduction restricted to the coupled-design block. Only the m design
    # columns that couple to a scenario contribute (the rest of C_dk is exactly
    # zero), so the reduction fills only the coupled × coupled sub-block.
    if m > 0
        # tmp_red = A_kk⁻¹ * C_dk_red' for all scenarios in one batched multi-RHS
        # cuDSS solve (m RHS per scenario). Zero-copy onto the (blk, m, ns) buffers.
        CUDSS.cudss_update(kkt.scenario_b_multi, kkt.C_dk_batched)
        CUDSS.cudss_update(kkt.scenario_x_multi, kkt.tmp_blk_nd_batched)
        CUDSS.cudss(
            "solve", kkt.scenario_solver.inner,
            kkt.scenario_x_multi, kkt.scenario_b_multi;
            asynchronous = kkt.scenario_solver.opt.cudss_asynchronous,
        )

        # D[:,:,k] = C_dk_red[:,:,k]' * tmp_red[:,:,k] (m×m) for all k in ONE batched
        # GEMM (each batch slice is independent, so this is the correct batched form —
        # NOT the fused-reshape trick that scrambles d/k for ns>1). Then scatter the
        # lower-or-diagonal half as `S -= D` into the Schur nzval at the precomputed
        # fill positions.
        cuBLAS.gemm_strided_batched!(
            'T', 'N', one(T), kkt.C_dk_batched, kkt.tmp_blk_nd_batched,
            zero(T), kkt.schur_block_batched,
        )
        _scatter_schur_block!(backend)(
            Snz, kkt.schur_block_batched, kkt.schur_fill_nzpos, m, ns; ndrange = m * m * ns,
        )
    end

    return
end

# --- factorize_kkt! ---
function MadNLP.factorize_kkt!(kkt::GPUSchurComplementKKTSystem)
    # cuDSS reads `nonzeros(schur_solver.tril)` (=== schur_csc.nzVal, populated by
    # build_kkt!) and runs factorization (first iter) / refactorization (later).
    return MadNLP.factorize!(kkt.linear_solver)
end

# --- solve_kkt! ---
function MadNLP.solve_kkt!(
    kkt::GPUSchurComplementKKTSystem{T},
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

    # rhs_d[coupled] -= Σ_k C_dk_red[:,:,k]' * rhs_k[:,k]. Only the coupled design
    # rows receive a contribution (non-coupled C_dk columns are exactly zero); the
    # design eq tail (the J0_eq row residual) is untouched by scenario elimination.
    # Accumulate the sum in the reduced buffer, then scatter-subtract.
    if m > 0
        fill!(kkt.rhs_d_red, zero(T))
        for k in 1:ns
            mul!(kkt.rhs_d_red, view(kkt.C_dk_batched, :, :, k)', view(kkt.rhs_k_batched, :, k), one(T), one(T))
        end
        @views kkt.rhs_d[kkt.coupled_design_local] .-= kkt.rhs_d_red
    end

    # Step 4: Solve the first-stage Schur complement system (size nd, SPD) with cuDSS.
    MadNLP.solve_linear_system!(kkt.linear_solver, kkt.rhs_d)

    # Step 5: Back-substitution. rhs_k[:,k] -= tmp_red[:,:,k] * rhs_d[coupled] per scenario.
    if m > 0
        @views kkt.rhs_d_red .= kkt.rhs_d[kkt.coupled_design_local]
        for k in 1:ns
            mul!(view(kkt.rhs_k_batched, :, k), view(kkt.tmp_blk_nd_batched, :, :, k), kkt.rhs_d_red, -one(T), one(T))
        end
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
function MadNLP.mul!(
    w::MadNLP.AbstractKKTVector{T, VT},
    kkt::GPUSchurComplementKKTSystem{T},
    x::MadNLP.AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where {T, VT <: CuVector{T}}
    n = MadNLP.num_variables(kkt)

    wx = @view(MadNLP.primal(w)[1:n])
    ws = @view(MadNLP.primal(w)[n+1:end])
    wy = MadNLP.dual(w)

    xx = @view(MadNLP.primal(x)[1:n])
    xs = @view(MadNLP.primal(x)[n+1:end])
    xz = @view(MadNLP.dual(x)[kkt.ind_ineq])

    wx .= beta .* wx
    mul!(wx, Symmetric(kkt.hess_csc, :L), xx, alpha, one(T))

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

function MadNLP.mul_hess_blk!(wx::VT, kkt::GPUSchurComplementKKTSystem{T}, t) where {T, VT <: CuVector{T}}
    n = MadNLP.num_variables(kkt)
    mul!(@view(wx[1:n]), Symmetric(kkt.hess_csc, :L), @view(t[1:n]))
    fill!(@view(wx[n+1:end]), 0)
    wx .+= t .* kkt.pr_diag
end
