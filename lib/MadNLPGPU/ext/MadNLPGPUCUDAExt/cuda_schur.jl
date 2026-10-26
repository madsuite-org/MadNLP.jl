###########################################################
##### CUDA wrappers for SchurComplementKKTSystem ##########
###########################################################

"""
    GPUSchurComplementKKTSystem

GPU-native Schur complement KKT system, the CUDA counterpart of CPU
[`MadNLP.SchurComplementKKTSystem`](@ref). Uses a single batched
`CuSparseMatrixCSC` holding all `ns` scenario blocks (factored by `CUDSSSolver`
with uniform batching) and CUBLAS strided batched GEMM for the Schur complement
accumulation.

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
    LS,   # linear solver for Schur complement S (LapackCUDASolver)
    LS2,  # batched scenario solver (CUDSSSolver)
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
    nc_eq_per_s::Int
    nc_ineq_per_s::Int
    blk_size::Int

    # Batched scenario block
    A_kk_batched::CSC_T
    nnz_per_scenario::Int

    # Dense blocks
    C_dk_batched::CuArray{T, 3}     # (blk_size, nd, ns) — per-scenario cross-block stored
                                    # with scenario-var dim first, so each slice
                                    # C_dk[:, :, k] is directly usable as the cuDSS
                                    # multi-RHS input for `A_kk \ C_dk'`.
    aug_com::MT                     # (nd, nd) — Schur complement S

    # Buffers
    diag_buffer::VT
    buffer::VT
    wy_eq_buf::VT                       # ns*nc_eq_per_s — preserves eq duals across J*Δx round-trip
    rhs_d::VT
    rhs_k_batched::MT                   # (blk_size, ns)
    tmp_blk_nd_batched::CuArray{T, 3}   # (blk_size, nd, ns)
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

    n_per_s_du_diag::Int
    gpu_du_diag_global::VI
    gpu_du_diag_nzpos::VI

    n_per_s_jeq_Akk::Int
    gpu_jeq_Akk_coo::VI
    gpu_jeq_Akk_nzpos::VI

    n_per_s_jeq_Cdk::Int
    gpu_jeq_Cdk_coo::VI
    gpu_jeq_Cdk_row::VI
    gpu_jeq_Cdk_col::VI

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

    n_per_s_ineq_S::Int
    gpu_ineq_S_row::VI
    gpu_ineq_S_col::VI
    gpu_ineq_S_jcoo1::VI
    gpu_ineq_S_jcoo2::VI
    gpu_ineq_S_bufidx::VI

    hess_S_coo::VI
    hess_S_row::VI
    hess_S_col::VI

    eq_global_indices::VI

    # Inequality/equality/bound indices
    n_eq::Int
    ind_eq::VI
    n_ineq::Int
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI

    # Solvers
    scenario_solver::LS2
    linear_solver::LS

    # Multi-RHS descriptors over `C_dk_batched` / `tmp_blk_nd_batched`, used to
    # solve A_kk * tmp[:, :, k] = C_dk[:, :, k] for all k in a single cuDSS
    # call (nd right-hand sides per scenario, ns batch). Zero-copy: both are
    # `cudss_update`'d onto the existing `(blk, nd, ns)` buffers each iteration.
    scenario_x_multi::CUDSS.CudssMatrix{T}
    scenario_b_multi::CUDSS.CudssMatrix{T}
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
    schur_scenario_opt_linear_solver=MadNLP.default_options(CUDSSSolver),
) where {T, VT <: CuVector{T}}

    n = cb.nvar
    m = cb.ncon
    ns_ineq = length(cb.ind_ineq)
    n_eq_total = m - ns_ineq
    nlb = length(cb.ind_lb)
    nub = length(cb.ind_ub)

    ns, nv, nd, nc = MadNLP._resolve_schur_dims(cb, n, m, schur_ns, schur_nv, schur_nd, schur_nc)

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
    )
    nc_eq_per_s = sym.nc_eq_per_s
    nc_ineq_per_s = sym.nc_ineq_per_s
    blk_size = sym.blk_size
    nnz_per_scenario = sym.nnz_per_scenario
    akk_csc_cpu = sym.akk_csc_template

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

    # --- Dense arrays on GPU ---
    aug_com = CuMatrix{T}(undef, nd, nd)
    fill!(aug_com, zero(T))
    C_dk_batched = CuArray{T, 3}(undef, blk_size, nd, ns)
    fill!(C_dk_batched, zero(T))
    tmp_blk_nd_batched = CuArray{T, 3}(undef, blk_size, nd, ns)

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
    # Size matches the eq_global_indices gather/scatter shape used in
    # solve_kkt! step 7. ns * nc_eq_per_s == n_eq_total under the uniform-
    # scenario invariant validated in _build_schur_symbolic.
    wy_eq_buf   = CuVector{T}(undef, ns * nc_eq_per_s)
    rhs_d       = CuVector{T}(undef, nd)
    rhs_k_batched = CuMatrix{T}(undef, blk_size, ns)
    solve_buffer = CuVector{T}(undef, blk_size * ns)

    # --- Transfer flattened index maps to GPU ---
    gpu_hess_Akk_coo    = CuVector{Int}(flat.all_hess_Akk_coo)
    gpu_hess_Akk_nzpos  = CuVector{Int}(flat.all_hess_Akk_nzpos)
    gpu_hess_Cdk_coo    = CuVector{Int}(flat.all_hess_Cdk_coo)
    gpu_hess_Cdk_row    = CuVector{Int}(flat.all_hess_Cdk_row)
    gpu_hess_Cdk_col    = CuVector{Int}(flat.all_hess_Cdk_col)
    gpu_pr_diag_global  = CuVector{Int}(flat.all_pr_diag_global)
    gpu_pr_diag_nzpos   = CuVector{Int}(flat.all_pr_diag_nzpos)
    gpu_du_diag_global  = CuVector{Int}(flat.all_du_diag_global)
    gpu_du_diag_nzpos   = CuVector{Int}(flat.all_du_diag_nzpos)
    gpu_jeq_Akk_coo     = CuVector{Int}(flat.all_jeq_Akk_coo)
    gpu_jeq_Akk_nzpos   = CuVector{Int}(flat.all_jeq_Akk_nzpos)
    gpu_jeq_Cdk_coo     = CuVector{Int}(flat.all_jeq_Cdk_coo)
    gpu_jeq_Cdk_row     = CuVector{Int}(flat.all_jeq_Cdk_row)
    gpu_jeq_Cdk_col     = CuVector{Int}(flat.all_jeq_Cdk_col)
    gpu_ineq_Akk_nzpos  = CuVector{Int}(flat.all_ineq_Akk_nzpos)
    gpu_ineq_Akk_jcoo1  = CuVector{Int}(flat.all_ineq_Akk_jcoo1)
    gpu_ineq_Akk_jcoo2  = CuVector{Int}(flat.all_ineq_Akk_jcoo2)
    gpu_ineq_Akk_bufidx = CuVector{Int}(flat.all_ineq_Akk_bufidx)
    gpu_ineq_Cdk_row    = CuVector{Int}(flat.all_ineq_Cdk_row)
    gpu_ineq_Cdk_col    = CuVector{Int}(flat.all_ineq_Cdk_col)
    gpu_ineq_Cdk_jcoo_d = CuVector{Int}(flat.all_ineq_Cdk_jcoo_d)
    gpu_ineq_Cdk_jcoo_v = CuVector{Int}(flat.all_ineq_Cdk_jcoo_v)
    gpu_ineq_Cdk_bufidx = CuVector{Int}(flat.all_ineq_Cdk_bufidx)
    gpu_ineq_S_row      = CuVector{Int}(flat.all_ineq_S_row)
    gpu_ineq_S_col      = CuVector{Int}(flat.all_ineq_S_col)
    gpu_ineq_S_jcoo1    = CuVector{Int}(flat.all_ineq_S_jcoo1)
    gpu_ineq_S_jcoo2    = CuVector{Int}(flat.all_ineq_S_jcoo2)
    gpu_ineq_S_bufidx   = CuVector{Int}(flat.all_ineq_S_bufidx)
    gpu_hess_S_coo      = CuVector{Int}(sym.hess_S_coo)
    gpu_hess_S_row      = CuVector{Int}(sym.hess_S_row)
    gpu_hess_S_col      = CuVector{Int}(sym.hess_S_col)
    gpu_eq_global_indices = CuVector{Int}(sym.eq_global_flat)

    # --- Create solvers ---
    quasi_newton = MadNLP.create_quasi_newton(hessian_approximation, cb, n; options=qn_options)
    # cuDSS is the only sparse batched solver wired into this path; users tune it
    # via `schur_scenario_opt_linear_solver` rather than swapping the type.
    scenario_solver = CUDSSSolver(A_kk_batched; opt=schur_scenario_opt_linear_solver)
    _linear_solver = linear_solver(aug_com; opt=opt_linear_solver)

    # --- Multi-RHS cuDSS descriptors for batched A_kk \ C_dk' ---
    # The descriptors hold the (blk × nd) shape per scenario and point at the
    # existing buffers each iteration.
    scenario_b_multi = CUDSS.CudssMatrix(T, blk_size, nd; nbatch=ns)
    scenario_x_multi = CUDSS.CudssMatrix(T, blk_size, nd; nbatch=ns)
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
        ns, nv, nd, nc, nc_eq_per_s, nc_ineq_per_s, blk_size,
        A_kk_batched, nnz_per_scenario,
        C_dk_batched, aug_com,
        diag_buffer, buffer, wy_eq_buf, rhs_d, rhs_k_batched, tmp_blk_nd_batched, solve_buffer,
        flat.n_per_s_hess_Akk, gpu_hess_Akk_coo, gpu_hess_Akk_nzpos,
        flat.n_per_s_hess_Cdk, gpu_hess_Cdk_coo, gpu_hess_Cdk_row, gpu_hess_Cdk_col,
        flat.n_per_s_pr_diag, gpu_pr_diag_global, gpu_pr_diag_nzpos,
        flat.n_per_s_du_diag, gpu_du_diag_global, gpu_du_diag_nzpos,
        flat.n_per_s_jeq_Akk, gpu_jeq_Akk_coo, gpu_jeq_Akk_nzpos,
        flat.n_per_s_jeq_Cdk, gpu_jeq_Cdk_coo, gpu_jeq_Cdk_row, gpu_jeq_Cdk_col,
        flat.n_per_s_ineq_Akk, gpu_ineq_Akk_nzpos, gpu_ineq_Akk_jcoo1, gpu_ineq_Akk_jcoo2, gpu_ineq_Akk_bufidx,
        flat.n_per_s_ineq_Cdk, gpu_ineq_Cdk_row, gpu_ineq_Cdk_col, gpu_ineq_Cdk_jcoo_d, gpu_ineq_Cdk_jcoo_v, gpu_ineq_Cdk_bufidx,
        flat.n_per_s_ineq_S, gpu_ineq_S_row, gpu_ineq_S_col, gpu_ineq_S_jcoo1, gpu_ineq_S_jcoo2, gpu_ineq_S_bufidx,
        gpu_hess_S_coo, gpu_hess_S_row, gpu_hess_S_col,
        gpu_eq_global_indices,
        n_eq_total, CuVector{Int}(cpu_ind_eq),
        ns_ineq, CuVector{Int}(cpu_ind_ineq), CuVector{Int}(cpu_ind_lb), CuVector{Int}(cpu_ind_ub),
        scenario_solver, _linear_solver,
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
    return (num_zero == 0) && (num_pos == size(kkt.aug_com, 1))
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
    n = MadNLP.num_variables(kkt)
    blk = kkt.blk_size
    backend = CUDABackend()
    nzval = kkt.A_kk_batched.nzVal
    nnz_s = kkt.nnz_per_scenario

    # Compute condensing diagonal for inequalities
    if kkt.n_ineq > 0
        Sigma_s = view(kkt.pr_diag, n+1:n+kkt.n_ineq)
        Sigma_d = @view(kkt.du_diag[kkt.ind_ineq])
        kkt.diag_buffer .= Sigma_s ./ (one(T) .- Sigma_d .* Sigma_s)
    end

    # Zero out assembly targets
    fill!(kkt.aug_com, zero(T))
    fill!(nzval, zero(T))
    fill!(kkt.C_dk_batched, zero(T))

    # Initialize S from design-design Hessian
    n_hess_S = length(kkt.hess_S_coo)
    if n_hess_S > 0
        _init_S_hess_kernel!(backend)(
            kkt.aug_com, kkt.hess, kkt.hess_S_coo, kkt.hess_S_row, kkt.hess_S_col;
            ndrange=n_hess_S,
        )
    end
    if nd > 0
        _init_S_diag_kernel!(backend)(kkt.aug_com, kkt.pr_diag, ns * nv, nd; ndrange=nd)
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

    # Scatter du_diag → A_kk diagonal
    n_du = kkt.n_per_s_du_diag
    if n_du > 0
        _scatter_to_Akk_batched!(backend)(
            nzval, kkt.du_diag, kkt.gpu_du_diag_global, kkt.gpu_du_diag_nzpos,
            nnz_s, n_du;
            ndrange=ns * n_du,
        )
    end

    # Scatter equality Jacobian → A_kk
    n_jA = kkt.n_per_s_jeq_Akk
    if n_jA > 0
        _scatter_to_Akk_batched!(backend)(
            nzval, kkt.jac, kkt.gpu_jeq_Akk_coo, kkt.gpu_jeq_Akk_nzpos,
            nnz_s, n_jA;
            ndrange=ns * n_jA,
        )
    end

    # Scatter equality Jacobian coupling → C_dk
    n_jC = kkt.n_per_s_jeq_Cdk
    if n_jC > 0
        _scatter_to_Cdk_batched!(backend)(
            kkt.C_dk_batched, kkt.jac, kkt.gpu_jeq_Cdk_coo,
            kkt.gpu_jeq_Cdk_col, kkt.gpu_jeq_Cdk_row, n_jC;
            ndrange=ns * n_jC,
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

    # Inequality condensation → S (atomic adds)
    n_iS = kkt.n_per_s_ineq_S
    if n_iS > 0
        _ineq_condense_S_kernel!(backend)(
            kkt.aug_com, kkt.jac, kkt.diag_buffer,
            kkt.gpu_ineq_S_row, kkt.gpu_ineq_S_col,
            kkt.gpu_ineq_S_jcoo1, kkt.gpu_ineq_S_jcoo2,
            kkt.gpu_ineq_S_bufidx;
            ndrange=ns * n_iS,
        )
    end

    # Factorize all scenario blocks in one batched cuDSS call
    MadNLP.factorize!(kkt.scenario_solver)

    # Compute tmp = A_kk^{-1} * C_dk' in one batched multi-RHS cuDSS solve:
    # for each scenario k, solve the (blk × blk) system with nd right-hand sides
    # packed as the columns of C_dk_batched[:, :, k]. The (blk, nd, ns) layouts
    # of both buffers line up directly with a cuDSS batched dense descriptor, so
    # this is zero-copy — we only retarget the matrix descriptors each iteration.
    CUDSS.cudss_update(kkt.scenario_b_multi, kkt.C_dk_batched)
    CUDSS.cudss_update(kkt.scenario_x_multi, kkt.tmp_blk_nd_batched)
    CUDSS.cudss(
        "solve", kkt.scenario_solver.inner,
        kkt.scenario_x_multi, kkt.scenario_b_multi;
        asynchronous=kkt.scenario_solver.opt.cudss_asynchronous,
    )

    # S -= Σ_k C_dk[:,:,k]' * tmp[:,:,k]
    # Reshape the (blk, nd, ns) buffers as (blk*ns, nd): column-major flattening
    # collapses the per-scenario contributions into a single GEMM, since
    # (C_2d' * tmp_2d)[a, b] = Σ_{i,k} C[i,a,k] * tmp[i,b,k] = Σ_k (C_dk[:,:,k]' * tmp[:,:,k])[a, b].
    # Avoids both the (nd × nd × ns) S_contrib buffer and the per-iteration
    # `dropdims(sum(...))` allocation that the batched-GEMM path required.
    C_2d = reshape(kkt.C_dk_batched, blk * ns, nd)
    tmp_2d = reshape(kkt.tmp_blk_nd_batched, blk * ns, nd)
    mul!(kkt.aug_com, C_2d', tmp_2d, -one(T), one(T))

    return
end

# --- factorize_kkt! ---
function MadNLP.factorize_kkt!(kkt::GPUSchurComplementKKTSystem)
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
    n = MadNLP.num_variables(kkt)
    blk = kkt.blk_size
    nc_eq = kkt.nc_eq_per_s
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

    # Step 2: Extract per-scenario RHS blocks via kernel
    if blk * ns > 0
        _extract_rhs_kernel!(backend)(
            kkt.rhs_k_batched, wx, wy, kkt.eq_global_indices,
            nv, nc_eq, ns, blk;
            ndrange=blk * ns,
        )
    end
    copyto!(kkt.rhs_d, view(wx, ns*nv+1:ns*nv+nd))

    # Step 3: Forward elimination — batched solve
    copyto!(kkt.solve_buffer, vec(kkt.rhs_k_batched))
    MadNLP.solve_linear_system!(kkt.scenario_solver, kkt.solve_buffer)
    copyto!(vec(kkt.rhs_k_batched), kkt.solve_buffer)

    # rhs_d -= Σ_k C_dk[:,:,k]' * rhs_k[:,k]: same column-major reshape trick as
    # build_kkt!. C_2d' * vec(rhs_k_batched) == Σ_k C_dk[:,:,k]' * rhs_k[:,k].
    # One launch instead of ns.
    C_2d = reshape(kkt.C_dk_batched, blk * ns, nd)
    mul!(kkt.rhs_d, C_2d', vec(kkt.rhs_k_batched), -one(T), one(T))

    # Step 4: Solve Schur complement
    MadNLP.solve_linear_system!(kkt.linear_solver, kkt.rhs_d)

    # Step 5: Back-substitution. rhs_k[:,k] -= tmp[:,:,k] * rhs_d for all k;
    # reshape tmp as (blk*ns, nd) so a single GEMV updates the flattened
    # rhs_k_batched in place.
    tmp_2d = reshape(kkt.tmp_blk_nd_batched, blk * ns, nd)
    mul!(vec(kkt.rhs_k_batched), tmp_2d, kkt.rhs_d, -one(T), one(T))

    # Step 6: Write back to w via kernel
    if blk * ns > 0
        _writeback_rhs_kernel!(backend)(
            wx, wy, kkt.rhs_k_batched, kkt.eq_global_indices,
            nv, nc_eq, ns, blk;
            ndrange=blk * ns,
        )
    end
    copyto!(view(wx, ns*nv+1:ns*nv+nd), kkt.rhs_d)

    # Step 7: Recover inequality duals and slacks
    if kkt.n_ineq > 0
        # Stash eq duals; mul! below overwrites all of wy.
        # Vectorized GPU gather/scatter — no @allowscalar.
        copyto!(kkt.wy_eq_buf, view(wy, kkt.eq_global_indices))

        # J * Δx
        mul!(wy, kkt.jt_csc', wx)

        # Restore equality duals
        view(wy, kkt.eq_global_indices) .= kkt.wy_eq_buf

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
