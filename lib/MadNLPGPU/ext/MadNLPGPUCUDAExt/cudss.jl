MadNLP.@kwdef mutable struct CudssSolverOptions <: MadNLP.AbstractOptions
    # Use LDLᵀ by default in CUDSS as Cholesky can lead to undefined behavior.
    cudss_algorithm::MadNLP.LinearFactorization = MadNLP.LDL
    cudss_ordering::ORDERING = DEFAULT_ORDERING
    cudss_perm::Vector{Cint} = Cint[]
    cudss_ir::Int = 0
    cudss_ir_tol::Float64 = 1e-8
    cudss_pivot_threshold::Float64 = 0.0
    cudss_pivot_epsilon::Float64 = 0.0
    cudss_matching_alg::String = "default"
    cudss_reordering_alg::String = "default"
    cudss_factorization_alg::String = "default"
    cudss_solve_alg::String = "default"
    cudss_matching::Bool = false
    cudss_pivoting::Bool = true
    cudss_hybrid_execute::Bool = false
    cudss_hybrid_memory::Bool = false
    cudss_hybrid_memory_limit::Int = 0
    cudss_superpanels::Bool = true
    cudss_schur::Bool = false
    cudss_deterministic::Bool = false
    cudss_device_indices::Vector{Cint} = Cint[]
    cudss_asynchronous::Bool = true
end

function set_cudss_options!(solver::CUDSS.CudssSolver, opt::CudssSolverOptions)
    if opt.cudss_ir > 0
        CUDSS.cudss_set(solver, "ir_n_steps", opt.cudss_ir)
        CUDSS.cudss_set(solver, "ir_tol", opt.cudss_ir_tol)
    end
    if opt.cudss_hybrid_memory
        CUDSS.cudss_set(solver, "hybrid_memory_mode", 1)
        if opt.cudss_hybrid_memory_limit > 0
            CUDSS.cudss_set(solver, "hybrid_device_memory_limit", opt.cudss_hybrid_memory_limit)
        end
    end
    if opt.cudss_hybrid_execute
        CUDSS.cudss_set(solver, "hybrid_execute_mode", 1)
    end
    if !opt.cudss_pivoting
        CUDSS.cudss_set(solver, "pivot_type", 'N')
    end
    if opt.cudss_pivot_epsilon > 0.0
        CUDSS.cudss_set(solver, "pivot_epsilon", opt.cudss_pivot_epsilon)
    end
    if opt.cudss_pivot_threshold > 0.0
        CUDSS.cudss_set(solver, "pivot_threshold", opt.cudss_pivot_threshold)
    end
    if opt.cudss_matching
        if solver.matrix.nbatch > 1
            # cuDSS matching (`matching_alg`) fails the *analysis* phase with
            # CUDSS_STATUS_NOT_SUPPORTED on a uniform-batch solver (through cuDSS 0.8),
            # so never enable it there (e.g. the two-stage per-scenario batch solver).
            Base.@warn "cuDSS matching is not supported on uniform-batch (ubatch) solvers; ignoring `cudss_matching = true` for this batched solver." maxlog = 1
        elseif pkgversion(CUDSS) < v"0.8"
            CUDSS.cudss_set(solver, "use_matching", 1)
            if opt.cudss_matching_alg != "default"
                CUDSS.cudss_set(solver, "matching_alg", opt.cudss_matching_alg)
            end
        else
            # cuDSS 0.8 dropped "use_matching": matching is enabled by selecting a
            # matching algorithm ("default" disables it), so fall back to "algo1".
            matching_alg = opt.cudss_matching_alg == "default" ? "algo1" : opt.cudss_matching_alg
            CUDSS.cudss_set(solver, "matching_alg", matching_alg)
        end
    end
    if opt.cudss_reordering_alg != "default"
        CUDSS.cudss_set(solver, "reordering_alg", opt.cudss_reordering_alg)
    end
    if opt.cudss_factorization_alg != "default"
        CUDSS.cudss_set(solver, "factorization_alg", opt.cudss_factorization_alg)
    end
    if opt.cudss_solve_alg != "default"
        CUDSS.cudss_set(solver, "solve_alg", opt.cudss_solve_alg)
    end
    if !opt.cudss_superpanels
        CUDSS.cudss_set(solver, "use_superpanels", 0)
    end
    if opt.cudss_schur
        CUDSS.cudss_set(solver, "schur_mode", 1)
    end
    if opt.cudss_deterministic
        CUDSS.cudss_set(solver, "deterministic_mode", 1)
    end
    return if !isempty(opt.cudss_device_indices)
        cudss_device_count = length(opt.cudss_device_indices)
        CUDSS.cudss_set(solver, "device_count", cudss_device_count)
        CUDSS.cudss_set(solver, "device_indices", opt.cudss_device_indices)
    end
end

mutable struct CUDSSSolver{T, V} <: MadNLP.AbstractLinearSolver{T}
    inner::CUDSS.CudssSolver{T}
    tril::cuSPARSE.CuSparseMatrixCSC{T, Cint}
    x_gpu::CUDSS.CudssMatrix{T}
    b_gpu::CUDSS.CudssMatrix{T}
    buffer::V
    diag::V

    opt::CudssSolverOptions
    logger::MadNLP.MadNLPLogger
end

function CUDSSSolver(
        csc::cuSPARSE.CuSparseMatrixCSC{T, Cint};
        opt = CudssSolverOptions(),
        logger = MadNLP.MadNLPLogger(),
    ) where {T}
    n, m = size(csc)
    @assert n == m

    view = 'U'
    structure = 'G'
    # We need view = 'F' for the sparse LU decomposition
    (opt.cudss_algorithm == MadNLP.LU) && error(logger, "The sparse LU of cuDSS is not supported.")
    (opt.cudss_algorithm == MadNLP.CHOLESKY) && (structure = "SPD")
    (opt.cudss_algorithm == MadNLP.LDL) && (structure = "S")

    solver = CUDSS.CudssSolver(csc.colPtr, csc.rowVal, csc.nzVal, structure, view)
    set_cudss_options!(solver, opt)

    if opt.cudss_ordering != DEFAULT_ORDERING
        if opt.cudss_ordering == METIS_ORDERING
            A = SparseMatrixCSC(csc)
            A = A + A' - LinearAlgebra.Diagonal(A)
            G = Metis.graph(A, check_hermitian = false)
            opt.cudss_perm, _ = Metis.permutation(G)
        elseif opt.cudss_ordering == AMD_ORDERING
            A = SparseMatrixCSC(csc)
            opt.cudss_perm = AMD.amd(A)
        elseif opt.cudss_ordering == SYMAMD_ORDERING
            A = SparseMatrixCSC(csc)
            opt.cudss_perm = AMD.symamd(A)
        elseif opt.cudss_ordering == COLAMD_ORDERING
            A = SparseMatrixCSC(csc)
            opt.cudss_perm = AMD.colamd(A)
        elseif opt.cudss_ordering == USER_ORDERING
            (!isempty(opt.cudss_perm) && isperm(opt.cudss_perm)) || error(logger, "The vector opt.cudss_perm is not a valid permutation.")
        else
            error(logger, "The ordering $(opt.cudss_ordering) is not supported.")
        end
        CUDSS.cudss_set(solver, "user_perm", opt.cudss_perm)
    end

    # Check if we want to use the batch solver for matrices with a common sparsity pattern
    nbatch = solver.matrix.nbatch
    if nbatch > 1
        CUDSS.cudss_set(solver, "ubatch_size", nbatch)
    end

    # The phase "analysis" is "reordering" combined with "symbolic_factorization"
    x_gpu = CUDSS.CudssMatrix(T, n; nbatch = nbatch)
    b_gpu = CUDSS.CudssMatrix(T, n; nbatch = nbatch)
    CUDSS.cudss("analysis", solver, x_gpu, b_gpu, asynchronous = opt.cudss_asynchronous)

    # Allocate additional buffer for iterative refinement
    # Always allocate it to support dynamic updates to opt.cudss_ir
    buffer = CuVector{T}(undef, n * nbatch)

    # Scratch for the factor diagonal, used to recover the inertia when matching is on
    # (cuDSS misreports it as (0, 0)); only the nbatch == 1 path ever queries inertia.
    diag = CuVector{T}(undef, n)

    return CUDSSSolver(
        solver, csc,
        x_gpu, b_gpu, buffer, diag,
        opt, logger,
    )
end

function MadNLP.factorize!(M::CUDSSSolver)
    CUDSS.cudss_update(M.inner.matrix, nonzeros(M.tril))
    # A cuDSS error during (re)factorization is translated into a FactorizationException
    # (→ ERROR_IN_STEP_COMPUTATION) rather than crashing with a raw CUDSSError. This does
    # not interfere with the inertia→regularization recovery, which only runs when
    # factorize! returns normally and reports indefiniteness via `inertia` (info != 0
    # yields a dummy inertia).
    try
        if M.inner.fresh_factorization
            CUDSS.cudss("factorization", M.inner, M.x_gpu, M.b_gpu, asynchronous = M.opt.cudss_asynchronous)
        else
            CUDSS.cudss("refactorization", M.inner, M.x_gpu, M.b_gpu, asynchronous = M.opt.cudss_asynchronous)
        end
    catch e
        # Log the cuDSS status before erasing it: FactorizationException is field-less, so an OOM
        # (CUDSS_STATUS_ALLOC_FAILED), a not-initialized handle, or an internal error would
        # otherwise be indistinguishable from numerical breakdown once the IPM maps this to
        # ERROR_IN_STEP_COMPUTATION.
        if e isa CUDSS.CUDSSError
            @warn(M.logger, "cuDSS (re)factorization failed with status $(e.code); reporting a FactorizationException (-> ERROR_IN_STEP_COMPUTATION).")
            throw(FactorizationException())
        else
            rethrow(e)
        end
    end
    return M
end

function MadNLP.solve_linear_system!(M::CUDSSSolver{T, V}, xb::V) where {T, V}
    if M.opt.cudss_ir > 0
        copyto!(M.buffer, xb)
        CUDSS.cudss_update(M.b_gpu, M.buffer)
    else
        CUDSS.cudss_update(M.b_gpu, xb)
    end
    CUDSS.cudss_update(M.x_gpu, xb)
    # A cuDSS failure here — e.g. iterative refinement on a numerically singular system
    # reporting CUDSS_STATUS_EXECUTION_FAILED — is a step-computation failure, not a bug.
    # Translate it into a MadNLP SolveException so the IPM maps it to
    # ERROR_IN_STEP_COMPUTATION (-3) instead of crashing with a raw CUDSSError.
    try
        CUDSS.cudss("solve", M.inner, M.x_gpu, M.b_gpu, asynchronous = M.opt.cudss_asynchronous)
    catch e
        # As in factorize!: attach the cuDSS status before converting to the field-less
        # SolveException, so OOM vs numerical breakdown is diagnosable.
        if e isa CUDSS.CUDSSError
            @warn(M.logger, "cuDSS solve failed with status $(e.code); reporting a SolveException (-> ERROR_IN_STEP_COMPUTATION).")
            throw(SolveException())
        else
            rethrow(e)
        end
    end
    return xb
end

MadNLP.input_type(::Type{CUDSSSolver}) = :csc
MadNLP.default_options(::Type{CUDSSSolver}) = CudssSolverOptions()
MadNLP.is_inertia(M::CUDSSSolver) = (M.inner.matrix.nbatch == 1)  # Uncomment if MadNLP.LU is supported -- (M.opt.cudss_algorithm ∈ (MadNLP.CHOLESKY, MadNLP.LDL))
function MadNLP.inertia(M::CUDSSSolver)
    @assert M.inner.matrix.nbatch == 1
    n = size(M.tril, 1)
    info = CUDSS.cudss_get(M.inner, "info")

    # cudss_set(M.inner, "diag", buffer)  # specify the vector to update in `cudss_get`
    # cudss_get(M.inner, "diag")          # update the vector specified in `cudss_set`
    #
    # `buffer` contains the diagonal of the factorized matrix.

    return if M.opt.cudss_algorithm == MadNLP.CHOLESKY
        if info == 0
            return (n, 0, 0)
        else
            return (n - 2, 1, 1) # if factorization fails, return a dummy inertia
        end
    elseif M.opt.cudss_algorithm == MadNLP.LDL
        # N.B.: cuDSS does not always return the correct inertia.
        if info == 0
            if M.opt.cudss_matching
                # cuDSS (through 0.8) reports inertia (0, 0) whenever matching is enabled,
                # even though the factorization is correct — trusting it sends
                # InertiaBased/InertiaAuto into an endless regularization bump and then
                # restoration. Recover the inertia from the sign counts of the factor
                # diagonal instead. Caveat: exact only for 1×1 pivots, which holds for the
                # (quasi-definite) condensed KKT family this solver targets.
                CUDSS.cudss_set(M.inner, "diag", M.diag)
                CUDSS.cudss_get(M.inner, "diag")
                d = Array(M.diag)
                z = zero(eltype(d))
                npos = count(>(z), d)
                nneg = count(<(z), d)
                return (npos, n - npos - nneg, nneg)
            end
            (k, l) = CUDSS.cudss_get(M.inner, "inertia")
            @assert 0 ≤ k + l ≤ n
            return (k, n - k - l, l)
        else
            return (0, 1, n) # if factorization fails, return a dummy inertia
        end
    else
        error(M.logger, "Unsupported cudss_algorithm")
    end
end
MadNLP.improve!(M::CUDSSSolver) = false
MadNLP.is_supported(::Type{CUDSSSolver}, ::Type{Float32}) = true
MadNLP.is_supported(::Type{CUDSSSolver}, ::Type{Float64}) = true
MadNLP.introduce(M::CUDSSSolver) = "cuDSS v$(CUDSS.version())"
MadNLP.is_async(M::CUDSSSolver) = M.opt.cudss_asynchronous
