# Option presets for models on NVIDIA GPUs: the linear solvers of the GPU option presets
# (MadNLPCoreGPUExt), and a preset of their own, more specific than the shared one.

MadCore.default_sparse_solver(::MadCore.AbstractNLPModel{T, <:CuVector{T}}) where {T} = CUDSSSolver
MadCore.default_dense_solver(::MadCore.AbstractNLPModel{T, <:CuVector{T}}) where {T} = LapackCUDASolver

# The augmented KKT system is factorized by cuDSS after Ruiz scaling and a dual shift, with
# matching and small pivots bounded; GMRES refines each solve on the unshifted system.
const GPU_DUAL_SHIFT = 1.0e-10
const GPU_PIVOT_EPSILON = 1.0e-10

function MadNLPCore.MadNLPOptions{T}(
        nlp::MadNLPCore.AbstractNLPModel{T, VT};
        dense_callback = MadNLPCore.is_dense_callback(nlp),
        callback = dense_callback ? MadNLPCore.DenseCallback : MadNLPCore.SparseCallback,
        kkt_system = dense_callback ? MadNLPCore.DenseCondensedKKTSystem : MadNLPCore.SparseKKTSystem,
        linear_solver = dense_callback ? MadNLPCore.default_dense_solver(nlp) : MadNLPCore.default_sparse_solver(nlp),
        tol = MadNLPCore.get_tolerance(T, kkt_system),
        bound_relax_factor = (kkt_system == MadNLPCore.SparseCondensedKKTSystem) ? tol : T(1.0e-8),
    ) where {T, VT <: CuVector{T}}
    augmented = kkt_system == MadNLPCore.SparseKKTSystem
    return MadNLPCore.MadNLPOptions{T}(
        tol = tol,
        callback = callback,
        kkt_system = kkt_system,
        linear_solver = linear_solver,
        bound_relax_factor = bound_relax_factor,
        kkt_scaler = augmented ? MadCore.RuizScaler() : MadCore.NoScaler(),
        dual_shift = augmented ? T(GPU_DUAL_SHIFT) : zero(T),
        iterator = augmented ? MadCore.KrylovIterator : MadCore.RichardsonIterator,
    )
end

# cuDSS on the augmented KKT system of a GPU model: LDLᵀ with small pivots bounded and with
# matching, without which the solve of this indefinite matrix has a backward error near 1e-5.
# Matching depends on the values, so the analysis waits for the first matrix.
function MadCore.default_options(
        ::MadNLPCore.AbstractNLPModel{T, VT}, ::Type{MadNLPCore.SparseKKTSystem}, ::Type{CUDSSSolver},
    ) where {T, VT <: CuVector{T}}
    opt = MadCore.default_options(CUDSSSolver)
    opt.cudss_algorithm = MadCore.LDL
    opt.cudss_pivot_epsilon = GPU_PIVOT_EPSILON
    opt.cudss_matching = true
    opt.cudss_deferred_analysis = true
    return opt
end
