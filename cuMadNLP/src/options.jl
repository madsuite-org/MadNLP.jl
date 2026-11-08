# Option presets of MadNLPCore for models on NVIDIA GPUs

# The augmented KKT system is factorized by cuDSS after Ruiz scaling and a dual shift, with
# small pivots bounded; iterative refinement works with the unshifted system.
const GPU_DUAL_SHIFT = 1.0e-10
const GPU_PIVOT_EPSILON = 1.0e-10

function MadNLPCore.MadNLPOptions{T}(
        nlp::MadNLPCore.AbstractNLPModel{T, VT};
        dense_callback = MadNLPCore.is_dense_callback(nlp),
        callback = dense_callback ? MadNLPCore.DenseCallback : MadNLPCore.SparseCallback,
        kkt_system = dense_callback ? MadNLPCore.DenseCondensedKKTSystem : MadNLPCore.SparseKKTSystem,
        linear_solver = dense_callback ? LapackCUDASolver : CUDSSSolver,
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
    )
end

# cuDSS on the augmented KKT system of a GPU model: LDLᵀ with small pivots bounded
function MadCore.default_options(
        ::MadNLPCore.AbstractNLPModel{T, VT}, ::Type{MadNLPCore.SparseKKTSystem}, ::Type{CUDSSSolver},
    ) where {T, VT <: CuVector{T}}
    opt = MadCore.default_options(CUDSSSolver)
    opt.cudss_algorithm = MadCore.LDL
    opt.cudss_pivot_epsilon = GPU_PIVOT_EPSILON
    return opt
end
