# Option presets for models whose data live on a GPU. The GPU entry points choose the linear
# solvers through `default_sparse_solver` and `default_dense_solver`. The barrier parameter
# follows the quality-function rule, which may raise it at most twofold per iteration, and the
# slacks are moved onto the constraints after an accepted step.
const GPU_SIGMA_MAX = 2.0
const GPU_MU_MIN = 1.0e-11

function MadNLPCore.MadNLPOptions{T}(
        nlp::MadNLPCore.AbstractNLPModel{T, VT};
        dense_callback = MadNLPCore.is_dense_callback(nlp),
        callback = dense_callback ? MadNLPCore.DenseCallback : MadNLPCore.SparseCallback,
        kkt_system = dense_callback ? MadNLPCore.DenseCondensedKKTSystem : MadNLPCore.SparseCondensedKKTSystem,
        linear_solver = dense_callback ? MadNLPCore.default_dense_solver(nlp) : MadNLPCore.default_sparse_solver(nlp),
        tol = MadNLPCore.get_tolerance(T, kkt_system),
        bound_relax_factor = (kkt_system == MadNLPCore.SparseCondensedKKTSystem) ? tol : T(1.0e-8),
    ) where {T, VT <: AbstractGPUVector{T}}
    return MadNLPCore.MadNLPOptions{T}(
        tol = tol,
        callback = callback,
        kkt_system = kkt_system,
        linear_solver = linear_solver,
        bound_relax_factor = bound_relax_factor,
        barrier = MadNLPCore.QualityFunctionUpdate{T}(sigma_max = T(GPU_SIGMA_MAX), mu_min = T(GPU_MU_MIN)),
        slack_reset = true,
    )
end
