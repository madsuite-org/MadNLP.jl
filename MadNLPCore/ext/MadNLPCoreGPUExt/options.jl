# Option presets for models whose data live on a GPU. The GPU entry points choose the linear
# solvers through `default_sparse_solver` and `default_dense_solver`. With the augmented KKT
# system, the barrier parameter follows the quality-function rule, which may raise it at most
# twofold per iteration, and the slacks are moved onto the constraints after an accepted step.
# The other systems keep the monotone rule: the quality-function rule solves with the KKT system
# before the first factorization, which the condensed system has not done yet with its zero
# initial multipliers.
const GPU_SIGMA_MAX = 2.0
# the floor of the barrier parameter, relative to the tolerance: 1e-11 at the default tol = 1e-8
const GPU_MU_MIN_RATIO = 1.0e-3

function MadNLPCore.MadNLPOptions{T}(
        nlp::MadNLPCore.AbstractNLPModel{T, VT};
        dense_callback = MadNLPCore.is_dense_callback(nlp),
        callback = dense_callback ? MadNLPCore.DenseCallback : MadNLPCore.SparseCallback,
        kkt_system = dense_callback ? MadNLPCore.DenseCondensedKKTSystem : MadNLPCore.SparseCondensedKKTSystem,
        linear_solver = dense_callback ? MadNLPCore.default_dense_solver(nlp) : MadNLPCore.default_sparse_solver(nlp),
        tol = MadNLPCore.get_tolerance(T, kkt_system),
        bound_relax_factor = (kkt_system == MadNLPCore.SparseCondensedKKTSystem) ? tol : T(1.0e-8),
    ) where {T, VT <: AbstractGPUVector{T}}
    adaptive = kkt_system == MadNLPCore.SparseKKTSystem ? (;
        barrier = MadNLPCore.QualityFunctionUpdate{T}(sigma_max = T(GPU_SIGMA_MAX), mu_min = T(GPU_MU_MIN_RATIO) * tol),
        slack_reset = true,
    ) : (;)
    return MadNLPCore.MadNLPOptions{T}(;
        tol = tol,
        callback = callback,
        kkt_system = kkt_system,
        linear_solver = linear_solver,
        bound_relax_factor = bound_relax_factor,
        adaptive...,
    )
end
