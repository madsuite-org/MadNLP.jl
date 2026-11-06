# Option presets of MadNLPCore for models on AMD GPUs

function MadNLPCore.MadNLPOptions{T}(
    nlp::MadNLPCore.AbstractNLPModel{T,VT};
    dense_callback = MadNLPCore.is_dense_callback(nlp),
    callback = dense_callback ? MadNLPCore.DenseCallback : MadNLPCore.SparseCallback,
    kkt_system = dense_callback ? MadNLPCore.DenseCondensedKKTSystem : MadNLPCore.SparseCondensedKKTSystem,
    linear_solver = LapackROCmSolver,
    tol = MadNLPCore.get_tolerance(T,kkt_system),
    bound_relax_factor = tol,
) where {T, VT <: ROCVector{T}}
    return MadNLPCore.MadNLPOptions{T}(
        tol = tol,
        callback = callback,
        kkt_system = kkt_system,
        linear_solver = linear_solver,
        bound_relax_factor = bound_relax_factor,
    )
end
