# Option presets of MadNLPCore for models on Apple GPUs

function MadNLPCore.MadNLPOptions{T}(
        nlp::MadNLPCore.AbstractNLPModel{T, VT};
        callback = MadNLPCore.SparseCallback,
        kkt_system = MadNLPCore.SparseCondensedKKTSystem,
        linear_solver = SDSSolver,
        tol = MadNLPCore.get_tolerance(T, kkt_system),
        bound_relax_factor = tol,
    ) where {T, VT <: MtlVector{T}}
    return MadNLPCore.MadNLPOptions{T}(
        tol = tol,
        callback = callback,
        kkt_system = kkt_system,
        linear_solver = linear_solver,
        bound_relax_factor = bound_relax_factor,
    )
end
