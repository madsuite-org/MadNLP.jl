# Option presets of MadNLP for models on the GPU

function MadNLP.MadNLPOptions{T}(
        nlp::MadNLP.AbstractNLPModel{T, VT};
        dense_callback = MadNLP.is_dense_callback(nlp),
        callback = dense_callback ? MadNLP.DenseCallback : MadNLP.SparseCallback,
        kkt_system = dense_callback ? MadNLP.DenseCondensedKKTSystem : MadNLP.SparseCondensedKKTSystem,
        linear_solver = dense_callback ? LapackCUDASolver : CUDSSSolver,
        tol = MadNLP.get_tolerance(T, kkt_system),
        bound_relax_factor = (kkt_system == MadNLP.SparseCondensedKKTSystem) ? tol : T(1.0e-8),
    ) where {T, VT <: CuVector{T}}
    return MadNLP.MadNLPOptions{T}(
        tol = tol,
        callback = callback,
        kkt_system = kkt_system,
        linear_solver = linear_solver,
        bound_relax_factor = bound_relax_factor,
    )
end
