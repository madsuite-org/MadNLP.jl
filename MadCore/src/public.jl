# The public API of MadCore: the names that the packages built on MadCore may use.
# Other names are internal and can change in any release.

const PUBLIC_NAMES = (
    # KKT systems
    :_kktmul!, :_madnlp_unsafe_wrap,
    :AbstractCondensedKKTSystem, :AbstractDenseKKTSystem, :AbstractKKTSystem,
    :AbstractReducedKKTSystem, :AbstractUnreducedKKTSystem, :DenseCondensedKKTSystem,
    :DenseKKTSystem, :ScaledSparseKKTSystem, :ScenarioBlockMap,
    :SchurComplementKKTSystem, :SparseCondensedKKTSystem, :SparseKKTSystem,
    :SparseUnreducedKKTSystem, :block_structure, :build_condensed_aug_coord!,
    :build_condensed_aug_symbolic, :build_hessian_structure, :build_kkt!,
    :compress_hessian!, :compress_jacobian!, :create_kkt_system, :diag_add!,
    :factorize_kkt!, :get_hessian, :get_jacobian, :get_slack_regularization,
    :get_sparse_condensed_ext, :getptr, :is_inertia_correct, :jtprod!, :mul_hess_blk!,
    :nnz_jacobian, :num_variables, :nzval, :regularize_diagonal!,
    :should_regularize_dual,
    # KKT vectors
    :AbstractKKTVector, :PrimalVector, :UnreducedKKTVector, :dual, :dual_lb, :dual_ub,
    :full, :number_dual, :number_primal, :primal, :primal_dual, :slack, :variable,
    # Callbacks
    :_eval_cons_wrapper!, :_eval_f_wrapper, :_eval_grad_f_wrapper!, :_eval_jac_wrapper!,
    :_eval_lag_hess_wrapper!, :_hess_sparsity_wrapper!, :_jac_sparsity_wrapper!,
    :AbstractCallback, :AbstractEqualityTreatment, :AbstractFixedVariableTreatment,
    :DenseCallback, :EnforceEquality, :MakeParameter, :ModelBarrier, :NoFixedVariables,
    :RelaxBound, :RelaxEquality, :SparseCallback, :SparseWrapperModel, :create_array,
    :create_callback, :create_sparse_fixed_handler, :eval_cons_wrapper!,
    :eval_f_wrapper, :eval_grad_f_wrapper!, :eval_jac_wrapper!, :eval_lag_hess_wrapper!,
    :initialize!, :n_constraints, :n_variables, :set_con_scale_sparse!,
    :set_jac_scale_sparse!, :set_obj_scale!, :set_scaling!, :unpack_cons!, :unpack_x!,
    :unpack_y!, :unpack_z!, :update_z!,
    # Linear solvers
    :AbstractIterator, :AbstractLapackSolver, :AbstractLinearSolver, :CHOLMODSolver,
    :DummySolver, :FactorizationException, :InertiaException, :KrylovIterator,
    :LDLSolver, :LapackCPUSolver, :LapackOptions, :LinearSolverBarrier,
    :LinearSolverException, :Ma27Solver, :Ma57Solver, :Ma77Solver, :Ma86Solver,
    :Ma97Solver, :MumpsSolver, :PanuaPardisoSolver, :PardisoMKLSolver,
    :RichardsonIterator, :SolveException, :SymbolicException, :UmfpackSolver,
    :backend_package, :create_linear_solver, :default_options, :factorize!,
    :factorize_bunchkaufman!, :factorize_cholesky!, :factorize_evd!, :factorize_lu!,
    :factorize_qr!, :factorize_wrapper!, :improve!, :inertia, :input_type, :introduce,
    :is_async, :is_factorized, :is_inertia, :is_supported, :set_block_structure!,
    :setup!, :setup_bunchkaufman!, :setup_cholesky!, :setup_evd!, :setup_lu!,
    :setup_qr!, :solve_bunchkaufman!, :solve_cholesky!, :solve_evd!,
    :solve_linear_system!, :solve_lu!, :solve_qr!, :solve_refine!, :transfer_matrix!,
    # Quasi-Newton
    :BFGS, :CompactLBFGS, :DampedBFGS, :ExactHessian, :QuasiNewtonOptions,
    :create_quasi_newton, :update!,
    # Sparse matrix tools
    :SparseMatrixCOO, :coo_to_csc, :diag!, :findIJ, :force_lower_triangular!,
    :get_tril_to_full, :transfer!, :tril_to_full!,
    # Interior-point helpers
    :AbstractMadNLPSolver, :InvalidNumberException, :MadNLPExecutionStats,
    :NotEnoughDegreesOfFreedomException, :adjust_boundary!, :count_lu_bounds,
    :default_sparse_solver, :dual_inf_perturbation!, :finish_aug_solve!, :get_alpha_max,
    :get_average_complementarity, :get_inf_barrier, :get_inf_total,
    :get_min_complementarity, :get_sc, :get_sd, :get_tolerance, :has_constraints,
    :is_dense_callback, :is_valid, :parse_option, :print_ignored_options, :print_init,
    :print_iter, :print_summary, :reduce_rhs!, :reset_bound_dual!, :set_aug_diagonal!,
    :set_aug_rhs!, :set_initial_bounds!, :set_initial_rhs!, :set_options!, :solve_kkt!,
    :solve_refine_wrapper!,
    # Status and option values
    :QR,
    :AMD, :BUNCHKAUFMAN, :CHOLESKY, :DIVERGING_ITERATES, :ERROR,
    :ERROR_IN_STEP_COMPUTATION, :EVD, :INFEASIBLE_PROBLEM_DETECTED, :INFO, :INITIAL,
    :INTERNAL_ERROR, :INVALID_NUMBER_CONSTRAINTS, :INVALID_NUMBER_DETECTED,
    :INVALID_NUMBER_GRADIENT, :INVALID_NUMBER_HESSIAN_LAGRANGIAN,
    :INVALID_NUMBER_JACOBIAN, :INVALID_NUMBER_OBJECTIVE, :LDL, :LINESEARCH_SUCCEEDED,
    :LU, :MAXIMUM_ITERATIONS_EXCEEDED, :MAXIMUM_WALLTIME_EXCEEDED, :METIS,
    :NOT_ENOUGH_DEGREES_OF_FREEDOM, :REGULAR, :RESTORATION_FAILED, :RESTORE, :ROBUST,
    :SCALING_NONE, :SEARCH_DIRECTION_BECOMES_TOO_SMALL, :SOLVED_TO_ACCEPTABLE_LEVEL,
    :SOLVE_SUCCEEDED, :Status, :USER_REQUESTED_STOP,
    # Logging
    :var"@debug", :var"@error", :var"@info", :var"@notice", :var"@trace", :var"@warn",
    # Options, counters and timers
    :timing_callbacks, :timing_linear_solver, :timing_madnlp,
    :AbstractOptions, :AbstractTimer, :AbstractUserCallback, :HSLOrdering, :HSLScaling,
    :LinearFactorization, :LogLevels, :MC30, :MC64, :MC77, :MadNLPCounters,
    :MadNLPLogger, :NoUserCallback, :UserCallbackRegular, :UserCallbackRestore,
    :UserCallbackRobust, :blas_num_threads, :create_timer, :get_level,
    :get_status_output, :madsuite, :set_blas_num_threads, :timer_flush!, :timer_start!,
    :timer_stop!,
)

"""
    MadCore.public_names()

Return the names of the public API of MadCore, including the getter and setter of
each field in `SOLVER_FIELDS`. A package built on MadCore imports them to use the API
without qualification:

```julia
for name in MadCore.public_names()
    @eval import MadCore: \$name
end
```
"""
function public_names()
    accessors = Iterators.flatten((Symbol("get_", f), Symbol("set_", f, "!")) for f in SOLVER_FIELDS)
    return (PUBLIC_NAMES..., accessors..., :public_names)
end

# The keyword `public` does not parse on Julia 1.10.
@static if VERSION >= v"1.11"
    eval(Expr(:public, filter(name -> !Base.isexported(@__MODULE__, name), collect(public_names()))...))
end
