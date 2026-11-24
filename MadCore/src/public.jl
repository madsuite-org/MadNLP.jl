# The public API of MadCore: what a solver or a linear solver built on MadCore needs. The
# packages of this repository are pinned to the same version and may use internal names.

const PUBLIC_NAMES = (
    # KKT systems
    :AbstractKKTSystem, :block_structure, :build_condensed_aug_coord!,
    :build_condensed_aug_symbolic, :build_hessian_structure, :build_kkt!,
    :compress_hessian!, :compress_jacobian!, :create_kkt_system, :factorize_kkt!,
    :get_hessian, :get_jacobian, :get_sparse_condensed_ext, :is_inertia_correct, :jtprod!,
    :kktmul!, :madnlp_unsafe_wrap, :num_variables, :regularize_diagonal!,
    :ScaledSparseKKTSystem, :SparseCondensedKKTSystem, :SparseKKTSystem,
    :SparseUnreducedKKTSystem,
    # KKT vectors
    :AbstractKKTVector, :dual, :dual_lb, :dual_ub, :full, :primal, :primal_dual,
    :PrimalVector, :slack, :UnreducedKKTVector, :variable,
    # Callbacks
    :AbstractCallback, :AbstractEqualityTreatment, :AbstractFixedVariableTreatment,
    :create_array, :create_callback, :create_sparse_fixed_handler, :EnforceEquality,
    :eval_cons_callback!, :eval_cons_wrapper!, :eval_f_callback, :eval_f_wrapper,
    :eval_grad_f_callback!, :eval_grad_f_wrapper!, :eval_jac_callback!, :eval_jac_wrapper!,
    :eval_lag_hess_callback!, :eval_lag_hess_wrapper!, :hess_sparsity_callback!,
    :initialize!, :jac_sparsity_callback!, :MakeParameter, :n_constraints, :n_variables,
    :NoFixedVariables, :parse_indexes, :RelaxBound, :RelaxEquality, :set_con_scale_sparse!,
    :set_jac_scale_sparse!, :set_obj_scale!, :set_scaling!, :SparseCallback,
    :treat_equality_initialize!, :treat_fixed_variable_initialize!, :unpack_cons!,
    :unpack_x!, :unpack_y!, :unpack_z!, :update_z!,
    # Linear solvers
    :AbstractLinearSolver, :default_options, :factorize!, :factorize_wrapper!, :inertia,
    :introduce, :is_supported, :LDLSolver, :LinearSolverException, :MumpsSolver,
    :set_block_structure!, :solve_linear_system!, :SolveException,
    # Quasi-Newton
    :create_quasi_newton, :ExactHessian, :QuasiNewtonOptions, :update!,
    # Sparse matrix tools
    :coo_to_csc, :force_lower_triangular!, :SparseMatrixCOO, :transfer!,
    # Interior-point helpers
    :AbstractMadNLPSolver, :adjust_boundary!, :default_sparse_solver, :define_solver_accessors,
    :dual_inf_perturbation!, :finish_aug_solve!, :get_alpha_max, :get_alpha_z,
    :get_average_complementarity, :get_inf_compl, :get_inf_du, :get_inf_pr,
    :get_min_complementarity, :get_sc, :get_sd, :initialize_variable,
    :InvalidNumberException, :MadNLPExecutionStats, :NotEnoughDegreesOfFreedomException,
    :print_ignored_options, :print_init, :print_iter, :print_summary, :reduce_rhs!,
    :reset_bound_dual!, :set_aug_diagonal!, :set_aug_rhs!, :set_initial_bounds!,
    :set_options!, :solve_kkt!, :solve_refine_wrapper!, :SparseWrapperModel, :SubVector,
    :transfer_values!, :unpack_obj, :update_con_scale_sparse!, :update_x!,
    :UserCallbackRegular,
    # Status and option values
    :CHOLESKY, :CompactLBFGS, :DIVERGING_ITERATES, :ERROR, :ERROR_IN_STEP_COMPUTATION,
    :INFEASIBLE_PROBLEM_DETECTED, :INFO, :INITIAL, :INTERNAL_ERROR,
    :INVALID_NUMBER_CONSTRAINTS, :INVALID_NUMBER_DETECTED, :INVALID_NUMBER_GRADIENT,
    :INVALID_NUMBER_HESSIAN_LAGRANGIAN, :INVALID_NUMBER_JACOBIAN, :INVALID_NUMBER_OBJECTIVE,
    :LDL, :LINESEARCH_SUCCEEDED, :MAXIMUM_ITERATIONS_EXCEEDED, :MAXIMUM_WALLTIME_EXCEEDED,
    :NOT_ENOUGH_DEGREES_OF_FREEDOM, :REGULAR, :RESTORATION_FAILED, :RESTORE, :ROBUST,
    :SEARCH_DIRECTION_BECOMES_TOO_SMALL, :SOLVE_SUCCEEDED, :SOLVED_TO_ACCEPTABLE_LEVEL,
    :Status, :UNBOUNDED_PROBLEM_DETECTED, :USER_REQUESTED_STOP,
    # Logging
    :var"@debug", :var"@info", :var"@notice", :var"@trace", :var"@warn",
    # Options, counters and timers
    :AbstractOptions, :format_time, :get_level, :get_mu,
    :get_primary_options, :get_status_output, :get_tau, :LogLevels, :MadNLPCounters,
    :MadNLPLogger, :madsuite,
)

"""
    MadCore.public_names()

Return the names of the public API of MadCore. A package built on MadCore imports them
to use the API without qualification:

```julia
for name in MadCore.public_names()
    @eval import MadCore: \$name
end
```
"""
public_names() = (PUBLIC_NAMES..., :public_names)

# The keyword `public` does not parse on Julia 1.10.
@static if VERSION >= v"1.11"
    eval(Expr(:public, filter(name -> !Base.isexported(@__MODULE__, name), collect(public_names()))...))
end
