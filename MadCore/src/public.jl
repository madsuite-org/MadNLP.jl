# The public API of MadCore: what a solver or a linear solver built on MadCore needs. The
# packages of this repository are pinned to the same version and may use internal names.

const PUBLIC_NAMES = (
    # KKT systems
    :AbstractCondensedKKTSystem, :AbstractKKTSystem, :AbstractReducedKKTSystem,
    :AbstractUnreducedKKTSystem, :ScaledSparseKKTSystem, :SparseCondensedKKTSystem,
    :SparseKKTSystem, :SparseUnreducedKKTSystem, :block_structure,
    :build_condensed_aug_coord!, :build_condensed_aug_symbolic,
    :build_hessian_structure, :build_kkt!, :compress_hessian!, :compress_jacobian!,
    :create_kkt_system, :factorize_kkt!, :get_hessian, :get_jacobian,
    :get_sparse_condensed_ext, :is_inertia_correct, :jtprod!, :kktmul!,
    :madnlp_unsafe_wrap, :nnz_jacobian, :num_variables, :regularize_diagonal!,
    :should_regularize_dual,
    # KKT vectors
    :AbstractKKTVector, :PrimalVector, :UnreducedKKTVector, :dual, :dual_lb, :dual_ub,
    :full, :primal, :primal_dual, :slack, :variable,
    # Callbacks
    :AbstractCallback, :AbstractEqualityTreatment, :AbstractFixedVariableTreatment,
    :EnforceEquality, :MakeParameter, :NoFixedVariables, :RelaxBound, :RelaxEquality,
    :SparseCallback, :create_array, :create_callback, :create_sparse_fixed_handler,
    :eval_cons_callback!, :eval_cons_wrapper!, :eval_f_callback, :eval_f_wrapper,
    :eval_grad_f_callback!, :eval_grad_f_wrapper!, :eval_jac_callback!,
    :eval_jac_wrapper!, :eval_lag_hess_callback!, :eval_lag_hess_wrapper!,
    :hess_sparsity_callback!, :initialize!, :jac_sparsity_callback!, :n_constraints,
    :n_variables, :set_con_scale_sparse!, :set_jac_scale_sparse!, :set_obj_scale!,
    :set_scaling!, :unpack_cons!, :unpack_x!, :unpack_y!, :unpack_z!, :update_z!,
    # Linear solvers
    :AbstractLinearSolver, :LDLSolver, :LinearSolverException, :MumpsSolver,
    :SolveException, :backend_package, :create_linear_solver, :default_options,
    :factorize!, :factorize_wrapper!, :improve!, :inertia, :input_type, :introduce,
    :is_factorized, :is_inertia, :is_supported, :set_block_structure!,
    :solve_linear_system!,
    # Quasi-Newton
    :ExactHessian, :QuasiNewtonOptions, :create_quasi_newton, :update!,
    # Sparse matrix tools
    :SparseMatrixCOO, :coo_to_csc, :force_lower_triangular!, :transfer!,
    # Interior-point helpers
    :AbstractMadNLPSolver,
    :InvalidNumberException, :MadNLPExecutionStats, :NotEnoughDegreesOfFreedomException,
    :SubVector, :adjust_boundary!, :default_sparse_solver, :finish_aug_solve!,
    :get_alpha_max, :get_alpha_z, :get_inf_compl, :get_inf_du, :get_inf_pr, :get_kkt,
    :print_ignored_options, :print_iter, :print_summary, :reduce_rhs!,
    :set_aug_diagonal!, :set_initial_bounds!, :set_options!, :solve_kkt!,
    :solve_refine_wrapper!, :transfer_values!, :update_con_scale_sparse!, :update_x!,
    # Status and option values
    :DIVERGING_ITERATES, :ERROR, :ERROR_IN_STEP_COMPUTATION,
    :INFEASIBLE_PROBLEM_DETECTED, :INFO, :INITIAL, :INTERNAL_ERROR,
    :INVALID_NUMBER_CONSTRAINTS, :INVALID_NUMBER_DETECTED, :INVALID_NUMBER_GRADIENT,
    :INVALID_NUMBER_HESSIAN_LAGRANGIAN, :INVALID_NUMBER_JACOBIAN,
    :INVALID_NUMBER_OBJECTIVE, :MAXIMUM_ITERATIONS_EXCEEDED, :MAXIMUM_WALLTIME_EXCEEDED,
    :NOT_ENOUGH_DEGREES_OF_FREEDOM, :REGULAR, :RESTORATION_FAILED,
    :SEARCH_DIRECTION_BECOMES_TOO_SMALL, :SOLVED_TO_ACCEPTABLE_LEVEL, :SOLVE_SUCCEEDED,
    :Status, :USER_REQUESTED_STOP,
    # Logging
    :var"@debug", :var"@info", :var"@notice", :var"@trace",
    # Options, counters and timers
    :AbstractOptions, :LogLevels, :MadNLPCounters, :MadNLPLogger, :get_level,
    :get_status_output, :madsuite,
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
