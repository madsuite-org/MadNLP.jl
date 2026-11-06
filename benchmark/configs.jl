# The solver configurations. `configs(device)` returns `name => options`, the keyword
# arguments that select the configuration in `madnlp(model; options...)`. Add an entry here
# to benchmark another KKT system, linear solver or option set. Included by run.jl after the
# solver packages are loaded.

# The option set of the condensed KKT system on the host; on a CUDA device MadNLPGPU selects it.
const CONDENSED = (;
    kkt_system = MadNLP.SparseCondensedKKTSystem,
    equality_treatment = MadNLP.RelaxEquality,
    fixed_variable_treatment = MadNLP.RelaxBound,
    dual_initialization_method = MadNLP.DualInitializeSetZero,
)

# Ipopt as a baseline, with its default MUMPS. Returns the fields of MadNLP's statistics that a
# result row reads, so that it goes through the same loop. Ipopt reports no time breakdown,
# factorization count or refinement count.
const IPOPT_STATUS = Dict(
    :first_order => MadNLP.SOLVE_SUCCEEDED, :acceptable => MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL,
    :small_step => MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL, :infeasible => MadNLP.INFEASIBLE_PROBLEM_DETECTED,
    :max_iter => MadNLP.MAXIMUM_ITERATIONS_EXCEEDED, :max_time => MadNLP.MAXIMUM_WALLTIME_EXCEEDED,
)
function run_ipopt(m; tol, max_wall_time, _...)
    NLPModels.reset!(m)
    s = NLPModelsIpopt.ipopt(m; tol, max_wall_time, print_level = 0)
    counters = (;
        factorization_cnt = 0, backsolve_cnt = 0, ir = 0, obj_cnt = NLPModels.neval_obj(m), obj_grad_cnt = NLPModels.neval_grad(m),
        con_cnt = NLPModels.neval_cons(m), con_jac_cnt = NLPModels.neval_jac(m), lag_hess_cnt = NLPModels.neval_hess(m),
        total_time = s.elapsed_time, linear_solver_time = NaN, eval_function_time = NaN, init_time = NaN,
    )
    return (;
        status = get(IPOPT_STATUS, s.status, MadNLP.ERROR_IN_STEP_COMPUTATION), iter = s.iter, objective = s.objective,
        primal_feas = s.primal_feas, dual_feas = s.dual_feas, solution = s.solution, counters,
    )
end

function configs(device::Symbol)
    c = Pair{String, NamedTuple}[]
    # MadNLP's own choice for the model: the augmented system with MUMPS on the host, the
    # condensed system with cuDSS on a CUDA device
    push!(c, "default" => (;))
    if device == :cpu
        push!(c, "ipopt" => (; solver = run_ipopt))
        push!(c, "condensed_cholmod" => (; CONDENSED..., linear_solver = MadNLP.CHOLMODSolver))
        if HSL.LIBHSL_isfunctional()   # the registry's HSL_jll carries no solver
            push!(c, "augmented_ma27" => (; linear_solver = MadNLP.Ma27Solver))
            push!(c, "augmented_ma57" => (; linear_solver = MadNLP.Ma57Solver))
        end
    elseif device == :cuda
        # Bunch-Kaufman pivoting needs cuDSS's matching, which reads the matrix values: the
        # analysis runs at the first factorization, the factorization re-pivots at every
        # iteration, and a solve rejected by iterative refinement redoes the analysis
        push!(
            c, "augmented_cudss" => (;
                kkt_system = MadNLP.SparseKKTSystem, linear_solver = cuMadNLP.CUDSSSolver,
                cudss_pivoting = true, cudss_pivot_type = 'B', cudss_matching = true, cudss_matching_alg = "algo6",
                cudss_deferred_analysis = true, cudss_refactorization = false, cudss_reanalyze_on_improve = true,
            ),
        )
    end
    return c
end

# MadNLP ignores an option it does not know, so a mistyped option would measure the default
# under the name of the configuration.
function validate(configs)
    for (name, opts) in configs
        haskey(opts, :solver) && continue   # options of another solver
        known = Set(fieldnames(MadNLP.MadNLPOptions))
        haskey(opts, :linear_solver) && union!(known, fieldnames(typeof(MadNLP.default_options(opts.linear_solver))))
        unknown = setdiff(keys(opts), known)
        isempty(unknown) || error("configuration $name: unknown options $(join(unknown, ", "))")
    end
    return
end
