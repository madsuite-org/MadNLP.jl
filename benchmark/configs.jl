# The solver configurations. `configs(device)` returns `name => options`, the keyword
# arguments that select the configuration in `madnlp(model; options...)`. Add an entry here
# to benchmark another KKT system, linear solver or option set. Included by run.jl after the
# solver packages are loaded.

# The option set of the condensed KKT system on the host; on a CUDA device cuMadNLP selects it.
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
# The evaluation counts Ipopt prints at the end of its log; the counters of an ExaModel do not
# record them. `nothing` without a log.
function ipopt_counts(path)
    (isempty(path) || !isfile(path)) && return nothing
    text = read(path, String)
    count_of(what) = (m = match(Regex("Number of $what evaluations\\s*=\\s*(\\d+)"), text); m === nothing ? 0 : parse(Int, m[1]))
    return (;
        obj_cnt = count_of("objective function"), obj_grad_cnt = count_of("objective gradient"),
        # Ipopt counts one evaluation of the constraints under both of its kinds
        con_cnt = max(count_of("equality constraint"), count_of("inequality constraint")),
        con_jac_cnt = max(count_of("equality constraint Jacobian"), count_of("inequality constraint Jacobian")),
        lag_hess_cnt = count_of("Lagrangian Hessian"),
    )
end

function run_ipopt(m; tol, acceptable_tol, max_wall_time, output_file = "", _...)
    NLPModels.reset!(m)
    solver = HSL.LIBHSL_isfunctional() ?
        (; linear_solver = "ma57", hsllib = HSL.HSL_jll.libhsl_path, ma57_automatic_scaling = "yes") : (;)
    log = isempty(output_file) ? (;) : (; output_file, file_print_level = 5)
    s = NLPModelsIpopt.ipopt(m; tol, acceptable_tol, max_wall_time, print_level = 0, solver..., log...)
    logged = ipopt_counts(output_file)
    counters = (;
        factorization_cnt = 0, backsolve_cnt = 0, ir = 0, obj_cnt = NLPModels.neval_obj(m), obj_grad_cnt = NLPModels.neval_grad(m),
        con_cnt = NLPModels.neval_cons(m), con_jac_cnt = NLPModels.neval_jac(m), lag_hess_cnt = NLPModels.neval_hess(m),
        total_time = s.elapsed_time, linear_solver_time = NaN, eval_function_time = NaN, init_time = NaN,
        something(logged, (;))...,
    )
    return (;
        status = get(IPOPT_STATUS, s.status, MadNLP.ERROR_IN_STEP_COMPUTATION), iter = s.iter, objective = s.objective,
        primal_feas = s.primal_feas, dual_feas = s.dual_feas, solution = s.solution, counters,
    )
end

function configs(device::Symbol)
    c = Pair{String, NamedTuple}[]
    # The augmented system with MA57 on the host where a licensed HSL_jll is installed, else with
    # MUMPS, and the condensed system with cuDSS on a CUDA device
    ma57 = (; linear_solver = MadNLP.Ma57Solver, ma57_automatic_scaling = true)
    push!(c, "default" => (device == :cpu && HSL.LIBHSL_isfunctional() ? ma57 : (;)))
    if device == :cpu
        push!(c, "augmented_mumps" => (; linear_solver = MadNLP.MumpsSolver))
        push!(c, "ipopt" => (; solver = run_ipopt))
        push!(c, "condensed_cholmod" => (; CONDENSED..., linear_solver = MadNLP.CHOLMODSolver))
        if HSL.LIBHSL_isfunctional()   # the registry's HSL_jll carries no solver
            push!(c, "augmented_ma27" => (; linear_solver = MadNLP.Ma27Solver))
            push!(c, "augmented_ma57" => ma57)
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
