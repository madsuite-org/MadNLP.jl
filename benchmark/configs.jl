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

function configs(device::Symbol)
    c = Pair{String, NamedTuple}[]
    # MadNLP's own choice for the model: the augmented system with MUMPS on the host, the
    # condensed system with cuDSS on a CUDA device
    push!(c, "default" => (;))
    if device == :cpu
        push!(c, "condensed_cholmod" => (; CONDENSED..., linear_solver = MadNLP.CHOLMODSolver))
        if MadNLPHSL.HSL.LIBHSL_isfunctional()   # the registry's HSL_jll carries no solver
            push!(c, "augmented_ma27" => (; linear_solver = MadNLPHSL.Ma27Solver))
            push!(c, "augmented_ma57" => (; linear_solver = MadNLPHSL.Ma57Solver))
        end
    elseif device == :cuda
        # Bunch-Kaufman pivoting needs cuDSS's matching, which reads the matrix values: the
        # analysis runs at the first factorization, the factorization re-pivots at every
        # iteration, and a solve rejected by iterative refinement redoes the analysis
        push!(
            c, "augmented_cudss" => (;
                kkt_system = MadNLP.SparseKKTSystem, linear_solver = MadNLPGPU.CUDSSSolver,
                cudss_pivoting = true, cudss_pivot_type = 'B', cudss_matching = true, cudss_matching_alg = "algo6",
                cudss_deferred_analysis = true, cudss_refactorization = false, cudss_reanalyze_on_improve = true,
            ),
        )
    end
    return c
end
