# The `madnlp` executable: an AMPL solver for .nl files, on the CPU.
#
#   madnlp model[.nl] [-AMPL] [key=value ...]
#
# AMPL runs a solver as `madnlp stub -AMPL` and passes options in the environment variable
# `madnlp_options`; options on the command line are added to those. With -AMPL the solution
# is written to stub.sol for AMPL to read.

using AmplNLReader: AmplNLReader, AmplModel, write_sol

const USAGE = """usage: madnlp model[.nl] [-AMPL] [key=value ...]
Solves an AMPL .nl file with MadNLP (CPU). With -AMPL, writes model.sol.
Options (also read from the environment variable madnlp_options), e.g.
  tol=1e-8 max_iter=500 print_level=INFO linear_solver=LapackCPUSolver"""

function main(args::Vector{String})::Cint
    trace = CNLPLib.tracing(APP.trace)
    ct0 = trace ? CNLPLib.compile_ns() : UInt64(0)
    rc = _main(args)
    trace ? CNLPLib.report(APP.trace, "madnlp", CNLPLib.compile_ns() - ct0) : (flush(stdout); flush(stderr))
    return rc
end

function _main(args::Vector{String})::Cint
    (isempty(args) || args[1] in ("-h", "--help", "-?")) && (println(USAGE); return isempty(args) ? 1 : 0)
    args[1] == "-v" && (println("MadNLP ", VERSION_STRING, " (MadNLPApp)"); return 0)
    stub = args[1]
    ampl = false
    opts = String[]
    env = get(ENV, "madnlp_options", "")
    isempty(env) || append!(opts, split(env))
    for a in args[2:end]
        if a == "-AMPL"
            ampl = true
        elseif occursin('=', a)
            push!(opts, a)
        else
            println(stderr, "madnlp: unknown argument ", a, "\n", USAGE)
            return 2
        end
    end
    path = endswith(stub, ".nl") ? stub : stub * ".nl"
    isfile(path) || (println(stderr, "madnlp: no file ", path); return 2)
    kw = options([String(k) => String(v) for (k, v) in (split(o, '='; limit = 2) for o in opts)])
    nlp = AmplModel(path)
    try
        # AmplModel carries AmplNLReader's own meta type; the wrapper gives MadNLP a
        # standard NLPModelMeta.
        stats = MadNLPAppCore.run_madnlp(MadCore.SparseWrapperModel(Vector, nlp), kw)
        if ampl
            # AMPL's duals are the objective's sensitivities to the constraints' right-hand
            # sides, -y for NLPModels' y (grad f + J'y - zL + zU = 0), in either sense.
            write_sol(
                nlp, "MadNLP: " * string(stats.status), Vector{Float64}(stats.solution),
                -Vector{Float64}(stats.multipliers)
            )
        end
        return MadNLPAppCore.status_of(stats.status) <= CNLPLib.CNLP_ACCEPTABLE ? 0 : 1
    finally
        finalize(nlp)
    end
end

"The `madnlp` executable's entry: bin/madnlp is a C launcher that calls this in libmadnlp."
Base.@ccallable function madnlp_main(argc::Cint, argv::Ptr{Ptr{UInt8}})::Cint
    args = String[unsafe_string(unsafe_load(argv, i)) for i in 1:argc]
    return try
        main(args)
    catch e
        println(stderr, "madnlp: ", sprint(showerror, e))
        Cint(2)
    end
end
