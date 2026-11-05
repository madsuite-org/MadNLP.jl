"""
    MadNLPBenchmark

The benchmarking loop behind `benchmark/run.jl`: it solves every instance of a problem set
with every requested configuration and writes one CSV file per set and configuration, with a
metadata file beside it. `benchmark/README.md` states the protocol.
"""
module MadNLPBenchmark

using MadNLP, NLPModels
using Dates, DelimitedFiles, LinearAlgebra, Printf, TOML

export run_benchmark, read_results, solved, score_time, sgm, size_class

"The columns of a result file, in order."
const HEADER = [
    "case", "status", "iters", "nvar", "ncon", "nnzj", "nnzh",
    "obj", "primal_feas", "dual_feas",
    "n_fact", "n_backsolve", "n_obj", "n_grad", "n_con", "n_jac", "n_hess",
    "t_total", "t_linsolve", "t_eval", "t_init",
    # violation of the model's own constraints and bounds at the returned point, evaluated
    # independently of the solver: constraint rows beyond [lcon, ucon], variables beyond
    # [lvar, uvar]. The solver's primal_feas is the residual of its scaled, relaxed problem.
    "viol_con", "viol_var",
]

# Status and time recorded for an instance that could not be built or whose solve threw.
const ERROR_STATUS = -100

sanitize(s) = replace(strip(String(s)), r"[^A-Za-z0-9.+-]+" => "_")

"A directory name for the machine: the CPU model, or the name of the device if one is given."
hardware_tag(device_name = nothing) =
    sanitize(device_name === nothing ? Sys.cpu_info()[1].model : device_name)

dims(m) = (NLPModels.get_nvar(m), NLPModels.get_ncon(m), NLPModels.get_nnzj(m), NLPModels.get_nnzh(m))

function violation(m, stats)
    x = Array(stats.solution)
    x = x[begin:(begin + NLPModels.get_nvar(m) - 1)]   # the solver appends its slacks
    c = Array(NLPModels.cons(m, convert(typeof(NLPModels.get_x0(m)), x)))
    lcon, ucon = Array(NLPModels.get_lcon(m)), Array(NLPModels.get_ucon(m))
    lvar, uvar = Array(NLPModels.get_lvar(m)), Array(NLPModels.get_uvar(m))
    viol_con = maximum(max.(lcon .- c, c .- ucon, 0); init = 0.0)
    viol_var = maximum(max.(lvar .- x, x .- uvar, 0); init = 0.0)
    return (viol_con, viol_var)
end

function result_row(case, m, stats)
    c = stats.counters
    viol = try
        violation(m, stats)
    catch err
        @warn "$case: could not evaluate the violation of the returned point" exception = err
        (NaN, NaN)
    end
    return Any[
        case, Int(stats.status), stats.iter, dims(m)...,
        stats.objective, stats.primal_feas, stats.dual_feas,
        c.factorization_cnt, c.backsolve_cnt,
        c.obj_cnt, c.obj_grad_cnt, c.con_cnt, c.con_jac_cnt, c.lag_hess_cnt,
        c.total_time, c.linear_solver_time, c.eval_function_time, c.init_time,
        viol...,
    ]
end

function error_row(case, m)
    d = m === nothing ? (0, 0, 0, 0) : dims(m)
    return Any[case, ERROR_STATUS, 0, d..., NaN, NaN, NaN, 0, 0, 0, 0, 0, 0, 0, NaN, NaN, NaN, NaN, NaN, NaN]
end

function write_meta(path, fields)
    meta = Dict{String, Any}(
        "timestamp" => string(Dates.now()),
        "host" => gethostname(),
        "julia_version" => string(VERSION),
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
        "cpu_model" => strip(Sys.cpu_info()[1].model),
        "cpu_threads" => Sys.CPU_THREADS,
        "memory_gb" => round(Sys.total_memory() / 2^30; digits = 1),
    )
    # the versions of the loaded packages that the active project depends on directly
    direct = keys(get(TOML.parsefile(Base.active_project()), "deps", Dict{String, Any}()))
    for mod in Base.loaded_modules_array()
        name = string(nameof(mod))
        v = pkgversion(mod)
        name in direct && v !== nothing && (meta["pkg_" * name] = string(v))
    end
    merge!(meta, fields)
    open(path, "w") do io
        for (k, v) in sort!(collect(meta); by = first)
            println(io, k, " = ", v isa AbstractString ? repr(String(v)) : v)
        end
    end
    return
end

"""
    run_benchmark(set, cases, configs; outdir, backend = nothing, options = (;), ...)

Solve every case with every configuration and write `<outdir>/<set>_<config>.csv`.

- `cases`: a vector of `name => build`, where `build(backend)` returns an `AbstractNLPModel`.
- `configs`: a vector of `name => options`, the keyword arguments that select the
  configuration in `madnlp(model; options...)`.
- `options`: keyword arguments passed to every solve (tolerance, wall limit, print level).
- `meta`: extra fields for the metadata file.
- `warmup`: solve once, discarded, before the timed solve whenever a configuration meets a
  model type for the first time in the session, so that no recorded time includes the
  compilation of that type's code. A discarded solve that reaches the wall limit is recorded
  instead of being repeated.
- `overwrite`: start the result files afresh. By default a case already present in a result
  file is skipped, so that an interrupted run is resumed by the same command.
- `release`: called with the model after its solves, for models that hold outside resources;
  `reclaim`: called after each case, to return device memory before the next model is built.
"""
function run_benchmark(
        set::AbstractString, cases, configs; outdir::AbstractString, backend = nothing,
        options = (;), meta = Dict{String, Any}(), warmup::Bool = true, overwrite::Bool = false,
        release = m -> nothing, reclaim = () -> nothing,
    )
    mkpath(outdir)
    file(config) = joinpath(outdir, "$(set)_$(config).csv")
    rows = Dict{String, Matrix{Any}}()
    done = Dict{String, Set{String}}()
    for (config, opts) in configs
        rows[config] = reshape(Any[HEADER...], 1, :)
        done[config] = Set{String}()
        if !overwrite && isfile(file(config)) && filesize(file(config)) > 0
            old = readdlm(file(config), ',', Any)
            old[1, :] == HEADER || error("$(file(config)) has different columns; move it away or pass overwrite = true")
            rows[config] = old
            union!(done[config], string.(old[2:end, 1]))
        end
        write_meta(
            file(config) * ".meta.toml",
            merge(Dict{String, Any}("set" => set, "config" => config, "config_options" => string(opts), "options" => string(options)), meta),
        )
    end
    warmed = Set{Tuple{String, DataType}}()
    for (case, build) in cases
        todo = [(config, opts) for (config, opts) in configs if case ∉ done[config]]
        isempty(todo) && continue
        m = try
            build(backend)
        catch err
            @warn "$case: the model could not be built" exception = (err, catch_backtrace())
            nothing
        end
        for (config, opts) in todo
            row = if m === nothing
                error_row(case, m)
            else
                try
                    stats = nothing
                    if warmup && (config, typeof(m)) ∉ warmed
                        stats = madnlp(m; options..., opts...)
                        push!(warmed, (config, typeof(m)))
                        stats.status == MadNLP.MAXIMUM_WALLTIME_EXCEEDED || (stats = nothing)
                    end
                    stats === nothing && (stats = madnlp(m; options..., opts...))
                    result_row(case, m, stats)
                catch err
                    @warn "$case [$config]: the solve threw" exception = (err, catch_backtrace())
                    error_row(case, m)
                end
            end
            @info @sprintf("%-32s %-22s status %4d  iter %5d  time %10.3f s", case, config, row[2], row[3], row[18])
            rows[config] = vcat(rows[config], permutedims(row))
            writedlm(file(config), rows[config], ',')   # after every solve: an interrupted run keeps its rows
        end
        m === nothing || release(m)
        m = nothing
        GC.gc()
        reclaim()
    end
    return [file(config) for (config, _) in configs]
end

# ------------------------------------------------------------------------------------------
# Reading and scoring result files

"The rows of a result file as a vector of `NamedTuple`s."
function read_results(path::AbstractString)
    data = readdlm(path, ',', Any)
    names = Tuple(Symbol.(data[1, :]))
    return [NamedTuple{names}(Tuple(data[i, :])) for i in axes(data, 1)[(begin + 1):end]]
end

"`SOLVE_SUCCEEDED` or `SOLVED_TO_ACCEPTABLE_LEVEL`."
solved(r) = r.status == Int(MadNLP.SOLVE_SUCCEEDED) || r.status == Int(MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL)

"The time an instance is scored with: its solve time if solved, the wall limit otherwise."
score_time(r, wall) = solved(r) && r.t_total isa Real && isfinite(r.t_total) ? min(Float64(r.t_total), wall) : wall

"Shifted geometric mean of `times` with shift `shift`."
sgm(times; shift = 10.0) = isempty(times) ? NaN : exp(sum(t -> log(t + shift), times) / length(times)) - shift

"Size class of an instance by the nonzeros of its Jacobian and Hessian together."
size_class(r) = (n = r.nnzj + r.nnzh; n < 2^18 ? :small : n < 2^20 ? :medium : :large)

end # module
