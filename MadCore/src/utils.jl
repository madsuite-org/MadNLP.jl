abstract type AbstractOptions end

# Pass MadNLP current status to the user callback
abstract type AbstractUserCallbackStatus end
struct UserCallbackRegular <: AbstractUserCallbackStatus end
struct UserCallbackRestore <: AbstractUserCallbackStatus end
struct UserCallbackRobust <: AbstractUserCallbackStatus end

abstract type AbstractUserCallback end
# By default, the user callback is deactivated
struct NoUserCallback <: AbstractUserCallback end
function (cb::NoUserCallback)(solver, mode::AbstractUserCallbackStatus)
    return true
end


# MadNLPLogger
@kwdef mutable struct MadNLPLogger
    print_level::LogLevels = INFO
    file_print_level::LogLevels = INFO
    file::Union{IOStream,Nothing} = nothing
end

get_level(logger::MadNLPLogger) = logger.print_level
get_file_level(logger::MadNLPLogger) = logger.file_print_level
get_file(logger::MadNLPLogger) = logger.file
finalize(logger::MadNLPLogger) = logger.file != nothing && close(logger.file)

for (name,level,color) in [(:trace,TRACE,7),(:debug,DEBUG,6),(:info,INFO,256),(:notice,NOTICE,256),(:warn,WARN,5),(:error,ERROR,9)]
    @eval begin
        macro $name(logger,str)
            gl = $get_level
            gfl= $get_file_level
            gf = $get_file
            l = $level
            c = $color
            code = quote
                if $gl($logger) <= $l
                    if $c == 256
                        println($str)
                    else
                        printstyled($str,"\n",color=$c)
                    end
                end
                if $gf($logger) != nothing && $gfl($logger) <= $l
                    println($gf($logger),$str)
                end
            end
            esc(code)
        end
    end
end

# Wrapper around BLAS.scal! using an explicit length and stride.
function _scal!(a::T, x::AbstractVector{T}) where T
    return BLAS.scal!(length(x), a, x, 1)
end

# BLAS
# _ger! wraps ger! to dispatch on the data type.
_ger!(alpha::T, x::AbstractVector{T}, y::AbstractVector{T}, A::AbstractMatrix{T}) where T = BLAS.ger!(alpha, x, y, A)

# Similarly, _syr! wraps syr! to dispatch on the data type.
_syr!(uplo::Char, alpha::T, x::AbstractVector{T}, A::AbstractMatrix{T}) where T = BLAS.syr!(uplo, alpha, x, A)

# Similarly, _symv! wraps symv! and dispatch based on the data type
function _symv!(uplo::Char, alpha::T, A::AbstractMatrix{T}, x::AbstractVector{T}, beta::T, y::AbstractVector{T}) where T
    return BLAS.symv!(uplo, alpha, A, x, beta, y)
end

# Similarly, _syrk! wraps syrk! to dispatch on the data type.
function _syrk!(uplo::Char, trans::Char, alpha::T, A::AbstractMatrix{T}, beta::T, C::AbstractMatrix{T}) where T
    return BLAS.syrk!(uplo, trans, alpha, A, beta, C)
end

# Similarly, _trsm! wraps trsm! to dispatch on the data type.
function _trsm!(side::Char, uplo::Char, transa::Char, diag::Char, alpha::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T
    return BLAS.trsm!(side, uplo, transa, diag, alpha, A, B)
end

# Similarly, _dgmm! wraps dgmm! to dispatch on the data type.
function _dgmm!(side::Char, A::AbstractMatrix{T}, x::AbstractVector{T}, B::AbstractMatrix{T}) where T
    if side == 'L'
        copyto!(B, A)
        lmul!(Diagonal(x), B)
    elseif side == 'R'
        copyto!(B, A)
        rmul!(B, Diagonal(x))
    else
        error("Unsupported side = $side.")
    end
end

const blas_num_threads = Ref{Int}(1)
function set_blas_num_threads(n::Integer;permanent::Bool=false)
    permanent && (blas_num_threads[]=n)
    BLAS.set_num_threads(n)
end
macro blas_safe_threads(args...)
    code = quote
        set_blas_num_threads(1)
        Threads.@threads($(args...))
        set_blas_num_threads(blas_num_threads[])
    end
    return esc(code)
end

# unsafe wrap
function _madnlp_unsafe_wrap(vec::VT, n, shift=1) where VT
    return unsafe_wrap(VT, pointer(vec,shift), n)
end

# Type definitions for noncontiguous views
const SubVector{Tv,VT, VI} = SubArray{Tv, 1, VT, Tuple{VI}, false}

"""
    AbstractTimer

Accumulates the time spent in a code region into a `Float64` field of
`MadNLPCounters`. A timer implements [`timer_start!`](@ref),
[`timer_stop!`](@ref) and [`timer_flush!`](@ref); [`create_timer`](@ref)
picks the timer for a vector type. Regions may nest. The timer keeps the start
marks itself, so a call through the abstract `MadNLPCounters.timer` field does
not allocate.
"""
abstract type AbstractTimer end

"""
    WallTimer <: AbstractTimer

Measures host wall-clock time. Used for host arrays, where every region
has finished when it returns.
"""
struct WallTimer <: AbstractTimer
    starts::Vector{Float64}
end
WallTimer() = WallTimer(Float64[])

"""
    timer_start!(timer::AbstractTimer)

Mark the start of a timed region.
"""
timer_start!(timer::WallTimer) = (push!(timer.starts, time()); nothing)

"""
    timer_stop!(timer::AbstractTimer, cnt::MadNLPCounters, ::Val{field})

Mark the end of the innermost open region and add its duration to
`getfield(cnt, field)`. A timer for asynchronous devices may add the duration
later, in a subsequent call or in [`timer_flush!`](@ref).
"""
function timer_stop!(timer::WallTimer, cnt, ::Val{field}) where {field}
    setfield!(cnt, field, getfield(cnt, field) + (time() - pop!(timer.starts)))
    return
end

"""
    timer_flush!(timer::AbstractTimer, cnt::MadNLPCounters)

Add every pending duration to `cnt`, waiting for the device if needed.
"""
timer_flush!(::WallTimer, cnt) = nothing

"""
    create_timer(::Type{VT})

Return the [`AbstractTimer`](@ref) used for a solver working on vectors of type `VT`.
"""
create_timer(::Type{VT}) where {VT} = WallTimer()

"""
    @timed_region cnt field ex

Evaluate `ex` and add the time it takes to `getfield(cnt, field)` with
`cnt.timer`. `field` is a literal `Symbol`. Returns the value of `ex`.
"""
macro timed_region(cnt, field, ex)
    return quote
        local _cnt = $(esc(cnt))
        timer_start!(_cnt.timer)
        local _val = $(esc(ex))
        timer_stop!(_cnt.timer, _cnt, Val($(esc(field))))
        _val
    end
end

@kwdef mutable struct MadNLPCounters
    k::Int = 0 # total iteration counter
    l::Int = 0 # backtracking line search counter
    t::Int = 0 # restoration phase counter
    ir::Int = 0 # iterative refinement counter

    start_time::Float64
    timer::AbstractTimer = WallTimer()

    linear_solver_time::Float64 = 0.
    eval_function_time::Float64 = 0.
    solver_time::Float64 = 0.
    total_time::Float64 = 0.
    init_time::Float64 = 0.

    obj_cnt::Int = 0
    obj_grad_cnt::Int = 0
    con_cnt::Int = 0
    con_jac_cnt::Int = 0
    lag_hess_cnt::Int = 0

    factorization_cnt::Int = 0
    backsolve_cnt::Int = 0

    t1::Float64 = 0.
    t2::Float64 = 0.
    t3::Float64 = 0.
    t4::Float64 = 0.
    t5::Float64 = 0.
    t6::Float64 = 0.
    t7::Float64 = 0.
    t8::Float64 = 0.

    acceptable_cnt::Int = 0
    unsuccessful_iterate::Int = 0
    restoration_fail_count::Int = 0
end

"""
    timing_callbacks(ips::MadNLPSolver; ntrials=10)

Return the average timings spent in each callback for `ntrials` different trials.
Results are returned inside a named-tuple.

"""
function timing_callbacks(ips; ntrials=10)
    t_f, t_c, t_g, t_j, t_h = (0.0, 0.0, 0.0, 0.0, 0.0)
    for _ in 1:ntrials
        t_f += @elapsed eval_f_wrapper(ips, ips.x)
        t_c += @elapsed eval_cons_wrapper!(ips, ips.c, ips.x)
        t_g += @elapsed eval_grad_f_wrapper!(ips, ips.f,ips.x)
        t_j += @elapsed eval_jac_wrapper!(ips, ips.kkt, ips.x)
        t_h += @elapsed eval_lag_hess_wrapper!(ips, ips.kkt, ips.x, ips.y)
    end
    return (
        time_eval_objective   = t_f / ntrials,
        time_eval_constraints = t_c / ntrials,
        time_eval_gradient    = t_g / ntrials,
        time_eval_jacobian    = t_j / ntrials,
        time_eval_hessian     = t_h / ntrials,
    )
end

"""
    timing_linear_solver(ips::MadNLPSolver; ntrials=10)

Return the average timings spent in the linear solver for `ntrials` different trials.
Results are returned inside a named-tuple.

"""
function timing_linear_solver(ips; ntrials=10)
    t_build, t_factorize, t_backsolve = (0.0, 0.0, 0.0)
    for _ in 1:ntrials
        t_build     += @elapsed build_kkt!(ips.kkt)
        t_factorize += @elapsed factorize_kkt!(ips.kkt)
        t_backsolve += @elapsed solve_kkt!(ips.kkt, ips.d)
    end
    return (
        time_build_kkt = t_build / ntrials,
        time_factorization = t_factorize / ntrials,
        time_backsolve = t_backsolve / ntrials,
    )
end

"""
    timing_madnlp(ips::MadNLPSolver; ntrials=10)

Return the average time spent in the callbacks and in the linear solver,
for `ntrials` different trials.

Results are returned as a named-tuple.

"""
function timing_madnlp(ips; ntrials=10)
    return (
        time_linear_solver=timing_linear_solver(ips; ntrials=ntrials),
        time_callbacks=timing_callbacks(ips; ntrials=ntrials),
    )
end
