# The tolerances have the type `T` of the solver, which can be any `Real`
# (e.g. a `ForwardDiff.Dual`), not only a type that converts to `Float64`.
@kwdef mutable struct RichardsonOptions{T <: Real} <: AbstractOptions
    richardson_max_iter::Int = 10
    richardson_tol::T
    richardson_acceptable_tol::T
end

Base.convert(::Type{RichardsonOptions{T}}, opt::RichardsonOptions) where {T} =
    RichardsonOptions{T}(opt.richardson_max_iter, opt.richardson_tol, opt.richardson_acceptable_tol)

struct RichardsonIterator{T, KKT <: AbstractKKTSystem{T}} <: AbstractIterator{T}
    kkt::KKT
    opt::RichardsonOptions{T}
    cnt::MadNLPCounters
    logger::MadNLPLogger
end

function RichardsonIterator(
    kkt::AbstractKKTSystem{T};
    opt = default_options(RichardsonIterator, sqrt(eps(T))),
    logger = MadNLPLogger(),
    cnt = MadNLPCounters()
) where {T}
    return RichardsonIterator{T, typeof(kkt)}(
        kkt, opt, cnt, logger
    )
end

default_options(::Type{RichardsonIterator}, tol) = RichardsonOptions(richardson_tol=tol^(5/4), richardson_acceptable_tol=tol^(5/8))

function solve_refine!(
    x::VT,
    iterator::R,
    b::VT,
    w::VT
    ) where {T, VT, R <: RichardsonIterator{T}}
    @debug(iterator.logger, "Iterative solver initiated")

    norm_b = norm(full(b), Inf)
    residual_ratio = zero(T)

    fill!(full(x), zero(T))
    # a zero right-hand side takes no backsolve
    iterator.cnt.ir = 0

    if norm_b != zero(T)
        copyto!(full(w), full(b))

        while true
            solve_kkt!(iterator.kkt, w)
            axpy!(one(T), full(w), full(x))
            copyto!(full(w), full(b))

            @timed_region iterator.cnt :kkt_mul_time mul!(w, iterator.kkt, x, -one(T), one(T))
            iterator.cnt.kkt_mul_cnt += 1

            norm_w = norm(full(w), Inf)
            norm_x = norm(full(x), Inf)
            residual_ratio = norm_w / (min(norm_x, 1e6 * norm_b) + norm_b)

            if mod(iterator.cnt.ir, 10)==0
                @debug(iterator.logger,"iterator.cnt.ir ||res||")
            end
            @debug(iterator.logger, @sprintf("%4i %6.2e", iterator.cnt.ir, residual_ratio))
            iterator.cnt.ir += 1

            if (iterator.cnt.ir >= iterator.opt.richardson_max_iter) || (residual_ratio < iterator.opt.richardson_tol)
                break
            end
        end
    end

    @debug(
        iterator.logger,
        @sprintf(
            "Iterative solver terminated with %4i refinement steps and residual = %6.2e",
            iterator.cnt.ir, residual_ratio
        ),
    )

    return residual_ratio < iterator.opt.richardson_acceptable_tol
end

