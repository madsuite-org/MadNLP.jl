@kwdef mutable struct RichardsonOptions <: AbstractOptions
    richardson_max_iter::Int = 10
    richardson_tol::Float64
    richardson_acceptable_tol::Float64
end

struct RichardsonIterator{T, KKT <: AbstractKKTSystem{T}} <: AbstractIterator{T}
    kkt::KKT
    opt::RichardsonOptions
    cnt::MadNLPCounters
    logger::MadNLPLogger
end

function RichardsonIterator(
    kkt;
    opt = RichardsonOptions(),
    logger = MadNLPLogger(),
    cnt = MadNLPCounters()
)
    return RichardsonIterator(
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

    if norm_b != zero(T)
        copyto!(full(w), full(b))
        iterator.cnt.ir = 0

        while true
            solve_kkt!(iterator.kkt, w)
            axpy!(1., full(w), full(x))
            copyto!(full(w), full(b))

            mul!(w, iterator.kkt, x, -one(T), one(T))

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

