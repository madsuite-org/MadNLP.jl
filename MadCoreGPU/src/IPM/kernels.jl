# NOTE: All of these could probably be made more efficient via custom kernels or nonallocating mapreduce.
#       For now we just use the standard mapreduce functionality which may allocate.

function get_varphi(
    obj_val,
    x_lr::VT,
    xl_r::VT,
    xu_r::VT,
    x_ur::VT,
    mu
) where {T, VT <: AbstractGPUVectorOrSubVector{T}}
    return obj_val + mapreduce(
        (x1,x2) -> _get_varphi(x1,x2,mu), +, x_lr, xl_r
    ) + mapreduce(
        (x1,x2) -> _get_varphi(x1,x2,mu), +, xu_r, x_ur
    )
end

function get_inf_du(f::VT, zl::VT, zu::VT, jacl::VT, sd) where {VT <: AbstractGPUVectorOrSubVector}
    return mapreduce((f,zl,zu,jacl) -> abs(f-zl+zu+jacl), max, f, zl, zu, jacl; init = zero(eltype(f))) / sd
end


function get_inf_compl(x_lr::VT, xl_r::VT, zl_r::VT, xu_r::VT, x_ur::VT, zu_r::VT, mu, sc) where {VT <: AbstractGPUVectorOrSubVector}
    return max(
        mapreduce(
            (x_lr, xl_r, zl_r) -> abs((x_lr-xl_r)*zl_r-mu),
            max,
            x_lr, xl_r, zl_r;
            init = zero(eltype(x_lr))
        ),
        mapreduce(
            (xu_r, x_ur, zu_r) -> abs((xu_r-x_ur)*zu_r-mu),
            max,
            xu_r, x_ur, zu_r;
            init = zero(eltype(x_lr))
        )
    ) / sc
end

function get_min_complementarity(
    x_lr::AbstractGPUVectorOrSubVector{T},
    xl_r::AbstractGPUVectorOrSubVector{T},
    zl_r::AbstractGPUVectorOrSubVector{T},
    x_ur::AbstractGPUVectorOrSubVector{T},
    xu_r::AbstractGPUVectorOrSubVector{T},
    zu_r::AbstractGPUVectorOrSubVector{T}
) where T
    cc_lb = mapreduce((x_l, xl, zl) -> (x_l-xl)*zl, min, x_lr, xl_r, zl_r, init=T(Inf))
    cc_ub = mapreduce((x_u, xu, zu) -> (xu-x_u)*zu, min, x_ur, xu_r, zu_r, init=T(Inf))
    return min(cc_lb,cc_ub)
end

function get_varphi_d(
    f::AbstractGPUVectorOrSubVector{T},
    x::AbstractGPUVectorOrSubVector{T},
    xl::AbstractGPUVectorOrSubVector{T},
    xu::AbstractGPUVectorOrSubVector{T},
    dx::AbstractGPUVectorOrSubVector{T},
    mu,
) where T
    return mapreduce(
        (f,x,xl,xu,dx)-> (f - mu/(x-xl) + mu/(xu-x)) * dx,
        +,
        f, x, xl, xu, dx;
        init = zero(T)
    )
end

function get_alpha_max(
    x::AbstractGPUVectorOrSubVector{T},
    xl::AbstractGPUVectorOrSubVector{T},
    xu::AbstractGPUVectorOrSubVector{T},
    dx::AbstractGPUVectorOrSubVector{T},
    tau,
) where T
    return min(
        mapreduce(
            (x, xl, dx) -> dx < 0 ? (-x+xl)*tau/dx : T(Inf),
            min,

            x, xl, dx,
            init = one(T)
        ),
        mapreduce(
            (x, xu, dx) -> dx > 0 ? (-x+xu)*tau/dx : T(Inf),
            min,
            x, xu, dx,
            init = one(T)
        )
    )
end

function get_alpha_z(
    zl_r::AbstractGPUVectorOrSubVector{T},
    zu_r::AbstractGPUVectorOrSubVector{T},
    dzl::AbstractGPUVectorOrSubVector{T},
    dzu::AbstractGPUVectorOrSubVector{T},
    tau,
)  where T
    return min(
        mapreduce(
            (zl_r, dzl) -> dzl < 0 ? (-zl_r)*tau/dzl : T(Inf),
            min,
            zl_r, dzl,
            init = one(T)
        ),
        mapreduce(
            (zu_r, dzu) -> dzu < 0 ? (-zu_r)*tau/dzu : T(Inf),
            min,
            zu_r, dzu,
            init = one(T)
        )
    )
end

function get_rel_search_norm(x::AbstractGPUVectorOrSubVector{T}, dx::AbstractGPUVectorOrSubVector{T}) where T
    return mapreduce(
        (x,dx) -> abs(dx) / (one(T) + abs(x)),
        max,
        x, dx
    )
end
