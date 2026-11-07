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

function get_rel_search_norm(x::AbstractGPUVectorOrSubVector{T}, dx::AbstractGPUVectorOrSubVector{T}) where T
    return mapreduce(
        (x,dx) -> abs(dx) / (one(T) + abs(x)),
        max,
        x, dx
    )
end
