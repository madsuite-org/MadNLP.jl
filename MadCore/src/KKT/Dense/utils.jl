
# For templating
const AbstractDenseKKTSystem{T, VT, MT, QN} = Union{
    DenseKKTSystem{T, VT, MT, QN},
    DenseCondensedKKTSystem{T, VT, MT, QN},
}

#=
    Generic functions
=#

function jtprod!(y::AbstractVector, kkt::AbstractDenseKKTSystem, x::AbstractVector)
    nx = size(kkt.hess, 1)
    ind_ineq = kkt.ind_ineq
    ns = length(ind_ineq)
    yx = view(y, 1:nx)
    ys = view(y, 1+nx:nx+ns)
    # / x
    mul!(yx, kkt.jac', x)
    # / s
    ys .= -@view(x[ind_ineq])
    return
end

function compress_jacobian!(kkt::AbstractDenseKKTSystem)
    return
end

nnz_jacobian(kkt::AbstractDenseKKTSystem) = length(kkt.jac)
