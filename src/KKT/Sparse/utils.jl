
# Template to dispatch on sparse representation
const AbstractSparseKKTSystem{T, VT, MT, QN} = Union{
    SparseKKTSystem{T, VT, MT, QN},
    SparseCondensedKKTSystem{T, VT, MT, QN},
    SparseUnreducedKKTSystem{T, VT, MT, QN},
}

#=
    Generic sparse methods
=#
function build_hessian_structure(cb::SparseCallback, ::Type{<:ExactHessian})
    hess_I = create_array(cb, Int32, cb.nnzh)
    hess_J = create_array(cb, Int32, cb.nnzh)
    _hess_sparsity_wrapper!(cb,hess_I,hess_J)
    return hess_I, hess_J
end
# NB. Quasi-Newton methods require only the sparsity pattern
#     of the diagonal term to store the term ξ I.
function build_hessian_structure(cb::SparseCallback, ::Type{<:AbstractQuasiNewton})
    hess_I = create_array(cb, Int32, cb.nvar)
    hess_J = create_array(cb, Int32, cb.nvar)
    hess_I .= 1:cb.nvar
    hess_J .= 1:cb.nvar
    return hess_I, hess_J
end

function jtprod!(y::AbstractVector, kkt::AbstractSparseKKTSystem, x::AbstractVector)
    mul!(y, kkt.jac_com', x)
end

get_jacobian(kkt::AbstractSparseKKTSystem) = kkt.jac_callback

nnz_jacobian(kkt::AbstractSparseKKTSystem) = nnz(kkt.jac_raw)

function compress_jacobian!(kkt::AbstractSparseKKTSystem{T}) where T
    ns = length(kkt.ind_ineq)
    kkt.jac[end-ns+1:end] .= -one(T)
    transfer!(kkt.jac_com, kkt.jac_raw, kkt.jac_csc_map)
end

function compress_jacobian!(kkt::AbstractSparseKKTSystem{T, VT, MT}) where {T, VT, MT<:Matrix{T}}
    ns = length(kkt.ind_ineq)
    kkt.jac[end-ns+1:end] .= -one(T)
    copyto!(kkt.jac_com, kkt.jac_raw)
end

function compress_hessian!(kkt::AbstractSparseKKTSystem)
    transfer!(kkt.hess_com, kkt.hess_raw, kkt.hess_csc_map)
end

function initialize!(kkt::AbstractSparseKKTSystem{T}) where T
    fill!(kkt.reg, one(T))
    fill!(kkt.pr_diag, one(T))
    fill!(kkt.du_diag, zero(T))
    fill!(kkt.hess, zero(T))
    fill!(kkt.l_lower, zero(T))
    fill!(kkt.u_lower, zero(T))
    fill!(kkt.l_diag, one(T))
    fill!(kkt.u_diag, one(T))
    fill!(nonzeros(kkt.hess_com), zero(T)) # so that mul! in the initial primal-dual solve has no effect
end

