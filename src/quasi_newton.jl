
"""
    AbstractHessian{T, VT}

Abstract type for representation of second-order information.

"""
abstract type AbstractHessian{T, VT} end

"""
    AbstractQuasiNewton{T, VT} <: AbstractHessian{T, VT}

Abstract type for quasi-Newton approximation.

"""
abstract type AbstractQuasiNewton{T, VT} <: AbstractHessian{T, VT} end

"""
    update!(
        qn::AbstractQuasiNewton,
        Bk::AbstractMatrix,
        sk::AbstractVector,
        yk::AbstractVector,
    )

Update the matrix `Bk` encoding the (direct) Hessian approximation
with the secant vectors `sk` and `yk`.

Return `true` if the update succeeded, `false` otherwise.
"""
function update! end

"""
    init!(
        qn::AbstractHessian{T},
        Bk::AbstractArray{T},
        g0::AbstractVector{T},
        f0::T,
    ) where T

Instantiate the Hessian estimate `Bk` with the quasi-Newton algorithm `qn`.
The function uses the initial gradient `g0` and the initial objective
`f0` to build the initial estimate.

"""
function init! end

curvature(::Val{SCALAR1}, sk, yk) = dot(yk, sk) / dot(sk, sk)
curvature(::Val{SCALAR2}, sk, yk) = dot(yk, yk) / dot(sk, yk)
function curvature(::Val{SCALAR3}, sk, yk)
    sᵀy = dot(sk, yk)
    sᵀs = dot(sk, sk)
    yᵀy = dot(yk, yk)
    return ((sᵀy / sᵀs) + (yᵀy / sᵀy)) / 2
end
function curvature(::Val{SCALAR4}, sk, yk)
    sᵀy = dot(sk, yk)
    sᵀs = dot(sk, sk)
    yᵀy = dot(yk, yk)
    return sqrt((sᵀy / sᵀs) * (yᵀy / sᵀy))
end

@kwdef mutable struct QuasiNewtonOptions{T} <: AbstractOptions
    init_strategy::BFGSInitStrategy = SCALAR1
    max_history::Int = 6
    init_value::T = 1.0
    sigma_min::T = 1e-8
    sigma_max::T = 1e+8
end

"""
    BFGS{T, VT} <: AbstractQuasiNewton{T, VT}

BFGS quasi-Newton method. Update the direct Hessian approximation using
```math
B_{k+1} = B_k - \frac{(B_k s_k)(B_k s_k)^⊤}{s_k^⊤ B_k s_k} + \frac{y_k y_k^⊤}{y_k^⊤ s_k}
```

### Notes
The matrix is not updated if ``s_k^⊤ y_k < 10^{-8}``.

"""
struct BFGS{T, VT <: AbstractVector{T}} <: AbstractQuasiNewton{T, VT}
    init_strategy::BFGSInitStrategy
    is_instantiated::Base.RefValue{Bool}
    sk::VT
    yk::VT
    bsk::VT
    last_g::VT
    last_x::VT
    last_jv::VT
end

function create_quasi_newton(
    ::Type{BFGS},
    cb::AbstractCallback{T,VT},
    n;
    options=QuasiNewtonOptions{T}(),
    ) where {T,VT}
    BFGS(
        options.init_strategy,
        Ref(false),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
    )
end

function update!(qn::BFGS{T, VT}, Bk::AbstractMatrix, sk::AbstractVector, yk::AbstractVector) where {T, VT}
    yksk = dot(sk, yk)
    if yksk < T(1e-8)
        return false
    end
    # Initial approximation (Nocedal & Wright, p.143)
    if !qn.is_instantiated[]
        sksk = dot(sk, sk)
        Bk[diagind(Bk)] .= yksk ./ sksk
        qn.is_instantiated[] = true
    end
    # BFGS update
    _symv!('L', one(T), Bk, sk, zero(T), qn.bsk)
    alpha1 = one(T) / dot(sk, qn.bsk)
    alpha2 = one(T) / yksk
    _syr!('L', -alpha1, qn.bsk, Bk)  # Bk = Bk - alpha1 * bsk * bsk'
    _syr!('L', alpha2, yk, Bk)       # Bk = Bk + alpha2 * yk * yk'
    return true
end

struct DampedBFGS{T, VT <: AbstractVector{T}} <: AbstractQuasiNewton{T, VT}
    init_strategy::BFGSInitStrategy
    is_instantiated::Base.RefValue{Bool}
    sk::VT
    yk::VT
    bsk::VT
    rk::VT
    last_g::VT
    last_x::VT
    last_jv::VT
end

function create_quasi_newton(
    ::Type{DampedBFGS},
    cb::AbstractCallback{T,VT},
    n;
    options=QuasiNewtonOptions{T}(),
    ) where {T,VT}
    return DampedBFGS(
        options.init_strategy,
        Ref(false),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
    )
end

function update!(qn::DampedBFGS{T, VT}, Bk::AbstractMatrix, sk::AbstractVector, yk::AbstractVector) where {T, VT}
    yksk = dot(sk, yk)
    # Initial approximation (Nocedal & Wright, p.143)
    if !qn.is_instantiated[]
        sksk = dot(sk, sk)
        Bk[diagind(Bk)] .= yksk ./ sksk
        qn.is_instantiated[] = true
    end

    _symv!('L', one(T), Bk, sk, zero(T), qn.bsk)
    sBs = dot(sk, qn.bsk)

    # Procedure 18.2 (Nocedal & Wright, page 537)
    theta = if dot(sk, yk) < T(0.2) * sBs
        T(0.8) * sBs / (sBs - yksk)
    else
        one(T)
    end

    fill!(qn.rk, zero(T))
    axpy!(theta, yk, qn.rk)
    axpy!(one(T) - theta, qn.bsk, qn.rk)

    alpha1 = one(T) / sBs
    alpha2 = one(T) / dot(qn.rk, qn.sk)

    _syr!('L', -alpha1, qn.bsk, Bk)
    _syr!('L', alpha2, qn.rk, Bk)
    return true
end

function init!(qn::Union{BFGS, DampedBFGS}, Bk::AbstractMatrix{T}, g0::AbstractVector{T}, f0::T) where T
    norm_g0 = dot(g0, g0)
    # Initiate B0 with Gilbert & Lemaréchal rule.
    rho0 = if norm_g0 < sqrt(eps(T))
        one(T)
    elseif f0 ≈ zero(T)
        one(T) / norm_g0
    else
        abs(f0) / norm_g0
    end
    Bk[diagind(Bk)] .= T(2) * rho0
    return
end


"""
    CompactLBFGS{T, VT} <: AbstractQuasiNewton
"""
mutable struct CompactLBFGS{T, VT <: AbstractVector{T}, MT <: AbstractMatrix{T}, B} <: AbstractQuasiNewton{T, VT}
    init_strategy::BFGSInitStrategy
    sk::VT
    yk::VT
    last_g::VT
    last_x::VT
    last_jv::VT
    init_value::T
    sigma_min::T
    sigma_max::T
    max_mem::Int
    current_mem::Int
    skipped_iter::Int
    Sk::MT       # n x p
    Yk::MT       # n x p
    Lk::MT       # p x p
    Mk::MT       # p x p (for Cholesky factorization Mₖ = Jₖ Jₖᵀ)
    Tk::MT       # 2p x 2p
    DkLk::MT     # p x p
    U::MT        # n x p
    V::MT        # n x p
    E::MT        # (n+m) x 2p
    H::MT        # (n+m) x 2p
    Dk::VT       # p
    _w1::VT
    _w2::VT
    additional_buffers::B
    max_mem_reached::Bool
end

function create_quasi_newton(
    ::Type{CompactLBFGS},
    cb::AbstractCallback{T,VT},
    n;
    options=QuasiNewtonOptions{T}(),
    ) where {T, VT}
    return CompactLBFGS(
        options.init_strategy,
        fill!(create_array(cb, n), zero(T)),
        fill!(create_array(cb, n), zero(T)),
        fill!(create_array(cb, n), zero(T)),
        fill!(create_array(cb, n), zero(T)),
        fill!(create_array(cb, n), zero(T)),
        T(options.init_value),
        T(options.sigma_min),
        T(options.sigma_max),
        options.max_history,
        0,
        0,
        fill!(create_array(cb, n, 0), zero(T)),
        fill!(create_array(cb, n, 0), zero(T)),
        fill!(create_array(cb, 0, 0), zero(T)),
        fill!(create_array(cb, 0, 0), zero(T)),
        fill!(create_array(cb, 0, 0), zero(T)),
        fill!(create_array(cb, 0, 0), zero(T)),
        fill!(create_array(cb, 0, 0), zero(T)),
        fill!(create_array(cb, 0, 0), zero(T)),
        fill!(create_array(cb, 0, 0), zero(T)),
        fill!(create_array(cb, 0, 0), zero(T)),
        fill!(create_array(cb, 0), zero(T)),
        fill!(create_array(cb, 0), zero(T)),
        fill!(create_array(cb, 0), zero(T)),
        nothing,
        false,
    )
end

Base.size(qn::CompactLBFGS) = (size(qn.Sk, 1), qn.current_mem)

function _resize!(qn::CompactLBFGS{T, VT, MT}) where {T, VT, MT}
    n, k = size(qn)
    qn.Lk   = fill!(reshape(resize!(vec(qn.Lk), k*k), k, k), zero(T))
    qn.Mk   = fill!(reshape(resize!(vec(qn.Mk), k*k), k, k), zero(T))
    qn.Tk   = fill!(reshape(resize!(vec(qn.Tk), 4*k*k), 2*k, 2*k), zero(T))
    qn.DkLk = fill!(reshape(resize!(vec(qn.DkLk), k*k), k, k), zero(T))
    qn.U    = fill!(reshape(resize!(vec(qn.U), n*k), n, k), zero(T))
    qn.V    = fill!(reshape(resize!(vec(qn.V), n*k), n, k), zero(T))
    qn._w1  = fill!(resize!(qn._w1, k), zero(T))
    qn._w2  = fill!(resize!(qn._w2, 2*k), zero(T))
    return
end

function _reset!(qn::CompactLBFGS{T, VT, MT}) where {T, VT, MT}
    n, _ = size(qn)
    qn.current_mem = 0
    qn.skipped_iter = 0
    fill!(qn.last_jv, zero(T))
    qn.Dk = VT(undef, 0)
    qn.Sk = MT(undef, n, 0)
    qn.Yk = MT(undef, n, 0)
    qn.max_mem_reached = false
    _resize!(qn)
end

function _update_S_and_Y!(qn::CompactLBFGS, s::Vector, y::Vector)
    if qn.current_mem < qn.max_mem
        qn.current_mem += 1
        n, k = size(qn)
        vec_Sk = vec(qn.Sk)
        vec_Yk = vec(qn.Yk)
        resize!(vec_Sk, n*k)
        resize!(vec_Yk, n*k)
        qn.Sk = reshape(vec_Sk, n, k)
        qn.Yk = reshape(vec_Yk, n, k)
        view(qn.Sk, 1:n, k) .= s
        view(qn.Yk, 1:n, k) .= y
        _resize!(qn)
    else
        n, k = size(qn)
        # Shift
        @inbounds for i_ in 1:k-1, j in 1:n
            qn.Sk[j, i_] = qn.Sk[j, i_+1]
            qn.Yk[j, i_] = qn.Yk[j, i_+1]
        end
        # Latest element
        @inbounds for j in 1:n
            qn.Sk[j, k] = s[j]
            qn.Yk[j, k] = y[j]
        end
    end
end

function _update_L_and_D!(qn::CompactLBFGS{T}, sk::Vector, yk::Vector) where {T}
    # Update the strict lower triangular part of S'Y
    n, k = size(qn)
    if qn.max_mem_reached
        # shift of all coefficients
        @inbounds for i in 1:k-1
            qn.Dk[i] = qn.Dk[i+1]
            @inbounds for j in 1:i-1
                qn.Lk[i, j] = qn.Lk[i+1, j+1]
            end
        end
        # Compute the new last row of tril(S'Y)
        # It can be recovered with sₖᵀY or Yᵀsₖ
        lk = view(qn.Lk, k, 1:k)
        sk = view(qn.Sk, 1:n, k)
        mul!(lk, qn.Yk', sk)
        qn.Dk[k] = qn.Lk[k,k]
        qn.Lk[k,k] = zero(T)
    else
        # To be optimized...
        p = size(qn.Lk, 1)
        mul!(qn.Lk, qn.Sk', qn.Yk)
        push!(qn.Dk, qn.Lk[k,k])
        @inbounds for i in 1:p, j in i:p
            qn.Lk[i, j] = zero(T)
        end
        if qn.current_mem == qn.max_mem
            qn.max_mem_reached = true
        end
    end
end

function update!(qn::CompactLBFGS{T}, Bk, sk, yk) where {T}
    norm_sk, norm_yk = norm(sk), norm(yk)
    # Skip update if vectors are too small or local curvature is negative.
    if ((norm_sk < T(100) * eps(T)) ||
        (norm_yk < T(100) * eps(T)) ||
        (dot(sk, yk) < sqrt(eps(T)) * norm_sk * norm_yk)
    )
        qn.skipped_iter += 1
        if qn.skipped_iter >= 2
            _reset!(qn)
        end
        return false
    end

    # Refresh internal structures
    _update_S_and_Y!(qn, sk, yk)
    _update_L_and_D!(qn, sk, yk)

    # Load buffers
    k = qn.current_mem
    δ = qn._w1

    # Compute compact representation Bₖ = σₖI -UₖUₖᵀ + VₖVₖᵀ
    # Step 1: σₖ I
    sigma = curvature(Val(qn.init_strategy), sk, yk)    # σₖ
    sigma = clamp(sigma, qn.sigma_min, qn.sigma_max)
    Bk .= sigma                                         # Hₖ .= σₖI (diagonal Hessian approx.)

    # Step 2: Mₖ = σₖ Sₖᵀ Sₖ + Lₖ Dₖ⁻¹ Lₖᵀ
    δ .= one(T) ./ sqrt.(qn.Dk)                         # δₖ = 1 / √Dₖ
    _dgmm!('L', qn.Lk', δ, qn.DkLk)                     # Compute (1 / √Dₖ) * Lₖᵀ
    if T <: BlasReal
        _syrk!('L', 'T', one(T), qn.Sk, zero(T), qn.Mk) # Mₖ = Sₖᵀ Sₖ
        _syrk!('L', 'T', one(T), qn.DkLk, sigma, qn.Mk) # Mₖ = σₖ Sₖᵀ Sₖ + Lₖ Dₖ⁻¹ Lₖᵀ
    else
        mul!(qn.Mk, qn.Sk', qn.Sk)                      # Mₖ = Sₖᵀ Sₖ
        mul!(qn.Mk, qn.DkLk', qn.DkLk, one(T), sigma)   # Mₖ = σₖ Sₖᵀ Sₖ + Lₖ Dₖ⁻¹ Lₖᵀ
    end

    # Step 3: Perform a Cholesky factorization of Mₖ
    if T <: BlasReal
        LAPACK.potrf!('L', qn.Mk)                       # Mₖ = Jₖ Jₖᵀ (factorization)
    else
        cholesky!(Symmetric(qn.Mk, :L))                 # Mₖ = Jₖ Jₖᵀ (factorization)
    end

    # Step 4: Update Uₖ and Vₖ
    _dgmm!('R', qn.Yk, δ, qn.V)                         # Vₖ = Yₖ * (1 / √Dₖ)
    copyto!(qn.U, qn.Sk)                                # Uₖ = Sₖ
    mul!(qn.U, qn.V, qn.DkLk, one(T), sigma)            # Uₖ = σₖ Sₖ + Yₖ Dₖ⁻¹ Lₖᵀ
    if T <: BlasReal
        _trsm!('R', 'L', 'T', 'N', one(T), qn.Mk, qn.U) # Uₖ = (σₖ Sₖ + Yₖ Dₖ⁻¹ Lₖᵀ) Jₖ⁻ᵀ
    else
        rdiv!(qn.U, LowerTriangular(qn.Mk)')            # Uₖ = (σₖ Sₖ + Yₖ Dₖ⁻¹ Lₖᵀ) Jₖ⁻ᵀ
    end

    return true
end

function init!(qn::CompactLBFGS{T}, Bk::AbstractVector{T}, g0::AbstractVector{T}, f0::T) where T
    norm_g0 = dot(g0, g0)
    # Initiate B0 with Gilbert & Lemaréchal rule.
    rho0 = if norm_g0 < sqrt(eps(T))
        one(T)
    elseif f0 ≈ zero(T)
        one(T) / norm_g0
    else
        abs(f0) / norm_g0
    end
    Bk .= (T(2) * rho0 * qn.init_value)
    return
end

struct ExactHessian{T, VT} <: AbstractHessian{T, VT} end
create_quasi_newton(::Type{ExactHessian}, cb::AbstractCallback{T,VT}, n; options...) where {T,VT} = ExactHessian{T, VT}()
