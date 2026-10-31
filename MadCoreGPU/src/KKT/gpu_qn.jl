function MadCore._update_S_and_Y!(qn::MadCore.CompactLBFGS, s::AbstractGPUVector, y::AbstractGPUVector)
    return if qn.current_mem < qn.max_mem
        qn.current_mem += 1
        n, k = size(qn)
        vec_Sk = vec(qn.Sk)
        vec_Yk = vec(qn.Yk)
        resize!(vec_Sk, n * k)
        resize!(vec_Yk, n * k)
        qn.Sk = reshape(vec_Sk, n, k)
        qn.Yk = reshape(vec_Yk, n, k)
        view(qn.Sk, 1:n, k) .= s
        view(qn.Yk, 1:n, k) .= y
        MadCore._resize!(qn)
    else
        n, k = size(qn)
        # Shift
        backend = get_backend(s)
        _update_S_and_Y_kernel!(backend)(k, qn.Sk, qn.Yk, ndrange = n)
        # Latest element
        view(qn.Sk, 1:n, k) .= s
        view(qn.Yk, 1:n, k) .= y
    end
end

function MadCore._update_L_and_D!(qn::MadCore.CompactLBFGS{T}, sk::AbstractGPUVector, yk::AbstractGPUVector) where {T}
    # Update the strict lower triangular part of S'Y
    n, k = size(qn)
    return if qn.max_mem_reached
        # shift of all coefficients
        @inbounds for i in 1:(k - 1)
            @allowscalar qn.Dk[i] = qn.Dk[i + 1]
            @inbounds for j in 1:(i - 1)
                @allowscalar qn.Lk[i, j] = qn.Lk[i + 1, j + 1]
            end
        end
        # Compute the new last row of tril(S'Y)
        # It can be recovered with sₖᵀY or Yᵀsₖ
        lk = view(qn.Lk, k, 1:k)
        sk = view(qn.Sk, 1:n, k)
        mul!(lk, qn.Yk', sk)
        @allowscalar qn.Dk[k] = qn.Lk[k, k]
        @allowscalar qn.Lk[k, k] = zero(T)
    else
        # To be optimized...
        p = size(qn.Lk, 1)
        mul!(qn.Lk, qn.Sk', qn.Yk)
        @allowscalar push!(qn.Dk, qn.Lk[k, k])
        backend = get_backend(sk)
        _update_L_and_D_kernel!(backend)(qn.Lk, ndrange = (p, p))
        if qn.current_mem == qn.max_mem
            qn.max_mem_reached = true
        end
    end
end
