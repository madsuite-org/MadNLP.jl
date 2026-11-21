#=
    MadCore._update_S_and_Y!
=#

@kernel function _update_S_and_Y_kernel!(k, S, Y)
    i = @index(Global)
    @inbounds for j in 1:(k - 1)
        S[i, j] = S[i, j + 1]
        Y[i, j] = Y[i, j + 1]
    end
end

#=
    MadCore._update_L_and_D!
=#
@kernel function _update_L_and_D_kernel!(L)
    i, j = @index(Global, NTuple)
    T = eltype(L)
    @inbounds if i < j
        L[i, j] = zero(T)
    end
end
