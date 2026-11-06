function build_inertia_corrector(::Type{InertiaBased}, ::Type{VT}, n, m, nlb, nub, ind_lb, ind_ub) where VT
    return InertiaBased()
end
function build_inertia_corrector(::Type{InertiaIgnore}, ::Type{VT}, n, m, nlb, nub, ind_lb, ind_ub) where VT
    return InertiaIgnore()
end
function build_inertia_corrector(::Type{InertiaFree}, ::Type{VT}, n, m, nlb, nub, ind_lb, ind_ub) where VT
    p0 = UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)
    d0 = UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)
    t = VT(undef, n)
    wx= VT(undef, n)
    g = VT(undef, n)
    
    return InertiaFree(
        p0, d0, t, wx, g
    )
end
