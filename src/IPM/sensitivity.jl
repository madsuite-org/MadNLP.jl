# Parametric sensitivity ds/dθ = -M^-1 * N_θ
# where M is the KKT system and N_θ = [∂²L/∂x∂θ; ∂c/∂θ; 0; 0]
"""
    backsolve_kkt!(solver::AbstractMadNLPSolver, p::AbstractMatrix)
    backsolve_kkt!(solver::AbstractMadNLPSolver, p::AbstractVector)

Solves `M d = p` where `M` is the last KKT matrix factorized by [`solve!`](@ref).
Returns a NamedTuple `d = (; dx, dy, dzl, dzu)`.

- `p`: `[px; py; pzl; pzu]` or `[px; py]` with `pzl = pzu = 0`
"""
function backsolve_kkt!(solver::AbstractMadNLPSolver{T}, p::AbstractMatrix) where T
    if !(get_status(solver) in (SOLVE_SUCCEEDED, SOLVED_TO_ACCEPTABLE_LEVEL))
        error("backsolve_kkt! requires a solver on which solve! has converged")
    end

    nlp, cb, kkt = get_nlp(solver), get_cb(solver), get_kkt(solver)
    nvar, ncon, k = get_nvar(nlp), get_ncon(nlp), size(p, 2)
    if !(size(p, 1) in (nvar + ncon, 3nvar + ncon))
        throw(ArgumentError("p must be (nvar + ncon) × k or (3nvar + ncon) × k"))
    end

    rhs, d, w, zbuf = get_p(solver), get_d(solver), get__w4(solver), primal(get__w1(solver))
    nx = length(variable(get_x(solver)))
    ifree, ufree = _free_indices(cb)

    # rows of p = [px; py] or [px; py; pzl; pzu], the x and z blocks at the free variables
    ind_px, ind_py = ufree, nvar .+ (1:ncon)
    ind_pzl, ind_pzu = nvar + ncon .+ ufree, 2nvar + ncon .+ ufree
    x0 = get_x0(nlp)
    dx, dy, dzl, dzu = (fill!(similar(x0, n, k), zero(T)) for n in (nvar, ncon, nvar, nvar))

    # Back-solve each column of p and unpack solution into dx, dy, dzl, dzu
    for j in 1:k
        fill!(full(rhs), zero(T))
        view(primal(rhs), ifree) .= view(p, ind_px, j) .* (cb.obj_sign * cb.obj_scale[])
        dual(rhs) .= view(p, ind_py, j) .* cb.con_scale
        if size(p, 1) == 3nvar + ncon
            for (ind_pz, ind, rz, sign) in ((ind_pzl, kkt.ind_lb, dual_lb(rhs), one(T)), (ind_pzu, kkt.ind_ub, dual_ub(rhs), -one(T)))
                fill!(zbuf, zero(T))
                view(zbuf, ifree) .= view(p, ind_pz, j) .* (sign * cb.obj_scale[])
                rz .= view(zbuf, ind)
            end
        end
        if !solve_refine_wrapper!(d, solver, rhs, w)
            error("KKT back-solve failed")
        end
        view(dx, ufree, j) .= view(primal(d), ifree)
        unpack_y!(view(dy, :, j), cb, dual(d))
        for (dz, ind, dzd) in ((dzl, kkt.ind_lb, dual_lb(d)), (dzu, kkt.ind_ub, dual_ub(d)))
            fill!(zbuf, zero(T))
            zbuf[ind] .= dzd
            unpack_z!(view(dz, :, j), cb, view(zbuf, 1:nx))
        end
    end

    # julia shorthand for NamedTuple, (a = a, b = b, ...) <=> (; a, b, ...)
    return (; dx, dy, dzl, dzu)
end
backsolve_kkt!(solver::AbstractMadNLPSolver, p::AbstractVector) = map(vec, backsolve_kkt!(solver, reshape(p, :, 1)))

# Indices of the free variables in `cb` and in `nlp`
_free_indices(cb::AbstractCallback) = (1:get_nvar(cb.nlp), 1:get_nvar(cb.nlp))
_free_indices(cb::SparseCallback{T, VT, VI, NLP, FH}) where {T, VT, VI, NLP, FH<:MakeParameter} =
    (1:length(cb.fixed_handler.free), cb.fixed_handler.free)
_free_indices(cb::DenseCallback{T, VT, VI, NLP, FH}) where {T, VT, VI, NLP, FH<:MakeParameter} =
    (cb.fixed_handler.free, cb.fixed_handler.free)

"""
    sensitivity(solver::AbstractMadNLPSolver)
    sensitivity(solver::AbstractMadNLPSolver, Hxθ::AbstractMatrix, Jθ::AbstractMatrix)

Evaluates the parametric sensitivity jacobian `{dx, dy, dzl, dzu}/dθ` at the solution.
Returns a NamedTuple `(; dx, dy, dzl, dzu)` of matrices with `nθ` columns.

- `Hxθ`: `∂²L/∂x∂θ` at the solution, `nvar × nθ`
- `Jθ`: `∂c/∂θ` at the solution, `ncon × nθ`
"""
sensitivity(solver::AbstractMadNLPSolver, Hxθ::AbstractMatrix, Jθ::AbstractMatrix) =
    backsolve_kkt!(solver, -[Hxθ; Jθ])
function sensitivity(solver::AbstractMadNLPSolver)
    nlp, stats = get_nlp(solver), update!(MadNLPExecutionStats(solver), solver)
    x, y, nθ = stats.solution, stats.multipliers, get_npar(nlp)
    Hxθ = hess_par_dense!(nlp, x, y, similar(x, get_nvar(nlp), nθ))
    Jθ = jac_par_dense!(nlp, x, similar(x, get_ncon(nlp), nθ))
    return sensitivity(solver, Hxθ, Jθ)
end

"""
    sensitivity(solver::AbstractMadNLPSolver, dθ::AbstractVector)
    sensitivity(solver::AbstractMadNLPSolver, Hxθ::AbstractMatrix, Jθ::AbstractMatrix, dθ::AbstractVector)

Evaluates the parametric sensitivity directional derivative `{dx, dy, dzl, dzu}/dθ * dθ` at the solution along `dθ`.
Returns a NamedTuple `(; dx, dy, dzl, dzu)` of vectors.

- `dθ`: direction in the parameters, `nθ`
- `Hxθ`: `∂²L/∂x∂θ` at the solution, `nvar × nθ`
- `Jθ`: `∂c/∂θ` at the solution, `ncon × nθ`
"""
sensitivity(solver::AbstractMadNLPSolver, Hxθ::AbstractMatrix, Jθ::AbstractMatrix, dθ::AbstractVector) =
    backsolve_kkt!(solver, -[Hxθ * dθ; Jθ * dθ])
function sensitivity(solver::AbstractMadNLPSolver, dθ::AbstractVector)
    nlp, stats = get_nlp(solver), update!(MadNLPExecutionStats(solver), solver)
    x, y = stats.solution, stats.multipliers
    dθ = copyto!(similar(x, length(dθ)), dθ)
    Hxθdθ = hprod_par(nlp, x, y, dθ)
    Jθdθ = jprod_par(nlp, x, dθ)
    return backsolve_kkt!(solver, -[Hxθdθ; Jθdθ])
end
