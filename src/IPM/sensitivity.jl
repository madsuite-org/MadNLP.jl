# Parametric sensitivity ds/dθ = -M^-1 * N_θ where M is the KKT system and N_θ = [∂²L/∂x∂θ; ∂c/∂θ; 0; 0]
"""
    backsolve_kkt!(solver::AbstractMadNLPSolver, p::AbstractMatrix; refactorize_kkt = true)
    backsolve_kkt!(solver::AbstractMadNLPSolver, p::AbstractVector; refactorize_kkt = true)

Solves `M d = p` where `M` is the KKT matrix at the solution of [`solve!`](@ref).
Returns a NamedTuple `d = (; dx, dy, dzl, dzu)`.

- `p`: `[px; py; pzl; pzu]`
- `refactorize_kkt`: refactorize `M` at the solution
"""
function backsolve_kkt!(solver::AbstractMadNLPSolver{T}, p::AbstractMatrix; refactorize_kkt::Bool = true) where {T}
    # Check for convergence / supported features / dimension consistency
    _check_backsolve(solver, p)

    # Refactorize kkt system at the final iterate
    if refactorize_kkt
        _refactorize_kkt!(solver)
    end

    # Unpack solver info and get indicies
    nlp = get_nlp(solver)
    nvar, ncon, k = get_nvar(nlp), get_ncon(nlp), size(p, 2)
    ind_px, ind_py = 1:nvar, nvar .+ (1:ncon)
    ind_pzl, ind_pzu = nvar + ncon .+ (1:nvar), 2nvar + ncon .+ (1:nvar)
    dx, dy, dzl, dzu = (fill!(similar(get_x0(nlp), n, k), zero(T)) for n in (nvar, ncon, nvar, nvar))
    cb, kkt = get_cb(solver), get_kkt(solver)
    rhs, d, w, zbuf = get_p(solver), get_d(solver), get__w4(solver), primal(get__w1(solver))

    # Backsolve for each column of p and unpack solution into dx, dy, dzl, dzu
    for j in 1:k
        fill!(full(rhs), zero(T))
        view(primal(rhs), 1:nvar) .= view(p, ind_px, j) .* (cb.obj_sign * cb.obj_scale[])
        dual(rhs) .= view(p, ind_py, j) .* cb.con_scale
        for (ind_pz, ind, rz, sign) in (
                (ind_pzl, kkt.ind_lb, dual_lb(rhs), one(T)),
                (ind_pzu, kkt.ind_ub, dual_ub(rhs), -one(T)),
            )
            fill!(zbuf, zero(T))
            view(zbuf, 1:nvar) .= view(p, ind_pz, j) .* (sign * cb.obj_scale[])
            rz .= view(zbuf, ind)
        end
        if !solve_refine_wrapper!(d, solver, rhs, w)
            error("KKT backsolve failed")
        end
        view(dx, :, j) .= view(primal(d), 1:nvar)
        unpack_y!(view(dy, :, j), cb, dual(d))
        for (dz, ind, dzd) in ((dzl, kkt.ind_lb, dual_lb(d)), (dzu, kkt.ind_ub, dual_ub(d)))
            fill!(zbuf, zero(T))
            zbuf[ind] .= dzd
            unpack_z!(view(dz, :, j), cb, view(zbuf, 1:nvar))
        end
    end

    return (; dx, dy, dzl, dzu)
end
backsolve_kkt!(solver::AbstractMadNLPSolver, p::AbstractVector; refactorize_kkt::Bool = true) =
    map(vec, backsolve_kkt!(solver, reshape(p, :, 1); refactorize_kkt))

# Check for convergence / supported features / dimension consistency
function _check_backsolve(solver::AbstractMadNLPSolver, p::AbstractMatrix)
    if !(get_status(solver) in (SOLVE_SUCCEEDED, SOLVED_TO_ACCEPTABLE_LEVEL))
        error("backsolve_kkt! requires a solver on which solve! has converged")
    end
    cb = get_cb(solver)
    if cb.fixed_handler isa MakeParameter && !isempty(cb.fixed_handler.fixed)
        throw(ArgumentError("sensitivity does not support fixed variables with MakeParameter"))
    end
    if get_opt(solver).hessian_approximation <: AbstractQuasiNewton
        throw(ArgumentError("sensitivity does not support quasi-Newton Hessian approximations"))
    end
    nlp = get_nlp(solver)
    if size(p, 1) != 3get_nvar(nlp) + get_ncon(nlp)
        throw(ArgumentError("p must be (3nvar + ncon) × k"))
    end
    return nothing
end

# Refactorize kkt system at the final iterate
function _refactorize_kkt!(solver::AbstractMadNLPSolver)
    kkt = get_kkt(solver)
    if !get_opt(solver).hessian_constant
        eval_lag_hess_wrapper!(solver, kkt, get_x(solver), get_y(solver))
    end
    set_aug_diagonal!(kkt, solver)
    set_aug_rhs!(solver, kkt, get_c(solver), get_mu(solver))
    dual_inf_perturbation!(
        primal(get_p(solver)),
        get_ind_llb(solver),
        get_ind_uub(solver),
        get_mu(solver),
        get_opt(solver).kappa_d
    )
    if !inertia_correction!(get_inertia_corrector(solver), solver)
        error("KKT refactorization at the solution failed!")
    end
    if get_del_w(solver) > 0 || get_del_c(solver) > 0
        @warn(get_logger(solver), "KKT matrix at the solution was regularized, so the sensitivity is approximate")
    end
    return nothing
end

# Check for dimension consistency
function _check_par_sizes(solver::AbstractMadNLPSolver, Hxθ::AbstractMatrix, Jθ::AbstractMatrix)
    nlp = get_nlp(solver)
    if size(Hxθ, 1) != get_nvar(nlp) || size(Jθ, 1) != get_ncon(nlp) || size(Hxθ, 2) != size(Jθ, 2)
        throw(DimensionMismatch("Hxθ must be nvar × nθ and Jθ must be ncon × nθ"))
    end
    return nothing
end

# Check if the model has parameters & warn if variable bounds depends on parameters
function _check_par_model(solver::AbstractMadNLPSolver)
    nlp = get_nlp(solver)
    if get_npar(nlp) == 0
        throw(ArgumentError("the model has no parameters"))
    end
    nnzj_bounds_par = (get_nnzj_lvar_par(nlp), get_nnzj_uvar_par(nlp), get_nnzj_lcon_par(nlp), get_nnzj_ucon_par(nlp))
    if any(>(0), nnzj_bounds_par)
        @warn(get_logger(solver), "bounds that depend on parameters are not supported, so the result may be incorrect")
    end
    return nothing
end

# Unpack x, y, zl, zu at the solution
function _get_solution(solver::AbstractMadNLPSolver)
    nlp, cb = get_nlp(solver), get_cb(solver)
    x, zl, zu = (similar(get_x0(nlp)) for _ in 1:3)
    y = similar(get_x0(nlp), get_ncon(nlp))
    unpack_x!(x, cb, variable(get_x(solver)))
    unpack_y!(y, cb, get_y(solver))
    unpack_z!(zl, cb, variable(get_zl(solver)))
    unpack_z!(zu, cb, variable(get_zu(solver)))
    return (; x, y, zl, zu)
end

"""
    sensitivity!(solver::AbstractMadNLPSolver; refactorize_kkt = true)
    sensitivity!(solver::AbstractMadNLPSolver, Hxθ::AbstractMatrix, Jθ::AbstractMatrix; refactorize_kkt = true)

Evaluates the parametric sensitivity jacobian `{dx, dy, dzl, dzu}/dθ` at the solution.
Returns a NamedTuple `(; dx, dy, dzl, dzu)` of matrices with `nθ` columns.

- `Hxθ`: `∂²L/∂x∂θ` at the solution, `nvar × nθ`
- `Jθ`: `∂c/∂θ` at the solution, `ncon × nθ`
- `refactorize_kkt`: refactorize the KKT matrix at the solution
"""
function sensitivity!(solver::AbstractMadNLPSolver, Hxθ::AbstractMatrix, Jθ::AbstractMatrix; refactorize_kkt::Bool = true)
    _check_par_sizes(solver, Hxθ, Jθ)
    return backsolve_kkt!(solver, [-Hxθ; -Jθ; fill!(similar(Hxθ, 2size(Hxθ, 1), size(Hxθ, 2)), 0)]; refactorize_kkt)
end
function sensitivity!(solver::AbstractMadNLPSolver; refactorize_kkt::Bool = true)
    _check_par_model(solver)
    nlp, sol = get_nlp(solver), _get_solution(solver)
    x, y, nθ = sol.x, sol.y, get_npar(nlp)
    Hxθ = hess_par_dense!(nlp, x, y, similar(x, get_nvar(nlp), nθ))
    Jθ = jac_par_dense!(nlp, x, similar(x, get_ncon(nlp), nθ))
    return sensitivity!(solver, Hxθ, Jθ; refactorize_kkt)
end

"""
    sensitivity_jvp!(solver::AbstractMadNLPSolver, dθ::AbstractVector; refactorize_kkt = true)
    sensitivity_jvp!(solver::AbstractMadNLPSolver, Hxθ::AbstractMatrix, Jθ::AbstractMatrix, dθ::AbstractVector; refactorize_kkt = true)

Evaluates the parametric sensitivity jacobian-vector product `{dx, dy, dzl, dzu}/dθ * dθ` at the solution along `dθ`.
Returns a NamedTuple `(; dx, dy, dzl, dzu)` of vectors.

- `Hxθ`: `∂²L/∂x∂θ` at the solution, `nvar × nθ`
- `Jθ`: `∂c/∂θ` at the solution, `ncon × nθ`
- `dθ`: direction in the parameters, `nθ`
- `refactorize_kkt`: refactorize the KKT matrix at the solution
"""
function sensitivity_jvp!(solver::AbstractMadNLPSolver, Hxθ::AbstractMatrix, Jθ::AbstractMatrix, dθ::AbstractVector; refactorize_kkt::Bool = true)
    _check_par_sizes(solver, Hxθ, Jθ)
    return backsolve_kkt!(solver, [-(Hxθ * dθ); -(Jθ * dθ); fill!(similar(Hxθ, 2size(Hxθ, 1)), 0)]; refactorize_kkt)
end
function sensitivity_jvp!(solver::AbstractMadNLPSolver, dθ::AbstractVector; refactorize_kkt::Bool = true)
    _check_par_model(solver)
    nlp, sol = get_nlp(solver), _get_solution(solver)
    x, y = sol.x, sol.y
    dθ = copyto!(similar(x, length(dθ)), dθ)
    Hxθdθ = hprod_par(nlp, x, y, dθ)
    Jθdθ = jprod_par(nlp, x, dθ)
    return backsolve_kkt!(solver, [-Hxθdθ; -Jθdθ; fill!(similar(Hxθdθ, 2length(Hxθdθ)), 0)]; refactorize_kkt)
end

"""
    sensitivity_vjp!(solver::AbstractMadNLPSolver; vx = nothing, vy = nothing, vzl = nothing, vzu = nothing, refactorize_kkt = true)
    sensitivity_vjp!(solver::AbstractMadNLPSolver, Hxθ::AbstractMatrix, Jθ::AbstractMatrix; vx = nothing, vy = nothing, vzl = nothing, vzu = nothing, refactorize_kkt = true)

Evaluates the parametric sensitivity vector-jacobian product `vxᵀ dx/dθ + vyᵀ dy/dθ + vzlᵀ dzl/dθ + vzuᵀ dzu/dθ` at the solution.
Returns a vector with `nθ` entries.

- `Hxθ`: `∂²L/∂x∂θ` at the solution, `nvar × nθ`
- `Jθ`: `∂c/∂θ` at the solution, `ncon × nθ`
- `vx`, `vy`, `vzl`, `vzu`: vectors of length `nvar`, `ncon`, `nvar`, `nvar`
- `refactorize_kkt`: refactorize the KKT matrix at the solution
"""
function sensitivity_vjp!(
        solver::AbstractMadNLPSolver, Hxθ::AbstractMatrix, Jθ::AbstractMatrix;
        vx = nothing, vy = nothing, vzl = nothing, vzu = nothing, refactorize_kkt::Bool = true,
    )
    _check_par_sizes(solver, Hxθ, Jθ)
    sol = _get_solution(solver)
    d = _backsolve_vjp!(solver, sol, vx, vy, vzl, vzu, refactorize_kkt)
    return .-(Hxθ' * d.dx .+ Jθ' * d.dy)
end
function sensitivity_vjp!(
        solver::AbstractMadNLPSolver;
        vx = nothing, vy = nothing, vzl = nothing, vzu = nothing, refactorize_kkt::Bool = true,
    )
    _check_par_model(solver)
    nlp, sol = get_nlp(solver), _get_solution(solver)
    x, y = sol.x, sol.y
    d = _backsolve_vjp!(solver, sol, vx, vy, vzl, vzu, refactorize_kkt)
    return .-(htprod_par(nlp, x, y, d.dx) .+ jtprod_par(nlp, x, d.dy))
end

# Backsolve with M so that d.dx, d.dy are the x, y blocks of M⁻ᵀ [vx; vy; vzl; vzu]
function _backsolve_vjp!(solver::AbstractMadNLPSolver, sol, vx, vy, vzl, vzu, refactorize_kkt)
    if all(isnothing, (vx, vy, vzl, vzu))
        throw(ArgumentError("at least one of vx, vy, vzl, vzu is required"))
    end
    nlp, cb = get_nlp(solver), get_cb(solver)
    nvar, ncon = get_nvar(nlp), get_ncon(nlp)
    zl, zu = sol.zl, sol.zu
    vx = _zero_if_nothing(vx, nvar, zl, :vx)
    vy = _zero_if_nothing(vy, ncon, zl, :vy)
    vzl = _zero_if_nothing(vzl, nvar, zl, :vzl)
    vzu = _zero_if_nothing(vzu, nvar, zl, :vzu)
    p = [vx; vy; -cb.obj_sign .* zl .* vzl; -cb.obj_sign .* zu .* vzu]
    return backsolve_kkt!(solver, p; refactorize_kkt)
end

function _zero_if_nothing(v, n, x, name)
    if isnothing(v)
        return fill!(similar(x, n), zero(eltype(x)))
    end
    if length(v) != n
        throw(DimensionMismatch("$name must have length $n, got $(length(v))"))
    end
    return copyto!(similar(x, n), v)
end
