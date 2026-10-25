# Interior-point algorithm

```@meta
CurrentModule = MadNLP
```

We give a brief description of the interior-point algorithm
used in MadNLP, together with [the principal options](options.md)
impacting MadNLP's behavior. The algorithm is described in more length in the
[Ipopt paper](https://link.springer.com/article/10.1007/S10107-004-0559-Y).

MadNLP searches for a local solution of the nonlinear program:
```math
  \begin{aligned}
    \min_{x} \; & f(x) \;, \\
    \text{subject to} \quad & g_\ell \leq g(x) \leq g_u \; ,\\
                            & x_\ell \leq x \leq x_u \; ,
  \end{aligned}
```
where $$x \in \mathbb{R}^n$$ is the decision variable, $$f: \mathbb{R}^n \to \mathbb{R}$$
and $$g: \mathbb{R}^n \to \mathbb{R}^m$$ two smooth nonlinear functions.

## Pre-processing

Before running the interior-point method, MadNLP applies a pre-processing to the problem
to improve the numerical performance. The pre-processing operations are described below.

### Problem's reformulation

#### Slack variables
First, MadNLP splits the equality from the inequality constraints in the definition of the feasible set:
```math
g_\ell \leq g(x) \leq g_u \; .
```
If $$g_{\ell,i} < g_{u,i}$$, the constraint is reformulated as
```math
g_i(x) - s_i = 0  \; , \quad g_{\ell, i} \leq s_i \leq g_{u,i} \; ,
```
with $$s_i$$ an additional slack variable. By doing so, all the inequality
constraints are moved into the bound constraints. This benefits directly
to the interior-point algorithm, as it becomes much easier to compute a strictly feasible
initial point.

!!! info
    MadNLP only reformulates with a slack variable the inequality constraints,
    the equality constraints are left untouched.

As a result, we obtain an equivalent problem with the following structure:
```math
  \begin{aligned}
    \min_{w} \; & f(w) \; , \\
    \text{subject to} \quad & c(w) = 0 \; , \quad w \geq 0 \; .
  \end{aligned}
```
with $$w = (x, s)$$ and $$c(w) = (g_I(x) - s, g_E(x))$$, with ``I``
the index set for the inequality constraints and ``E`` the index set for equality constraints.

!!! info
    To simplify the exposition, we have assumed that the variables
    have all their lower bound sets to zero, and no upper bound.

The KKT stationary conditions associated to the reformulated problem are:
```math
  \begin{aligned}
    & \nabla f(w) + \nabla c(w)^\top y - z = 0 \; , \\
    & c(w) = 0 \; , \\
    & 0 \leq w \perp z \geq 0 \; .
  \end{aligned}
```
MadNLP looks for a primal-dual solution $$(w, y, z)$$ satisfying the KKT conditions.

The Lagrangian of the problem is:
```math
L(w, y, z) = f(w) + c(w)^\top y - z^\top w \; .
```

#### Fixed variables
If for some ``i=1, \cdots,n`` we have ``x_{\ell,i} = x_{u,i}``, the variable's lower bound is equal
to its upper bound, meaning the variable is fixed.
By default, MadNLP removes all the fixed variables in the problem.

Alternatively, MadNLP can relax all the fixed variables with a small parameter epsilon as
```math
x_{\ell,i} - \epsilon \leq x_i \leq x_{u,i} + \epsilon \; .
```
The behavior is specified by the option `fixed_variable_treatment`:
if set to [`MakeParameter`](@ref), MadNLP removes the fixed variables,
and if it is set to [`RelaxBound`](@ref) MadNLP relaxes the bounds.

#### Equality constraints

As discussed before, MadNLP keeps the equality constraints ``g_E(x) = 0 `` untouched in the problem.
However, it is sometimes appropriate to relax the equality constraints
by converting them to (tight) inequality constraints:
```math
-\tau \leq g_E(x) \leq \tau \; .
```
That behavior is determined by the option `equality_treatment`.
It is set to [`EnforceEquality`](@ref) by default. If otherwise it is set to [`RelaxEquality`](@ref),
the equality constraints are relaxed as inequality constraints.

### Scaling

Once the problem has been reformulated, MadNLP scales the objective
and the constraints to ensure that the gradient and the rows of the Jacobian
have all a norm being less than `nlp_scaling_max_gradient` (by default equal to `100.0`).
The lower the value of `nlp_scaling_max_gradient`, the more aggressive the scaling gets.

The scaling can be deactivated by setting `nlp_scaling=false`.

### Computing the initial primal-dual iterate

The user can pass to MadNLP an initial primal point ``x_0``. MadNLP modifies
it to ensure it is strictly feasible by applying the operation
```math
x_{0,j} = \max(x_{0,j}, \kappa_1) \; ,
```
with ``\kappa_1`` the parameter specified in the option `bound_push`.

If `dual_initialized=false` (default), MadNLP computes the initial dual multiplier ``y_0``
as solution of the least-square problem
```math
\begin{bmatrix}
I & J_0^\top \\
J_0 & 0
\end{bmatrix}
\begin{bmatrix}
d \\ y_0
\end{bmatrix}
=
-
\begin{bmatrix}
 \nabla f(x_0) - z_0  \\ 0
\end{bmatrix} \; .
```
Otherwise, if `dual_initialized=true`, MadNLP uses the values passed by the user.

!!! info
    The user cannot pass the initial value of the bound multiplier.
    MadNLP automatically sets ``z_0 = 1``.


## Interior-point iterations

MadNLP solves the KKT conditions iteratively using a globalized Newton algorithm.
The interior-point method reformulates the KKT conditions using a homotopy method, with,
for a positive barrier parameter $$\mu > 0$$:
```math
\begin{aligned}
& \nabla f(w) + \nabla c(w)^\top y - z = 0 \; , \\
& c(w) = 0 \; , \\
& WZe = \mu e \;, \; (w, z) > 0 \; .
\end{aligned}
```

The algorithm stops as soon as `max(inf_pr, inf_du, inf_compl) < tol`, with
```math
\begin{aligned}
& \texttt{inf\_pr} = \| c(w) \|_\infty \; , \\
& \texttt{inf\_du} = \| \nabla f(w) + \nabla c(w)^\top y - z \|_\infty \; , \\
& \texttt{inf\_compl} = \| WZe \|_\infty \; .
\end{aligned}
```
The tolerance `tol` is set to `1e-8` by default.

If the stopping criterion is not satisfied, MadNLP moves to the next iteration (here
denoted with an index ``k``). The iteration proceeds in four steps.

### Step 1: Computing the first and second-order sensitivities

First, MadNLP evaluates the Jacobian of the constraints ``J_k = \nabla c(w_k)^\top``
and the Hessian of the Lagrangian ``H_k = \nabla_{xx}^2 L(w_k, y_k, z_k)``.

MadNLP evaluates the sensitivities inside an [`AbstractCallback`](@ref),
which acts as a buffer between the solver and the model implemented with `NLPModels`.
Compared to the original model, the `AbstractCallback` adds the slack
variable and scales the problem appropriately.

By default the two matrices ``J_k`` and ``H_k`` are assumed sparse, with only the
non-zero entries being stored. If convenient, the user can evaluate them in dense format
by switching to a [`DenseCallback`](@ref) by setting the option:
```
callback=DenseCallback
```

### Step 2: Updating the barrier parameter

Once the sensitivities are evaluated, MadNLP updates the barrier parameter $$\mu$$.
By default, MadNLP uses the monotone update rule inspired by the Fiacco-McCormick rule: [`MonotoneUpdate`](@ref).
Alternatively, MadNLP provides two adaptive update rules, implemented in the solver respectively
as [`QualityFunctionUpdate`](@ref) and [`LOQOUpdate`](@ref).
The user can change the barrier update, e.g. by setting the option
```
barrier=QualityFunctionUpdate()
```

!!! info
    We recommend using an adaptive barrier update for difficult problems,
    in particular if we have a poor initial iterate ``x_0``.

### Step 3: Solving the primal-dual KKT system

With the new barrier parameter, we can compute a new KKT residual. MadNLP aims at decreasing
this residual by computing a Newton step ``(\Delta w, \Delta y, \Delta z)`` solution of
the primal-dual KKT system:
```math
\begin{bmatrix}
H_k +\delta_x I & J_k^\top & -I \\
J_k & -\delta_y & 0 \\
Z_k & 0 & W_k
\end{bmatrix}
\begin{bmatrix}
\Delta w \\ \Delta y \\ \Delta z
\end{bmatrix}
= -
\begin{bmatrix}
 \nabla f(w_k) + \nabla c(w_k)^\top y_k - z_k \\
 c(w_k)  \\
 W_k Z_k e - \mu e
\end{bmatrix} \; ,
```
with $$\delta_x$$ and $$\delta_y$$ appropriate primal-dual regularization terms
whose exact roles are detailed hereafter.


#### Solution of the primal-dual KKT system
The linear system is called a *primal-dual KKT system*. MadNLP can solve a symmetrized
version of the primal-dual KKT system, known as an [`AbstractUnreducedKKTSystem`](@ref).
However, it is beneficial to remove the blocks associated to $$\Delta z$$  and solve
the smaller [`AbstractReducedKKTSystem`](@ref):
```math
\begin{bmatrix}
H_k + \Sigma_k + \delta_x I & J_k^\top \\
J_k & -\delta_y
\end{bmatrix}
\begin{bmatrix}
\Delta w \\ \Delta y
\end{bmatrix}
= -
\begin{bmatrix}
 \nabla f(w_k) + \nabla c(w_k)^\top y_k - \mu_k W_k^{-1} e \\
 c(w_k)
\end{bmatrix}
```
with the diagonal matrix ``\Sigma_k = W_k^{-1} Z_k``.
The default implementation of the `AbstractReducedKKTSystem` is provided in [`SparseKKTSystem`](@ref).
If the problem exhibits significant ill-conditioning, the user can also use a [`ScaledSparseKKTSystem`](@ref),
which is more numerically stable than the [`SparseKKTSystem`](@ref).

There exists a smaller system [`AbstractCondensedKKTSystem`](@ref) that also removes
the blocks associated to the inequality constraints. This system
is useful if the problem has many inequality constraints, or more importantly,
[to run MadNLP on the GPU](tutorials/gpu.md). The user can switch the KKT system by using the option
```julia
kkt_system=SparseCondensedKKTSystem
```
Condensing a dense Jacobian row would turn the condensed matrix dense, so rows with
more than `dense_row_abs` nonzeros (default 512) and more than `dense_row_rel` times
the number of variables (default 0.1) are candidates to be kept out of the condensation,
with their multipliers, as a border of the condensed matrix. The densest candidates are
kept, as many as minimize the estimated nonzeros of the factored matrix (a problem whose
rows are all dense is better condensed into one dense block than bordered), and the
border is used only when it saves more than `dense_row_fill_min` nonzeros (default `1e7`;
a single row reaches it at about 4,500 nonzeros, where `nnz²/2` is `1e7`) and when the
linear solver can factor an indefinite matrix. The bordered matrix has a known
inertia, so the inertia test still applies. These thresholds are passed through
`kkt_options`, for example
```julia
kkt_options = Dict(:dense_row_abs => 100, :dense_row_fill_min => 0.0)
```
For certain highly structured problem, it may be beneficial to implement
a [custom KKT system](tutorials/kktsystem.md) to exploit the problem's structure.

#### Inertia-correction

The vector $$(\Delta w, \Delta y, \Delta z)$$ is a descent direction
if the reduced Hessian (the Hessian $$H_k$$ projected on the null-space of the Jacobian
$$J_k$$) of the primal-dual KKT system is **positive definite**.
The reduced Hessian is positive definite if and only if the inertia of the reduced primal-dual KKT system (the number of positive,
zero and negative eigenvalues) is exactly equal to ``(n, 0, m)``.

MadNLP uses an inertia correction mechanism that increases the values of the
primal and dual regularizations $$(\delta_x, \delta_y)$$ in the KKT system until the inertia
of the system is exactly ``(n, 0, m)``. This procedure requires using
an *inertia-revealing* linear solver that returns explicitly the inertia of the system
as an output of the factorization. This procedure can be costly, as it involves
re-factorizing the linear system each time a new regularization is tried.

If the linear solver does not compute the inertia of the KKT system, MadNLP
uses the inertia-free algorithm described [in this article](https://link.springer.com/article/10.1007/s10589-015-9820-y). This procedure is less stringent than the classical inertia-based correction,
in the sense that it doesn't require the reduced Hessian to be positive definite.
The inertia-free correction can be activated explicitly by
using the option `inertia_correction_method=InertiaFree`.

### Step 4: Finding the next iterate using the filter line-search

MadNLP uses the descent direction ``(\Delta w, \Delta y, \Delta z)`` to compute the next iterate as
```math
w_{k+1} = w_k + \alpha_k^p \Delta w \; , \quad
y_{k+1} = y_k + \alpha_k^d \Delta y \; , \quad
z_{k+1} = z_k + \alpha_k^d \Delta z \; .
```
First, the algorithm computes the maximum primal-dual steps ``(\alpha_k^{p,max}, \alpha_k^{d,max})``
satisfying the *fraction-to-boundary* rule:
```math
w_k + \alpha_k^{p,max} \Delta w \geq (1-\tau) w_k  \; , \quad
z_k + \alpha_k^{d,max} \Delta z \geq (1-\tau) z_k  \; .
```
This ensures that the iterates remain positive: ``(w_{k+1}, z_{k+1}) > 0``.
In MadNLP, ``\tau = \max(\tau_{min}, 1-\mu)``, with ``\tau_{min}`` a parameter
defined by the option `tau_min`.

Once the maximum steps computed with the fraction-to-boundary rule, MadNLP finds
the new iterate by using a backtracking line-search by testing different trial
values for $$\alpha_{k,l} = \frac{1}{2^l} \alpha_k^{p,max}$$. Once a new trial
iterate $$w_{k,l} = w_k + \alpha_{k,l} \Delta w$$ achieves sufficient progress,
the line-search stops and MadNLP proceeds to the next iteration.
The progress is measured by a [filter](https://link.springer.com/article/10.1007/s101070100244),
which accepts a new point either if it reduces the value of the barrier
function $$\phi_\mu(w) = f(w) - \mu \sum_{i=1}^n \log(w_i)$$ or it reduces
the constraint violation $$\|c(w)\|_1$$, by comparing these two values with those of a list of past iterates.


#### Feasibility restoration phase

If no acceptable step is found by the line search, MadNLP switches to a restoration
phase that attempts at projecting the current iterate onto the feasible set.
The problem solved by the restoration phase is:
```math
\begin{aligned}
\min_{w} \;& \| c(w) \|_1 \\
\text{subject to} \quad & x \geq 0 \; .
\end{aligned}
```
Once the constraint violation has been significantly reduced, MadNLP returns
to the classical algorithm. If the feasibility restoration phase converges
to a solution of the feasibility restoration problem with a positive objective,
the problem is detected as being locally infeasible.

