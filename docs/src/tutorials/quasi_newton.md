# Quasi-Newton methods

```@meta
CurrentModule = MadNLP
```
```@setup lbfgs
using NLPModels

```

Sometimes, the second-order derivatives are just too expensive to evaluate.
In that case, it is often a good idea to approximate the Hessian matrix using a Quasi-Newton algorithm.
The BFGS algorithm uses the first-order derivatives (gradient and tranposed-Jacobian
vector product) to approximate the Hessian of the Lagrangian. LBFGS is a variant of BFGS
that computes a low-rank approximation of the Hessian matrix.
By doing so, LBFGS does not have to store a large dense matrix in memory, rendering
the algorithm appropriate in the large-scale regime.

MadNLP implements several quasi-Newton approximation for the Hessian matrix.

## How to use LBFGS in MadNLP?

First, we focus on the limited-memory version of the BFGS algorithm,
commonly known as LBFGS. We refer to [this article](https://epubs.siam.org/doi/abs/10.1137/0916069) for a detailed description of the method.

We look at the `elec` optimization problem from the [COPS benchmark](https://www.mcs.anl.gov/~more/cops/):

```@example lbfgs
using Random
using ExaModels

function elec_model(np)
    Random.seed!(1)
    # Set the starting point to a quasi-uniform distribution of electrons on a unit sphere
    theta = (2pi) .* rand(np)
    phi = pi .* rand(np)

    core = ExaModels.ExaCore(Float64)
    ExaModels.@add_var(core, x, 1:np; start = [cos(theta[i])*sin(phi[i]) for i=1:np])
    ExaModels.@add_var(core, y, 1:np; start = [sin(theta[i])*sin(phi[i]) for i=1:np])
    ExaModels.@add_var(core, z, 1:np; start = [cos(phi[i]) for i=1:np])
    # Coulomb potential
    itr = [(i,j) for i in 1:np-1 for j in i+1:np]
    ExaModels.@add_obj(core, 1.0 / sqrt((x[i] - x[j])^2 + (y[i] - y[j])^2 + (z[i] - z[j])^2) for (i,j) in itr)
    # Unit-ball
    ExaModels.@add_con(core, x[i]^2 + y[i]^2 + z[i]^2 - 1 for i=1:np)

    return ExaModels.ExaModel(core)
end

```

The problem computes the positions of the electrons in an atom.
The potential of each electron depends on the positions of all the other electrons,
coupling together all the variables in the problem and resulting in a dense Hessian matrix.
Hence, the problem is good candidate for a quasi-Newton algorithm.

We start by solving the problem with the default options in MadNLP,
using a dense linear solver from LAPACK:

```@example lbfgs
using MadNLP
nh = 10
nlp = elec_model(nh)
results_hess = madnlp(
    nlp;
    linear_solver=LapackCPUSolver,
)
nothing

```
We observe that MadNLP converges in 21 iterations.

To replace the second-order derivatives by an LBFGS approximation,
you should pass the option `hessian_approximation=CompactLBFGS` to MadNLP.

```@example lbfgs
results_qn = madnlp(
    nlp;
    linear_solver=LapackCPUSolver,
    hessian_approximation=MadNLP.CompactLBFGS,
)
nothing

```

We observe that MadNLP converges in 93 iterations, but the solution time has
decreased by an order of magnitude. As expected, the number of Hessian evaluations is 0.

### How to tune the options in LBFGS?

You can tune the LBFGS options by using the option `quasi_newton_options`.
The option takes as input a `QuasiNewtonOptions` structure, with the following attributes
(we put on the right their default values):
- `init_strategy::BFGSInitStrategy = SCALAR1`
- `max_history::Int = 6`
- `init_value::Float64 = 1.0`
- `sigma_min::Float64 = 1e-8`
- `sigma_max::Float64 = 1e+8`

The most important parameter is `max_history`, which encodes the number of vectors used in the low-rank
approximation. For instance, we can increase the history to use the 20 past iterates using:

```@example lbfgs
qn_options = MadNLP.QuasiNewtonOptions(; max_history=20)
results_qn = madnlp(
    nlp;
    linear_solver=LapackCPUSolver,
    hessian_approximation=MadNLP.CompactLBFGS,
    quasi_newton_options=qn_options,
)
nothing

```

We observe that the total number of iterations has been reduced from 93 to 60.


### How is LBFGS implemented in MadNLP?

MadNLP implements the compact LBFGS algorithm described [in this article](https://link.springer.com/article/10.1007/bf01582063). At each iteration, the Hessian ``W_k`` is approximated by a
low rank positive definite matrix ``B_k``, defined as
```math
B_k = \xi_k I + U_k V_k^\top

```
with ``\xi > 0`` a scaling factor, ``U_k`` and ``V_k`` two ``n \times 2p`` matrices.
The number ``p`` denotes the number of vectors used when computing the limited memory updates
(the parameter `max_history` in MadNLP): the larger, the more accurate is the low-rank approximation.

Replacing the Hessian of the Lagrangian ``W_k`` by the low-rank matrix ``B_k``,
the KKT system solved in MadNLP rewrites as
```math
\begin{bmatrix}
\xi I + U V^\top + \Sigma_x & 0 & A^\top \\
0 & \Sigma_s & -I \\
A & -I & 0
\end{bmatrix}
\begin{bmatrix}
\Delta x \\ \Delta s \\ \Delta y
\end{bmatrix}
=
\begin{bmatrix}
r_1 \\ r_2 \\ r_3
\end{bmatrix}

```
The resulting KKT system has a low-rank structure  we can leverage using the Sherman-Morrison formula.
The method is detailed e.g. in Section 3.8 of [that paper](https://link.springer.com/article/10.1007/s10107-004-0560-5).


!!! info
    As MadNLP is designed to solve generic constrained optimization problems,
    it does not approximate directly the inverse of the Hessian matrix, as done
    in classical LBFGS implementations specialized
    on solving nonlinear problems with bound constraints.
    If your problem has no generic nonlinear constraints, we recommend for optimal performance
    using [LBFGSB](https://dl.acm.org/doi/pdf/10.1145/279232.279236)
    or the LBFGS implemented in [JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl/blob/main/src/lbfgs.jl).


## How to use BFGS in MadNLP?

On the contrary to LBFGS, the algorithm BFGS stores the matrix approximating
the Hessian in a dense matrix with size ``(n, n)``, where ``n`` is the number of variables
in the problem. Hence, we recommend using LBFGS whenever ``n`` becomes greater than 1,000.

That being said, BFGS can be practical to solve medium-scaled instances, as it provides
a better approximation than the LBFGS method. In particular, this is the case for the `elec`
optimization problem. As all the matrices involved are dense, BFGS works only if MadNLP
is storing its KKT system in dense format. We have to specify that explicitly
by passing to MadNLP the option `kkt_system=MadNLP.DenseKKTSystem`. Apart of that detail, solving the
problem with BFGS just amounts to


```@example lbfgs
results_qn = madnlp(
    nlp;
    linear_solver=LapackCPUSolver,
    kkt_system=MadNLP.DenseKKTSystem,
    hessian_approximation=MadNLP.BFGS,
)
nothing
```
Observe that MadNLP here converges in 62 iterations.
Indeed, BFGS computes a positive definite approximation of the Hessian matrix.

If the problem is non-convex, it is usually recommended to use the damped version of the BFGS algorithm, which
can be activated by replacing `MadNLP.BFGS` by `MadNLP.DampedBFGS` in the precedent call:
```@example lbfgs
results_qn = madnlp(
    nlp;
    linear_solver=LapackCPUSolver,
    kkt_system=MadNLP.DenseKKTSystem,
    hessian_approximation=MadNLP.DampedBFGS,
)
nothing
```


