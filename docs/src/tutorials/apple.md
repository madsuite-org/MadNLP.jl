# Running MadNLP on Apple silicon

```@meta
CurrentModule = MadNLP
```

On Macs with Apple silicon, MadNLP can use [Apple Accelerate](https://developer.apple.com/documentation/accelerate)
on the CPU and the Apple GPU through Metal. This tutorial shows which solver to pick for each kind of
model. The examples are not run when the documentation is built, because no documentation runner has an Apple GPU.

## Installation

Accelerate comes with macOS; [AppleAccelerate.jl](https://github.com/JuliaLinearAlgebra/AppleAccelerate.jl)
gives Julia access to it:
```julia
pkg> add MadNLP AppleAccelerate
```
For the GPU, also install Metal and mtlMadNLP, which needs Julia 1.13 or later:
```julia
pkg> add Metal https://github.com/madsuite-org/MadNLP.jl:mtlMadNLP
```

## Which BLAS does MadNLP use?

Julia routes BLAS and LAPACK calls through libblastrampoline, which keeps two tables: one for the
64-bit-integer interface (ILP64), used by Julia's `LinearAlgebra` and therefore by [`LapackCPUSolver`](@ref)
and the dense KKT systems, and one for the 32-bit-integer interface (LP64), used by MUMPS. Loading
AppleAccelerate (macOS 13.4 or later) registers Accelerate in both tables, for the whole Julia session.
Each later registration replaces the earlier one, so **the load order matters for MUMPS**:

| Load order | `LapackCPUSolver`, dense KKT systems | `MumpsSolver` |
|------------|--------------------------------------|---------------|
| `using MadNLP` only | OpenBLAS | OpenBLAS |
| `using AppleAccelerate` then `using MadNLP` | Accelerate | OpenBLAS |
| `using MadNLP` then `using AppleAccelerate` | Accelerate | Accelerate |

In the second row, MadNLP loads `OpenBLAS32_jll` for MUMPS, which registers OpenBLAS in the 32-bit table
after Accelerate. To put MUMPS on Accelerate regardless of the load order, register Accelerate again after
loading MadNLP:
```julia
using AppleAccelerate, MadNLP
AppleAccelerate.load_accelerate(; clear = false, load_ilp64 = true)
```
On PGLib AC-OPF, MUMPS ran no faster on Accelerate than on OpenBLAS: for sparse models, use
[`AccelerateSolver`](@ref) instead. You can check which library is in use with `LinearAlgebra.BLAS.get_config()`.
[`AccelerateSolver`](@ref) calls Accelerate directly and does not depend on any of this.

## Sparse models on the CPU: `AccelerateSolver`

Loading AppleAccelerate enables [`AccelerateSolver`](@ref), the sparse LDLᵀ factorization of Accelerate.
Its default threshold partial pivoting reports the inertia, so MadNLP uses it exactly like MUMPS or the HSL solvers:
```julia
using MadNLP, AppleAccelerate, ExaModels

nlp = airport_model(Float64, nothing)   # any model on the CPU; `airport_model` is defined in [Running MadNLP on the GPU](@ref)
results = madnlp(nlp; linear_solver = AccelerateSolver)
```
On PGLib AC-OPF instances with 300 to 10,000 buses, it takes the same number of iterations as
[`MumpsSolver`](@ref) on all but one instance (where they differ by 1%), with a linear-solver time 1.7–2.6× lower on an M-series Mac (e.g. 2.8 s against 6.8 s
on `case9241_pegase`).

With JuMP:
```julia
model = Model(() -> MadNLP.Optimizer(linear_solver = AccelerateSolver))
```

Its options take the enumerations of AppleAccelerate (`import AppleAccelerate as AA`):

| Option | Default | Values |
|--------|---------|--------|
| `accelerate_factorization` | `AA.SparseFactorizationLDLTTPP` | `SparseFactorizationLDLTSBK`, `SparseFactorizationLDLTUnpivoted`, `SparseFactorizationCholesky` |
| `accelerate_order` | `AA.SparseOrderDefault` | `SparseOrderAMD`, `SparseOrderMetis`, `SparseOrderCOLAMD` |
| `accelerate_scaling` | `AA.SparseScalingUser` (no scaling) | `SpraseScalingDefault` (sic), `SparseScalingEquilibriationInf` |
| `accelerate_pivtol` | `1e-4` | initial relative pivot threshold, raised by MadNLP up to `accelerate_pivtolmax` (`0.1`) when a solve is inaccurate |
| `accelerate_zero_tolerance` | `1e-20` | pivots smaller in absolute value are zero |

The defaults were the fastest in our benchmarks: Accelerate's scaling or a smaller pivot threshold make
the solves less accurate and add refinement steps. Keep threshold partial pivoting (TPP) unless you have a
reason not to: the other LDLᵀ variants report no inertia, so MadNLP falls back to its slower inertia-free regularization.

## Dense models on the CPU: `LapackCPUSolver`

For dense models (`callback = MadNLP.DenseCallback` with `DenseKKTSystem` or `DenseCondensedKKTSystem`),
simply load AppleAccelerate before solving: [`LapackCPUSolver`](@ref) then factorizes on Accelerate.
```julia
using MadNLP, AppleAccelerate

results = madnlp(
    nlp;
    callback = MadNLP.DenseCallback,
    kkt_system = MadNLP.DenseCondensedKKTSystem,
    linear_solver = LapackCPUSolver,
)
```
Compared with the default OpenBLAS, Accelerate solved dense QPs with 250 to 2000 variables 1.2–6.4× faster
(e.g. 0.32 s against 2.0 s for the condensed system with a Cholesky factorization at n = 2000).
For problems with inequality constraints, the condensed system with `lapack_algorithm = MadNLP.CHOLESKY`
is usually the fastest; on the augmented `DenseKKTSystem`, keep the default Bunch-Kaufman factorization, which reports the inertia.

## Models on the Apple GPU

Metal has no `Float64`, so models on the GPU must use `Float32`, and may need a looser tolerance `tol`
(e.g. `1e-4`).

### Sparse models

Instantiate the model on the GPU with ExaModels and the Metal backend; mtlMadNLP then solves it with
[`ShiftedPreset`](@ref MadNLPCore.ShiftedPreset): the augmented KKT system with the static LDLᵀ
factorization of [`SDSSolver`](@ref):
```julia
using Metal, mtlMadNLP, ExaModels

nlp = airport_model(Float32, MetalBackend())
results = madnlp(nlp; tol = 1f-4)
```

### Dense models

Metal has no dense factorization, so for dense models mtlMadNLP uses [`LapackCPUSolver`](@ref) by default,
which factorizes the KKT matrix on the CPU while the rest of MadNLP runs on the GPU. Two settings make this fast:

1. Load AppleAccelerate, so that the factorization and the dense BLAS run on Accelerate.
2. Allocate the model in **shared storage**. Apple GPUs share their memory with the CPU, but Metal.jl
   allocates arrays in private (GPU-only) storage by default, which forces a copy of the KKT matrix at every
   factorization. In shared storage, the CPU reads the GPU arrays in place.

```julia
using AppleAccelerate, Metal, mtlMadNLP

x0 = Metal.zeros(Float32, n; storage = Metal.SharedStorage)   # or mtl(x; storage = Metal.SharedStorage)
nlp = build_dense_model(x0)  # every array MadNLP allocates inherits the storage of the model
results = madnlp(nlp; callback = MadNLP.DenseCallback, kkt_system = MadNLP.DenseCondensedKKTSystem, tol = 1f-4)
```
On dense QPs with 250 to 2000 variables, shared storage was 1.1–1.5× faster than private storage.
To make shared storage the default for all Metal arrays, run `Metal.set_default_storage!(Metal.SharedStorage)`
once and restart Julia.

## Summary

| Model | Recommended setup |
|-------|-------------------|
| Sparse, `Float64` | CPU, `using AppleAccelerate`, `linear_solver = AccelerateSolver` |
| Dense, `Float64` | CPU, `using AppleAccelerate`, `LapackCPUSolver` |
| Sparse, `Float32` | GPU, ExaModels with `MetalBackend()`, mtlMadNLP defaults |
| Dense, `Float32` | GPU, `using AppleAccelerate`, model in `Metal.SharedStorage` |
