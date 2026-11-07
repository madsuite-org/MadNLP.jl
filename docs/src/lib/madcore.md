```@meta
CurrentModule = MadCore
```

# MadCore public API

MadCore provides the KKT systems, the callbacks and the linear solvers on which the
interior-point solvers of MadSuite are built. Its public API is what a solver or a linear
solver built on MadCore needs: the exported names and those returned by
[`public_names`](@ref). Other names are internal and can change in any release.

```@docs
public_names
```

The manipulation of the KKT systems and their right-hand sides is described in
[KKT systems](kkt.md), and the wrappers around the model in
[Callback wrappers](callbacks.md). The functions below complete these pages.

## KKT systems

```@docs
AbstractDenseKKTSystem
get_slack_regularization
get_sparse_condensed_ext
build_hessian_structure
build_condensed_aug_symbolic
build_condensed_aug_coord!
mul_hess_blk!
kktmul!
madnlp_unsafe_wrap
diag_add!
getptr
nzval
```

## KKT vectors

```@docs
variable
slack
```

## Callbacks

The following functions act on the callback only, and return values scaled by the callback.

```@docs
eval_f_callback
eval_cons_callback!
eval_grad_f_callback!
eval_jac_callback!
eval_lag_hess_callback!
jac_sparsity_callback!
hess_sparsity_callback!
```

The following functions take the solver, record the evaluation in its counters, and check
the first evaluation for invalid numbers.

```@docs
eval_f_wrapper
eval_grad_f_wrapper!
eval_cons_wrapper!
eval_jac_wrapper!
eval_lag_hess_wrapper!
```

Scaling, and conversion of a solution back to the original model:

```@docs
set_scaling!
set_obj_scale!
set_con_scale_sparse!
set_jac_scale_sparse!
unpack_x!
unpack_y!
unpack_z!
unpack_cons!
update_z!
```

Other functions:

```@docs
create_array
create_sparse_fixed_handler
n_variables
n_constraints
```
