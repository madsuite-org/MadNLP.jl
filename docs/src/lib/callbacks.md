```@meta
CurrentModule = MadNLP
```

# Callbacks

In MadNLP, a nonlinear program is implemented with a given `AbstractNLPModel`.
The model may have a form unsuitable for the interior-point algorithm.
For that reason, MadNLP wraps the `AbstractNLPModel` internally
using custom data structures, encoded as a `AbstractCallback`.
Depending on the setting, MadNLP chooses to wrap the `AbstractNLPModel`
as a [`DenseCallback`](@ref) or alternatively, as a [`SparseCallback`](@ref).

```@docs
AbstractCallback
DenseCallback
SparseCallback

```

The function [`create_callback`](@ref) allows to instantiate a `AbstractCallback`
from a given `NLPModel`:
```@docs
create_callback

```

Internally, a [`AbstractCallback`](@ref) reformulates the inequality
constraints as equality constraints by introducing additional slack variables.
The fixed variables are reformulated as parameters (using [`MakeParameter`](@ref))
or are relaxed (using [`RelaxBound`](@ref)). The equality constraints can
be kept as is with [`EnforceEquality`](@ref) (default option) or relaxed
as inequality constraints with [`RelaxEquality`](@ref). In that latter case,
MadNLP solves a relaxation of the original problem.

```@docs
AbstractFixedVariableTreatment
MakeParameter
RelaxBound

AbstractEqualityTreatment
EnforceEquality
RelaxEquality
```

MadNLP has to keep in memory all the indexes associated to the equality
and inequality constraints, as well as the indexes of the bounded variables and the fixed variables.
The indexes are stored explicitly as a `Vector{Int}` in the `AbstractCallback` structure used by MadNLP.
