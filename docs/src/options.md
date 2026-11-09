```@meta
CurrentModule = MadNLP
```

```@docs
MadNLPOptions
```

## Presets

The option `preset` chooses a set of options together for a sparse model: by default
[`AugmentedPreset`](@ref) on the host and [`ShiftedPreset`](@ref) on a GPU; [`CondensedPreset`](@ref)
selects the condensed KKT system. The preset chooses the KKT system, unless `kkt_system` is given,
and the options of the linear solver. Options given
explicitly take precedence over those of the preset, e.g. `madnlp(nlp; preset = ShiftedPreset)`
selects the shifted setting on the host (with CHOLMOD's LDLᵀ), and
`madnlp(nlp; preset = ShiftedPreset, barrier = MonotoneUpdate(1e-8, 10.0))` keeps the rest of it
with the monotone barrier rule.

```@docs
AbstractPreset
ShiftedPreset
AugmentedPreset
CondensedPreset
```
