```@meta
CurrentModule = MadNLP
```

```@docs
MadNLPOptions
```

## Presets

The option `preset` chooses a set of options together for a sparse model: by default
[`ShiftedPreset`](@ref), on the host and on a GPU. Options given explicitly take precedence over
those of the preset, e.g. `madnlp(nlp; barrier = MonotoneUpdate(1e-8, 10.0))` keeps the rest of
the shifted preset with the monotone barrier rule, and `madnlp(nlp; preset = AugmentedPreset())`
selects MadNLP's earlier defaults.

```@docs
AbstractPreset
ShiftedPreset
AugmentedPreset
```
