# HS15 with a dense Jacobian and Hessian; the model itself is defined in MadNLPCore.
HS15Model(; kwargs...) = MadNLP.MadNLPCore.HS15Model(; dense = true, kwargs...)
