## Install
```julia
pkg> add MadNLPHSL
```
Obtain a license and download HSL_jll.jl from [HSL](https://licences.stfc.ac.uk/products/Software/HSL/LibHSL).

Install this download into your current environment using:

```julia
import Pkg
Pkg.develop(path = "/full/path/to/HSL_jll.jl")
```

Alternatively, one can use a custom-compiled HSL library by overriding the `HSL_jll.jl` artifact.
This can be done by editing `~/.julia/artifacts/Overrides.toml`
```
# replace HSL_jll.jl artifact /usr/local/lib/libhsl.so
ecece3e2c69a413a0e935cf52e03a3ad5492e137 = "/usr/local"
```
