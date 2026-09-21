## Install
```julia
pkg> add MadNLPPardiso
```

## Build

To use Pardiso, the user needs to obtain the Pardiso shared libraries from [Panua](),
and place the license file in the home directory.
Then, the path to the shared library is specified via:
```
julia> ENV["JULIA_PARDISO"] = "/path/to/pardiso/lib/"
```
After obtaining the library and the license file, run
```julia
pkg> build MadNLPPardiso
```

