module CUDSS

using CUDACore, cuSPARSE
using GPUToolbox
using CUDSS_jll
using LinearAlgebra
using SparseArrays

if CUDACore.local_toolkit
    using CUDA_Runtime_Discovery
else
    import CUDSS_jll
end

# Supported cuDSS versions: [lo, hi)
const CUDSS_SUPPORTED_VERSIONS = (v"1.0.0", v"2")

# Calling `version()` dlopens libcudss. cuDSS 1.x also needs libnvrtc and libcublas from
# the same CUDA major version; they are resolved against the CUDA runtime already loaded
# by CUDA.jl. Checking here gives a clear error at `using CUDSS` instead of at the first
# solver call, and catches version skew between CUDSS.jl and the library.
function check_cudss_version(path)
  global libcudss
  cudss_version = try
    version()
  catch e
    libcudss = nothing
    error("cuDSS at $(path) could not be loaded: $(sprint(showerror, e)). " *
          "cuDSS 1.x also requires libnvrtc and libcublas from the same CUDA major version.")
  end
  lo, hi = CUDSS_SUPPORTED_VERSIONS
  if !(lo <= cudss_version < hi)
    libcudss = nothing # Reset the library to nothing because we are failing out (just in case)
    error("cuDSS found: $(path) is version $(cudss_version) but CUDSS.jl expects version $(lo.major).*")
  end
  return cudss_version
end

function __init__()
  if CUDACore.functional()
    global libcudss
    global CUDSS_INSTALLATION
    if haskey(ENV, "JULIA_CUDSS_LIBRARY_PATH") && Sys.islinux()
      libcudss = joinpath(ENV["JULIA_CUDSS_LIBRARY_PATH"], "libcudss.so")
      CUDSS_INSTALLATION = "CUSTOM"
      check_cudss_version(libcudss)
    elseif CUDACore.local_toolkit
      dirs = CUDA_Runtime_Discovery.find_toolkit()
      path = CUDA_Runtime_Discovery.get_library(dirs, "cudss"; optional=true)
      (path === nothing) && error("cuDSS is not available on your system (looked in $(join(dirs, ", "))).")
      libcudss = path
      CUDSS_INSTALLATION = "LOCAL"
      check_cudss_version(path)
    else
      !CUDSS_jll.is_available() && error("cuDSS is not available for your platform.")
      # MadNLPApp: the CUDA libraries are a separate download; without them CUDSS_jll
      # loads nothing, and cuDSS stays unloaded (a GPU solve is refused before it is used)
      isempty(CUDSS_jll.libcudss) && return
      libcudss = CUDSS_jll.libcudss
      CUDSS_INSTALLATION = "YGGDRASIL"
      check_cudss_version(libcudss)
    end
  end
end

using CUDACore: libraryPropertyType, cudaDataType, initialize_context, retry_reclaim, CUstream, unsafe_free!, HandleCache
import LinearAlgebra: lu, lu!, ldlt, ldlt!, cholesky, cholesky!, ldiv!, BlasFloat, BlasReal, checksquare, Factorization
import Base: \

abstract type AbstractCudssMatrix{T,INT} end
abstract type AbstractCudssSolver{T,INT} <: Factorization{T} end
const CudssInt = Union{Cint, Int64}

include("libcudss.jl")
include("error.jl")
include("types.jl")
include("helpers.jl")
include("management.jl")
include("interfaces.jl")
include("generic.jl")

end # module CUDSS
