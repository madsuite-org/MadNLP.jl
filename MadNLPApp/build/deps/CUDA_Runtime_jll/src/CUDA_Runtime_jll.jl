# Use baremodule to shave off a few KB from the serialized `.ji` file
baremodule CUDA_Runtime_jll
using Base
using Base: UUID
using LazyArtifacts
Base.include(@__MODULE__, joinpath("..", ".pkg", "platform_augmentation.jl"))
import JLLWrappers

JLLWrappers.@generate_main_file_header("CUDA_Runtime")
JLLWrappers.@generate_main_file("CUDA_Runtime", Base.UUID("76a88914-d11a-5bdc-97e0-2f5a05c973a2"))
end  # module CUDA_Runtime_jll
