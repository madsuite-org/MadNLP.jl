# Use baremodule to shave off a few KB from the serialized `.ji` file
baremodule CUDSS_jll
using Base
using Base: UUID
using LazyArtifacts
Base.include(@__MODULE__, joinpath("..", ".pkg", "platform_augmentation.jl"))
import JLLWrappers

JLLWrappers.@generate_main_file_header("CUDSS")
JLLWrappers.@generate_main_file("CUDSS", Base.UUID("4889d778-9329-5762-9fec-0578a5d30366"))
end  # module CUDSS_jll
