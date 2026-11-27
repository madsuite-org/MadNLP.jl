using Base.BinaryPlatforms

try
    using CUDA_Runtime_jll
catch
    # during initial package installation, CUDA_Runtime_jll may not be available.
    # in that case, we just won't select an artifact.
end

# can't use Preferences for the same reason
const CUDA_Runtime_jll_uuid = Base.UUID("76a88914-d11a-5bdc-97e0-2f5a05c973a2")
const preferences = Base.get_preferences(CUDA_Runtime_jll_uuid)
Base.record_compiletime_preference(CUDA_Runtime_jll_uuid, "version")
Base.record_compiletime_preference(CUDA_Runtime_jll_uuid, "local")
# the `local` preference is a boolean, but CUDA.jl's `set_runtime_version!` writes it
# as a string, and hand-written LocalPreferences.toml files use either spelling.
const local_toolkit = let pref = get(preferences, "local", false)
    pref isa Bool ? pref : something(tryparse(Bool, pref), false)
end

function cuda_comparison_strategy(_a::String, _b::String, a_requested::Bool, b_requested::Bool)
    # if we're using a local toolkit, we can't use artifacts
    if local_toolkit
        return false
    end

    # if either isn't a version number (e.g. "none"), perform a simple equality check
    a = tryparse(VersionNumber, _a)
    b = tryparse(VersionNumber, _b)
    if a === nothing || b === nothing
        return _a == _b
    end

    # if both b and a requested, then we fall back to equality
    if a_requested && b_requested
        return Base.thisminor(a) == Base.thisminor(b)
    end

    # otherwise, do the comparison between the the single version cap and the single version:
    function is_compatible(artifact::VersionNumber, host::VersionNumber)
        if host >= v"11.0"
            # enhanced compatibility, semver-style
            artifact.major == host.major &&
            Base.thisminor(artifact) <= Base.thisminor(host)
        else
            Base.thisminor(artifact) == Base.thisminor(host)
        end
    end
    if a_requested
        is_compatible(b, a)
    else
        is_compatible(a, b)
    end
end

function augment_platform!(platform::Platform)
    if !@isdefined(CUDA_Runtime_jll)
        # don't set to nothing or Pkg will download any artifact
        platform["cuda"] = "none"
    end

    if !haskey(platform, "cuda")
        CUDA_Runtime_jll.augment_platform!(platform)
    end
    BinaryPlatforms.set_compare_strategy!(platform, "cuda", cuda_comparison_strategy)

    return platform
end