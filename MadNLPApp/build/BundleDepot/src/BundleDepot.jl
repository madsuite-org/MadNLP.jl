"""
    BundleDepot

The first package of libmadnlp's image: when the library starts, its runtime looks for
artifacts only in the bundle's own depot (share/julia), not in a Julia installation's
(~/.julia, JULIA_DEPOT_PATH). A user's depot may hold the same artifacts by hash; those carry
the standard library names, not the private ones `cnlp install` gave the bundle's copies, so
loading them would tie this runtime to libraries another runtime in the process uses. Its
`__init__` runs before any other package's, since it is loaded first (build/libmadnlp.jl).
MADNLPAPP_DEPOT_TRACE=1 prints the depot path before and after.
"""
module BundleDepot

function __init__()
    ccall(:jl_generating_output, Cint, ()) == 0 || return   # not while the image is built
    trace = get(ENV, "MADNLPAPP_DEPOT_TRACE", "0") == "1"
    trace && println(stderr, "madnlp: depot path ", DEPOT_PATH)
    root = normpath(joinpath(Sys.BINDIR, ".."))
    keep = filter(d -> startswith(normpath(abspath(d)), root), DEPOT_PATH)
    isempty(keep) || (empty!(DEPOT_PATH); append!(DEPOT_PATH, keep))
    trace && println(stderr, "madnlp: depot path (bundle only) ", DEPOT_PATH)
    return nothing
end

end # module
