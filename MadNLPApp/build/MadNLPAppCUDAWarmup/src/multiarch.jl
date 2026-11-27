# Multi-architecture GPU kernels for libmadnlp (build/libmadnlp.jl): every kernel the
# warm-up compiled for the build machine's GPU is compiled for other compute capabilities as
# well, so that a GPU of those finds its kernel in the image instead of compiling it at run
# time. (From the earlier madnlp executable, after empower's gpu/kernels.)


# The compiler config CUDA.jl would build on a device of compute capability `cap`, from
# one it built for the current device: same kernel options, retargeted architecture.
function retarget(cfg::GPUCompiler.CompilerConfig, cap::VersionNumber)
    ptx = cfg.params.ptx
    sm = CUDACore.select_ptxas_sm(CUDACore.ptx_sm_support(ptx), cap)
    llvm_sm = CUDACore.select_llvm_sm(CUDACore.llvm_compat().sm, sm)
    t = cfg.target
    target = GPUCompiler.PTXCompilerTarget(;
        cap = CUDACore.base_version(llvm_sm), ptx = t.ptx,
        feature_set = llvm_sm.feature_set, debuginfo = t.debuginfo, minthreads = t.minthreads,
        maxthreads = t.maxthreads, blocks_per_sm = t.blocks_per_sm, maxregs = t.maxregs,
        fastmath = t.fastmath
    )
    params = CUDACore.CUDACompilerParams(; sm, ptx)
    return GPUCompiler.CompilerConfig(cfg; target, params)
end

const CompilerCaching = GPUCompiler.CompilerCaching

# Every (MethodInstance, config) whose CUDA results carry a compiled image.
function compiled_kernels()
    found = Tuple{Core.MethodInstance, GPUCompiler.CompilerConfig, Core.CodeInstance}[]
    Base.visit(Core.methodtable) do m
        specs = m.specializations
        for mi in (specs isa Core.MethodInstance ? (specs,) : specs)
            mi isa Core.MethodInstance || continue
            isdefined(mi, :cache) || continue
            ci = mi.cache
            while true
                if ci.owner isa GPUCompiler.GPUCompilerCacheToken && ci.owner.target isa GPUCompiler.PTXCompilerTarget
                    chain = CompilerCaching.read_analysis_results(ci)
                    while chain isa Core.Compiler.AnalysisResults
                        r = chain.result
                        r isa CompilerCaching.CachedResult && (r = getfield(r, 1))
                        if r isa GPUCompiler.PersistentJobResults
                            for e in r.entries
                                e.value isa CUDACoreResults && e.value.image !== nothing && push!(found, (mi, e.config, ci))
                            end
                        end
                        chain = isdefined(chain, :next) ? chain.next : nothing
                    end
                end
                isdefined(ci, :next) || break
                ci = ci.next
            end
        end
    end
    return found
end
const CUDACoreResults = CUDACore.CUDACompilerResults

# Every compiled image in the loaded package images, with the architecture it is for,
# so that the executable can load the ones for its GPU when it starts.
function compiled_results()
    out = Tuple{CUDACoreResults, CUDACore.SMVersion}[]
    seen = IdDict{Any, Nothing}()
    for (mi, cfg, ci) in compiled_kernels()
        chain = CompilerCaching.read_analysis_results(ci)
        while chain isa Core.Compiler.AnalysisResults
            r = chain.result
            r isa CompilerCaching.CachedResult && (r = getfield(r, 1))
            if r isa GPUCompiler.PersistentJobResults
                for e in r.entries
                    v = e.value
                    if v isa CUDACoreResults && v.image !== nothing && !haskey(seen, v)
                        seen[v] = nothing
                        push!(out, (v, e.config.params.sm))
                    end
                end
            end
            chain = isdefined(chain, :next) ? chain.next : nothing
        end
    end
    return out
end

# Compile every kernel already compiled for the current device for each capability in
# `caps` as well, with the configuration CUDA.jl builds on such a device, so a GPU of that
# capability finds its cubin in the cache instead of compiling at run time.
function compile_for(caps)
    # The architecture the package images were compiled on: no device is needed (nor,
    # inside juliac's build process, available) to compile for another one.
    all = compiled_kernels()
    isempty(all) && error("no compiled CUDA kernels found in the loaded package images")
    sms = [k[2].params.sm for k in all]
    built = argmax(sm -> count(==(sm), sms), unique(sms))
    ks = filter(k -> k[2].params.sm == built, all)
    println("build architecture ", built, ": ", length(ks), " of ", length(all), " cached kernels")
    n = 0
    failed = 0
    done = VersionNumber[]
    # in this process's world: a kernel cached by another package's image carries the world
    # range it was compiled in, which a job here cannot use
    world = Base.get_world_counter()
    for cap in caps
        # A CUDA's ptxas drops old architectures (CUDA 13 has no Volta, sm_70); the PTX
        # ISA still lists them, so the first kernel is the test.
        supported = true
        for (k, (mi, cfg, ci)) in enumerate(ks)
            cfg2 = retarget(cfg, cap)
            cfg2 == cfg && continue
            res = try
                # CUDA.jl expects its compile lock held (it releases it while ptxas runs).
                lock(CUDACore.cufunction_lock) do
                    CUDACore.compile_or_lookup(GPUCompiler.CompilerJob(mi, cfg2, world))
                end
            catch err
                if k == 1 && occursin("is not defined for option 'gpu-name'", sprint(showerror, err))
                    println("skipping compute capability ", cap, ": this CUDA's ptxas does not support it")
                    supported = false
                    break
                end
                failed += 1
                continue
            end
            res.image === nothing || (n += 1)
        end
        supported && push!(done, cap)
    end
    println("compiled ", length(ks), " kernels for ", join(done, ", "), ": ", n, " images, ", failed, " failed")
    mark = get(ENV, "MADNLPAPP_MULTIARCH_LOG", "")
    isempty(mark) || write(mark, string("kernels ", length(ks), " caps ", join(done, ","), " images ", n, " failed ", failed, "\n"))
    return n
end
