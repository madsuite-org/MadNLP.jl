#=
    Dense BLAS on Metal

    Metal has no BLAS for these operations. Apple GPUs share memory with the CPU, so the
    dense kernels run on the host BLAS (Accelerate, when AppleAccelerate is loaded) over
    views of the device arrays: wait for the queued GPU work, then wrap the buffers
    without copying. Arrays in private storage are copied to the host, and the output (the
    last argument) is copied back.
=#

_host(A::MtlArray) = is_shared(A) ? unsafe_wrap(Array, A) : Array(A)
_writeback!(A::MtlArray, hA::Array) = (is_shared(A) || copyto!(A, hA); A)

# Run `f` on host views of `args`, writing back the last one, which `f` updates.
function _on_host(f, args::MtlArray...)
    # Shared buffers are read in place, so the GPU work queued on them must be done;
    # `Array` waits by itself for private ones.
    any(is_shared, args) && Metal.synchronize()
    hargs = map(_host, args)
    GC.@preserve args f(hargs...)
    _writeback!(last(args), last(hargs))
    return last(args)
end

for T in (Float32, Float64)
    @eval begin
        MadCore._ger!(alpha::$T, x::MtlVector{$T}, y::MtlVector{$T}, A::MtlMatrix{$T}) =
            _on_host((hx, hy, hA) -> LinearAlgebra.BLAS.ger!(alpha, hx, hy, hA), x, y, A)
        MadCore._syr!(uplo::Char, alpha::$T, x::MtlVector{$T}, A::MtlMatrix{$T}) =
            _on_host((hx, hA) -> LinearAlgebra.BLAS.syr!(uplo, alpha, hx, hA), x, A)
        MadCore._symv!(uplo::Char, alpha::$T, A::MtlMatrix{$T}, x::MtlVector{$T}, beta::$T, y::MtlVector{$T}) =
            _on_host((hA, hx, hy) -> LinearAlgebra.BLAS.symv!(uplo, alpha, hA, hx, beta, hy), A, x, y)
        MadCore._syrk!(uplo::Char, trans::Char, alpha::$T, A::MtlMatrix{$T}, beta::$T, C::MtlMatrix{$T}) =
            _on_host((hA, hC) -> LinearAlgebra.BLAS.syrk!(uplo, trans, alpha, hA, beta, hC), A, C)
        MadCore._trsm!(side::Char, uplo::Char, transa::Char, diag::Char, alpha::$T, A::MtlMatrix{$T}, B::MtlMatrix{$T}) =
            _on_host((hA, hB) -> LinearAlgebra.BLAS.trsm!(side, uplo, transa, diag, alpha, hA, hB), A, B)
    end
end
