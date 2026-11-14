# Deferred reductions and elementwise assignments on device arrays, against their host
# results. `Arr` converts a host vector to the device array type under test.
function test_deferred_reductions(Arr)
    MadCore = MadNLP.MadCore
    n = 1000
    a, b, c = rand(n), rand(n) .- 0.5, rand(n)
    ind = collect(1:3:n)
    da, db, dc, dind = Arr(a), Arr(b), Arr(c), Arr(ind)

    @testset "_reduce and _values" begin
        r1 = MadCore._reduce((x, y) -> abs(x - y), max, 0.0, da, db)
        r2 = MadCore._reduce(abs, +, 0.0, dc)
        r3 = MadCore._reduce(abs, +, 0.0, MadCore._iview(dc, dind))     # view through an index vector
        r4 = MadCore._reduce(abs, +, 0.0, MadCore._iview(view(dc, 2:n), dind[1:(end - 1)]))  # view of a view
        r5 = MadCore._reduce(abs, max, -1.0, view(da, 1:0))              # empty: the initial value
        lz = MadCore._lazy((x, y) -> x / y, r1, r2)                       # a deferred combination
        v = MadCore._values(r1, r2, r3, r4, r5, lz, 7.0)
        @test v[1] ≈ maximum(abs.(a .- b))
        @test v[2] ≈ sum(abs, c)
        @test v[3] ≈ sum(abs, c[ind])
        @test v[4] ≈ sum(abs, view(c, 2:n)[ind[1:(end - 1)]])
        @test v[5] == -1.0
        @test v[6] ≈ maximum(abs.(a .- b)) / sum(abs, c)
        @test v[7] == 7.0
        # every slot was read, so the next batch starts on an empty buffer
        @test MadCore._value1(MadCore._reduce(identity, +, 0.0, da)) ≈ sum(a)
    end

    return @testset "_assign! and _assign2!" begin
        # destination and sources are views of one vector, through an index vector
        x = rand(2n); dx = Arr(x)
        dst, src = MadCore._iview(dx, Arr(collect(1:n))), view(dx, (n + 1):(2n))
        MadCore._assign!(dst, (d, s, y) -> d - s / y, dst, src, dc)
        @test Array(dx)[1:n] ≈ x[1:n] .- x[(n + 1):(2n)] ./ c
        @test Array(dx)[(n + 1):(2n)] == x[(n + 1):(2n)]
        # two independent assignments of different lengths in one kernel
        y1, y2 = Arr(zeros(n)), Arr(zeros(n ÷ 2))
        MadCore._assign2!(y1, (p, q) -> p * q, (da, db), y2, p -> 2p, (view(dc, 1:(n ÷ 2)),))
        @test Array(y1) ≈ a .* b
        @test Array(y2) ≈ 2 .* c[1:(n ÷ 2)]
    end
end

if HAS_CUDA && CUDA.functional()
    @testset "Deferred reductions (CUDA)" test_deferred_reductions(CuArray)
end
if HAS_AMDGPU && AMDGPU.functional()
    @testset "Deferred reductions (ROCm)" test_deferred_reductions(ROCArray)
end
