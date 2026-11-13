# Deferred reductions on the device (see `MadCore._reduce`). A reduction runs one kernel that
# writes a partial result per workgroup into a column of a buffer; `_values` copies the used
# columns to the host once and folds each column there. A launch and its column cost far less
# than a `mapreduce`, which waits for its result and sets up and frees its scratch every call.

# device vectors, their views, and views of their views (as `MadCore._iview` makes)
const GPUVec{T} = Union{
    AbstractGPUVector{T},
    SubArray{T, 1, <:AbstractGPUArray},
    SubArray{T, 1, <:SubArray{T, 1, <:AbstractGPUArray}},
}

const _RED_GROUPS = 128     # workgroups per reduction, so partial results per column
const _RED_GROUPSIZE = 256
const _RED_SLOTS = 64       # columns, so reductions outstanding at once

@kernel unsafe_indices = true function _block_reduce_kernel!(partials, op, init, bc, n, slot)
    gi = @index(Group, Linear)
    li = @index(Local, Linear)
    shared = @localmem eltype(partials) (_RED_GROUPSIZE,)
    acc = init
    i = (gi - 1) * _RED_GROUPSIZE + li
    while i <= n
        @inbounds acc = op(acc, bc[i])
        i += _RED_GROUPS * _RED_GROUPSIZE
    end
    @inbounds shared[li] = acc
    @synchronize
    s = _RED_GROUPSIZE ÷ 2
    while s > 0
        if li <= s
            @inbounds shared[li] = op(shared[li], shared[li + s])
        end
        @synchronize
        s ÷= 2
    end
    if li == 1
        @inbounds partials[gi, slot] = shared[1]
    end
end

mutable struct ReductionBuffer{T, MT}
    partials::MT                 # _RED_GROUPS × _RED_SLOTS on the device
    host::Matrix{T}
    ops::Vector{Any}             # (op, init) of each issued column
    issued::Int                  # columns written since the last reset
    fetched::Int                 # columns copied to `host`
    released::BitVector          # columns whose value has been read
end

const _RED_BUFFERS = IdDict{Any, Any}()

function _reduction_buffer(::Type{T}, like) where {T}
    backend = get_backend(like)
    return get!(_RED_BUFFERS, (backend, T)) do
        partials = similar(like, T, (_RED_GROUPS, _RED_SLOTS))
        ReductionBuffer{T, typeof(partials)}(partials, zeros(T, _RED_GROUPS, _RED_SLOTS), Any[], 0, 0, falses(_RED_SLOTS))
    end::ReductionBuffer{T}
end

struct DeviceScalar{T, B <: ReductionBuffer{T}} <: MadCore.AbstractDeferredScalar
    buf::B
    slot::Int
end

# the arrays a reduction reads, or the parent array of a view
_storage(A::AbstractGPUArray) = A
_storage(A::SubArray) = _storage(parent(A))

function MadCore._reduce(f::F, op::OP, init, A::GPUVec, As::GPUVec...) where {F, OP}
    T = typeof(init)
    buf = _reduction_buffer(T, _storage(A))
    buf.issued < _RED_SLOTS || error("more than $(_RED_SLOTS) deferred reductions outstanding")
    buf.issued += 1
    slot = buf.issued
    push!(buf.ops, (op, init))
    bc = Base.Broadcast.instantiate(Base.Broadcast.broadcasted(f, A, As...))
    _block_reduce_kernel!(get_backend(buf.partials), _RED_GROUPSIZE)(
        buf.partials, op, init, bc, length(bc), slot; ndrange = _RED_GROUPS * _RED_GROUPSIZE,
    )
    return DeviceScalar(buf, slot)
end

function MadCore._fetch!(x::DeviceScalar)
    buf = x.buf
    if buf.fetched < buf.issued
        # one copy of the columns written since the last fetch; the host waits for the device here
        first = buf.fetched + 1
        copyto!(buf.host, (first - 1) * _RED_GROUPS + 1, buf.partials, (first - 1) * _RED_GROUPS + 1,
            (buf.issued - buf.fetched) * _RED_GROUPS)
        buf.fetched = buf.issued
    end
    return
end

function MadCore._value(x::DeviceScalar)
    op, init = x.buf.ops[x.slot]
    return foldl(op, view(x.buf.host, :, x.slot); init = init)
end

# the buffer is reused once every issued reduction has been read
function MadCore._release!(x::DeviceScalar)
    buf = x.buf
    buf.released[x.slot] = true
    if count(buf.released) >= buf.issued
        buf.issued = buf.fetched = 0
        fill!(buf.released, false)
        empty!(buf.ops)
    end
    return
end

function _reset_reduction_buffers!()
    for buf in values(_RED_BUFFERS)
        buf.issued = buf.fetched = 0
        fill!(buf.released, false)
        empty!(buf.ops)
    end
    return
end

# Elementwise assignments without broadcast's aliasing copies (see `MadCore._assign!`)


@kernel function _assign_kernel!(dest, f, args)
    i = @index(Global, Linear)
    @inbounds dest[i] = f(map(a -> a[i], args)...)
end

@kernel function _assign2_kernel!(dest1, f1, args1, n1, dest2, f2, args2, n2)
    i = @index(Global, Linear)
    if i <= n1
        @inbounds dest1[i] = f1(map(a -> a[i], args1)...)
    end
    if i <= n2
        @inbounds dest2[i] = f2(map(a -> a[i], args2)...)
    end
end

function MadCore._assign!(dest::GPUVec, f::F, args::AbstractArray...) where {F}
    n = length(dest)
    n == 0 && return dest
    all(a -> length(a) == n, args) || throw(DimensionMismatch("_assign!: lengths differ"))
    _assign_kernel!(get_backend(dest))(dest, f, args; ndrange = n)
    return dest
end

function MadCore._assign2!(dest1::GPUVec, f1::F1, args1::Tuple, dest2::GPUVec, f2::F2, args2::Tuple) where {F1, F2}
    n1, n2 = length(dest1), length(dest2)
    all(a -> length(a) == n1, args1) && all(a -> length(a) == n2, args2) ||
        throw(DimensionMismatch("_assign2!: lengths differ"))
    max(n1, n2) == 0 && return
    _assign2_kernel!(get_backend(dest1))(dest1, f1, args1, n1, dest2, f2, args2, n2; ndrange = max(n1, n2))
    return
end
