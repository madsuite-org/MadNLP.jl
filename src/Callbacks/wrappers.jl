abstract type AbstractWrapperModel{T,VT} <: NLPModels.AbstractNLPModel{T,VT} end

struct DenseWrapperModel{T,VT,T2,VT2,MT2, I <: NLPModels.AbstractNLPModel{T2,VT2}} <: AbstractWrapperModel{T,VT}
    inner::I
    x::VT2
    y::VT2
    con::VT2
    grad::VT2
    jac::MT2
    hess::MT2
    meta::NLPModels.NLPModelMeta{T, VT}
    counters::NLPModels.Counters
end

struct SparseWrapperModel{T,VT,T2,VI2,VT2,I <: NLPModels.AbstractNLPModel{T2,VT2}} <: AbstractWrapperModel{T,VT}
    inner::I
    jrows::VI2
    jcols::VI2
    hrows::VI2
    hcols::VI2
    x::VT2
    y::VT2
    con::VT2
    grad::VT2
    jac::VT2
    hess::VT2
    meta::NLPModels.NLPModelMeta{T, VT}
    counters::NLPModels.Counters
end

"""
    DenseWrapperModel(Arr, m)

Construct a DenseWrapperModel (a subtype of `NLPModels.AbstractNLPModel{T,typeof(Arr(m.meta.x0))}`)
from a generic NLP Model.

DenseWrapperModel can be used to interface GPU-accelerated NLP models with solvers running on CPUs.
"""
function DenseWrapperModel(Arr, m::NLPModels.AbstractNLPModel)
    return DenseWrapperModel(
        m,
        similar(get_x0(m), m.meta.nvar),
        similar(get_x0(m), get_ncon(m)),
        similar(get_x0(m), get_ncon(m)),
        similar(get_x0(m), get_nvar(m)),
        similar(get_x0(m), get_ncon(m), get_nvar(m)),
        similar(get_x0(m), get_nvar(m), get_nvar(m)),
        NLPModels.NLPModelMeta(
            get_nvar(m),
            x0 = Arr(get_x0(m)),
            lvar = Arr(get_lvar(m)),
            uvar = Arr(get_uvar(m)),
            ncon = get_ncon(m),
            y0 = Arr(get_y0(m)),
            lcon = Arr(get_lcon(m)),
            ucon = Arr(get_ucon(m)),
            nnzj = get_nnzj(m),
            nnzh = get_nnzh(m),
            sparse_jacobian = false,
            sparse_hessian = false,
            minimize = get_minimize(m)
        ),
        NLPModels.Counters()
    )
end

"""
    SparseWrapperModel(Arr, m)

Construct a SparseWrapperModel (a subtype of `NLPModels.AbstractNLPModel{T,typeof(Arr(m.meta.x0))}`)
from a generic NLP Model.

SparseWrapperModel can be used to interface GPU-accelerated NLP models with solvers running on CPUs.
"""
function SparseWrapperModel(Arr, m::NLPModels.AbstractNLPModel)
    return SparseWrapperModel(
        m,
        similar(get_x0(m), Int, get_nnzj(m)),
        similar(get_x0(m), Int, get_nnzj(m)),
        similar(get_x0(m), Int, get_nnzh(m)),
        similar(get_x0(m), Int, get_nnzh(m)),
        similar(get_x0(m), get_nvar(m)),
        similar(get_x0(m), get_ncon(m)),
        similar(get_x0(m), get_ncon(m)),
        similar(get_x0(m), get_nvar(m)),
        similar(get_x0(m), get_nnzj(m)),
        similar(get_x0(m), get_nnzh(m)),
        NLPModels.NLPModelMeta(
            get_nvar(m),
            x0 = Arr(get_x0(m)),
            lvar = Arr(get_lvar(m)),
            uvar = Arr(get_uvar(m)),
            ncon = get_ncon(m),
            y0 = Arr(get_y0(m)),
            lcon = Arr(get_lcon(m)),
            ucon = Arr(get_ucon(m)),
            nnzj = get_nnzj(m),
            nnzh = get_nnzh(m),
            sparse_jacobian = true,
            sparse_hessian = true,
            minimize = get_minimize(m)
        ),
        NLPModels.Counters()
    )
end

function NLPModels.obj(
    m::M,
    x::V
) where {M <: AbstractWrapperModel, V <: AbstractVector}
    copyto!(m.x, x)
    return NLPModels.obj(m.inner, m.x)
end

function NLPModels.cons!(
    m::M,
    x::V,
    g::V
) where {M <: AbstractWrapperModel, V <: AbstractVector}
    copyto!(m.x, x)
    NLPModels.cons!(m.inner, m.x, m.con)
    copyto!(g, m.con)
    return g
end

function NLPModels.grad!(
    m::M,
    x::V,
    f::V
) where {M <: AbstractWrapperModel, V <: AbstractVector}
    copyto!(m.x, x)
    NLPModels.grad!(m.inner, m.x, m.grad)
    copyto!(f, m.grad)
    return f
end

function NLPModels.jac_structure!(
    m::M,
    rows::V,
    cols::V
) where {M <: SparseWrapperModel, V <: AbstractVector}
    NLPModels.jac_structure!(m.inner, m.jrows, m.jcols)
    copyto!(rows, m.jrows)
    copyto!(cols, m.jcols)
    return rows, cols
end

function NLPModels.hess_structure!(
    m::M,
    rows::V,
    cols::V
) where {M <: SparseWrapperModel, V <: AbstractVector}
    NLPModels.hess_structure!(m.inner, m.hrows, m.hcols)
    copyto!(rows, m.hrows)
    copyto!(cols, m.hcols)
    return rows, cols
end

function NLPModels.jtprod!(
    m::SparseWrapperModel,
    x::AbstractVector,
    v::AbstractVector,
    jtv::AbstractVector,
)
    copyto!(m.x, x)
    copyto!(m.grad, jtv)
    copyto!(m.con, v)
    NLPModels.jtprod!(m.inner, m.x, m.con, m.grad)
    copyto!(jtv, m.grad)
    return jtv
end

function NLPModels.jac_coord!(
    m::SparseWrapperModel,
    x::AbstractVector,
    jac::AbstractVector,
)
    copyto!(m.x, x)
    NLPModels.jac_coord!(m.inner, m.x, m.jac)
    copyto!(jac, m.jac)
    return jac
end

function NLPModels.hess_coord!(
    m::M,
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractVector;
    obj_weight = one(eltype(x))
) where {M <: SparseWrapperModel}
    copyto!(m.x, x)
    copyto!(m.y, y)
    NLPModels.hess_coord!(m.inner, m.x, m.y, m.hess; obj_weight=obj_weight)
    copyto!(hess, m.hess)
    return hess
end

function NLPModels.jac_dense!(
    m::Model,
    x::V,
    jac::M
) where {Model <: DenseWrapperModel, V <: AbstractVector, M <: AbstractMatrix}
    copyto!(m.x, x)
    NLPModels.jac_dense!(m.inner, m.x, m.jac)
    copyto!(jac, m.jac)
    return jac
end

function NLPModels.hess_dense!(
    m::Model,
    x::V,
    y::V,
    hess::M;
    obj_weight = one(eltype(x))
) where {Model <: DenseWrapperModel, V <: AbstractVector, M <: AbstractMatrix}
    copyto!(m.x, x)
    copyto!(m.y, y)
    NLPModels.hess_dense!(m.inner, m.x, m.y, m.hess; obj_weight=obj_weight)
    copyto!(hess, m.hess)
    return hess
end

"""
    ModelBarrier(m)

Wrap `m` behind a field of abstract type. Code specialized on the model type
(`MadNLPSolver`, the callbacks, `solve!`) is then compiled once per `(T, VT)` instead of
once per model type, and only the model's own evaluation functions compile for a new
model. Each evaluation is one dynamic call into `m`.

Properties other than `meta` and `counters` are read from `m`.
"""
struct ModelBarrier{T,VT} <: NLPModels.AbstractNLPModel{T,VT}
    inner::NLPModels.AbstractNLPModel{T,VT}
    meta::NLPModels.NLPModelMeta{T,VT}
    counters::NLPModels.Counters
    ModelBarrier{T,VT}(@nospecialize(inner), meta, counters) where {T,VT} =
        new{T,VT}(inner, meta, counters)
end

# Not specialized on the model type: the barrier is built once per solve, and reading
# `meta` through `invokelatest` keeps `getproperty` methods added later from
# invalidating this method and `madnlp`, which calls it.
ModelBarrier(@nospecialize(m::NLPModels.AbstractNLPModel)) =
    _model_barrier(m, Base.invokelatest(getproperty, m, :meta)::NLPModels.NLPModelMeta)
_model_barrier(@nospecialize(m), meta::NLPModels.NLPModelMeta{T,VT}) where {T,VT} =
    ModelBarrier{T,VT}(m, meta, NLPModels.Counters())
ModelBarrier(m::ModelBarrier) = m

Base.getproperty(m::ModelBarrier, s::Symbol) =
    s === :inner || s === :meta || s === :counters ? getfield(m, s) :
    Base.invokelatest(getproperty, getfield(m, :inner), s)
Base.hasproperty(m::ModelBarrier, s::Symbol) =
    s === :inner || s === :meta || s === :counters || Base.invokelatest(hasproperty, getfield(m, :inner), s)::Bool

NLPModels.obj(m::ModelBarrier{T}, x::AbstractVector) where {T} = Base.invokelatest(NLPModels.obj, m.inner, x)::T
NLPModels.grad!(m::ModelBarrier, x::AbstractVector, g::AbstractVector) =
    (Base.invokelatest(NLPModels.grad!, m.inner, x, g); g)
NLPModels.cons!(m::ModelBarrier, x::AbstractVector, c::AbstractVector) =
    (Base.invokelatest(NLPModels.cons!, m.inner, x, c); c)
NLPModels.jac_structure!(m::ModelBarrier, I::AbstractVector, J::AbstractVector) =
    (Base.invokelatest(NLPModels.jac_structure!, m.inner, I, J); (I, J))
NLPModels.hess_structure!(m::ModelBarrier, I::AbstractVector, J::AbstractVector) =
    (Base.invokelatest(NLPModels.hess_structure!, m.inner, I, J); (I, J))
NLPModels.jac_coord!(m::ModelBarrier, x::AbstractVector, J::AbstractVector) =
    (Base.invokelatest(NLPModels.jac_coord!, m.inner, x, J); J)
NLPModels.hess_coord!(m::ModelBarrier{T}, x::AbstractVector, y::AbstractVector, H::AbstractVector;
                      obj_weight = one(T)) where {T} =
    (Base.invokelatest(NLPModels.hess_coord!, m.inner, x, y, H; obj_weight = obj_weight); H)
NLPModels.jprod!(m::ModelBarrier, x::AbstractVector, v::AbstractVector, Jv::AbstractVector) =
    (Base.invokelatest(NLPModels.jprod!, m.inner, x, v, Jv); Jv)
NLPModels.jtprod!(m::ModelBarrier, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector) =
    (Base.invokelatest(NLPModels.jtprod!, m.inner, x, v, Jtv); Jtv)
NLPModels.jac_dense!(m::ModelBarrier, x::AbstractVector, J::AbstractMatrix) =
    (Base.invokelatest(NLPModels.jac_dense!, m.inner, x, J); J)
NLPModels.hess_dense!(m::ModelBarrier{T}, x::AbstractVector, y::AbstractVector, H::AbstractMatrix;
                      obj_weight = one(T)) where {T} =
    (Base.invokelatest(NLPModels.hess_dense!, m.inner, x, y, H; obj_weight = obj_weight); H)
