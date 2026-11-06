# Options

parse_option(::Type{Module},str::String) = eval(Symbol(str))
parse_option(::Type{<:AbstractUserCallback}, f::Any) = f
parse_option(type::Type{T},i::Int64) where {T<:Enum} = type(i)

function set_options!(opt::AbstractOptions, options)
    other_options = Dict{Symbol, Any}()
    for (key, val) in options
        if hasproperty(opt, key)
            T = fieldtype(typeof(opt), key)
            val isa T ? setproperty!(opt,key,val) :
                setproperty!(opt,key,parse_option(T,val))
        else
            other_options[key] = val
        end
    end
    return other_options
end

is_dense_callback(nlp) = !nlp.meta.sparse_jacobian && !nlp.meta.sparse_hessian

get_tolerance(::Type{T},::Type{KKT}) where {T, KKT} = 10^round(log10(eps(T))/2)
get_tolerance(::Type{T},::Type{SparseCondensedKKTSystem}) where T = 10^(round(log10(eps(T))/2))

"""
    default_sparse_solver(nlp)

Return the sparse linear solver used when the option `linear_solver` is not set.
MadCore and MadNLPCore return [`DummySolver`](@ref), which throws an error when the
solver is built; MadNLP sets the default to [`MumpsSolver`](@ref).
"""
default_sparse_solver(nlp) = DummySolver

function print_ignored_options(logger,option_dict)
    @warn(logger,"The following options are ignored: ")
    for (key,val) in option_dict
        @warn(logger," - "*string(key))
    end
end

function _get_primary_options(options)
    primary_opt = Dict{Symbol,Any}()
    remaining_opt = Dict{Symbol,Any}()
    for (k,v) in options
        if k in [:tol, :linear_solver, :callback, :kkt_system]
            primary_opt[k] = v
        else
            remaining_opt[k] = v
        end
    end

    return primary_opt, remaining_opt
end

"""
    default_options(
        nlp::AbstractNLPModel,
        kkt::Type,
        linear_solver::Type,
    )

Return the default options for `linear_solver` associated to the KKT system `kkt` and `nlp`.
"""
default_options(nlp::AbstractNLPModel, kkt, linear_solver) = default_options(linear_solver)
