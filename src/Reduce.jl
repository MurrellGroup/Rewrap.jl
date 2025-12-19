"""
    Reduce(f::Function; dims::NTuple{K,Int})

Reduce an array over dimensions `dims`, using e.g.
`sum`, `prod`, `minimum`, `maximum`, `any`, `all`, or `Statistics.mean`.

`f` must accept a `dims::Tuple{Vararg{Int}}` keyword argument, allowing
for reduction over specific dimensions. This should reduce the specified dimensions to singletons,
but not drop them.
"""
struct Reduce{f,dims} <: GlobalAxisOp{..,..} end

function Reduce(f::Function; dims::NTuple{K,Int}) where K
    all(>=(0), dims) || throw(ArgumentError("Reduce dimensions must be non-negative"))
    Reduce{f,dims}()
end

function (::Reduce{f,dims})(x::AbstractArray{<:Any,N}) where {N,f,dims}
    return f(x; dims)
end

# unwrap a lazy permute if only the reduced dimensions were being permuted
@generated function (r::Reduce{f,dims})(x::PermutedDimsArray{<:Any,N,perm}) where {N,perm,f,dims}
    moved = filter(i -> perm[i] != i, 1:N)
    dims′ = filter(<=(N), dims)

    if all(in(dims′), moved)
        parent_dims = Tuple(perm[d] for d in dims′)
        return quote
            f(parent(x); dims=$(QuoteNode(parent_dims)))
        end
    end

    return quote
        f(x; dims=$(QuoteNode(dims′)))
    end
end

(::Reduce{<:Any,()})(x::AbstractArray) = x

# resolve ambiguity
(::Reduce{<:Any,()})(x::PermutedDimsArray) = x
