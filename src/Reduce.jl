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

# Generic method for non-AbstractArray types, built on `f(x; dims)` with its
# `Base` reduction semantics (reduced dims kept as singletons); a type without
# such a method for `f` fails with its `MethodError`.
# `@inline` exposes the inner `f(x; dims)` to accelerator backends that lower by
# walking IR (see the note on the generic `Reshape` executor).
@inline function (op::Reduce{f,dims})(x) where {f,dims}
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

# generic method; more specific than `(::Reduce{f,dims})(x)` so no ambiguity
@inline (op::Reduce{<:Any,()})(x) = x
