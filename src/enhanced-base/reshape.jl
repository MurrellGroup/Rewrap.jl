const AnyOp = Union{LocalReshape,Int,Colon,Ellipsis}

"""
    Rewrap.reshape(x, ops...)
    Rewrap.reshape(x, ops::Tuple)

Reshape the array `x` using the given operations, which can include
`:` (Base.Colon), `..` (EllipsisNotation.Ellipsis), and integers.

Integers and colons can form a single contiguous sequence that becomes a `Split(.., sizes)`.

See also [`Base.reshape`](@ref).

```jldoctest
julia> x = view(rand(4, 5, 2), 1:3, :, :);

julia> x′ = Rewrap.reshape(x, Keep(), :);

julia> size(x′)
(3, 10)

julia> y′ = rand(2, 3) * x′; # project from 3 to 2

julia> size(y′)
(2, 10)

julia> y = Rewrap.reshape(y′, Keep(), Split(1, size(x)[2:end]));

julia> size(y)
(2, 5, 2)

julia> Rewrap.reshape(view(rand(2, 3), :, 1:2), :) |> summary # Rewrap owns `Rewrap.reshape`
"4-element view(::Vector{Float64}, 1:4) with eltype Float64"
```
"""
function reshape end

Rewrap.reshape(x::AbstractArray, args...) = Base.reshape(x, args...)

@constprop function Rewrap.reshape(x::AbstractArray, ops::Tuple{AnyOp,Vararg{AnyOp}})
    r = Reshape(ops, Val(ndims(x)))
    r(x)
end

# Generic sibling for non-AbstractArray types. Kept separate from the
# `AbstractArray` method (rather than relaxing it) so arrays still hit the more
# specific method and there is no ambiguity with the `args...` passthrough above.
# Opt-in is enforced downstream in the `Reshape` executor via `supports_fallback`.
# `@inline` so the `r(x)` executor call is exposed to accelerator backends that
# lower by walking IR (see the note on the generic `Reshape` executor).
@inline @constprop function Rewrap.reshape(x, ops::Tuple{AnyOp,Vararg{AnyOp}})
    r = Reshape(ops, Val(ndims(x)))
    r(x)
end

@constprop function Rewrap.reshape(x::AbstractArray, op1::AnyOp, ops::AnyOp...)
    return Rewrap.reshape(x, (op1, ops...))
end

@constprop function Rewrap.reshape(x, op1::AnyOp, ops::AnyOp...)
    return Rewrap.reshape(x, (op1, ops...))
end

## Base.reshape

const NonLocalOp = Union{Int,Colon,Ellipsis}

"""
    Base.reshape(x, ops...)
    Base.reshape(x, ops::Tuple)

Reshape the array `x` using the given operations, which can include
`:` (Base.Colon), `..` (EllipsisNotation.Ellipsis), and integers,
but there must be at least one `LocalReshape`.

See also [`Rewrap.reshape`](@ref).

```jldoctest
julia> x = view(rand(4, 5, 2), 1:3, :, :);

julia> x′ = reshape(x, Keep(), :);

julia> size(x′)
(3, 10)

julia> y′ = rand(2, 3) * x′; # project from 3 to 2

julia> size(y′)
(2, 10)

julia> y = reshape(y′, Keep(), Split(1, size(x)[2:end]));

julia> size(y)
(2, 5, 2)

julia> reshape(view(rand(2, 3), :, 1:2), Merge(..)) |> summary # can not use a single `:` (type piracy)
"4-element view(::Vector{Float64}, 1:4) with eltype Float64"
```
"""
Base.reshape

const LocalReshapeOpsTuple = Union{
    Tuple{LocalReshape,Vararg{AnyOp}},
    Tuple{NonLocalOp,LocalReshape,Vararg{AnyOp}},
    Tuple{NonLocalOp,NonLocalOp,LocalReshape,Vararg{AnyOp}},
    Tuple{NonLocalOp,NonLocalOp,NonLocalOp,LocalReshape,Vararg{AnyOp}},
    Tuple{NonLocalOp,NonLocalOp,NonLocalOp,NonLocalOp,LocalReshape,Vararg{AnyOp}}
}

@constprop function Base.reshape(x::AbstractArray, ops::LocalReshapeOpsTuple)
    return Rewrap.reshape(x, ops)
end

# Generic sibling for non-AbstractArray types. Dispatching on the
# `LocalReshape`-bearing tuple keeps this a legitimate (non-pirating) method on
# `Base.reshape`. Opt-in is enforced downstream via `supports_fallback`.
@constprop function Base.reshape(x, ops::LocalReshapeOpsTuple)
    return Rewrap.reshape(x, ops)
end

@constprop function Base.reshape(
    x::AbstractArray, op1::LocalReshape, ops::AnyOp...
)
    return Rewrap.reshape(x, (op1, ops...))
end

@constprop function Base.reshape(
    x::AbstractArray, op1::NonLocalOp, op2::LocalReshape, ops::AnyOp...
)
    return Rewrap.reshape(x, (op1, op2, ops...))
end

@constprop function Base.reshape(
    x::AbstractArray, op1::NonLocalOp, op2::NonLocalOp, op3::LocalReshape, ops::AnyOp...
)
    return Rewrap.reshape(x, (op1, op2, op3, ops...))
end

@constprop function Base.reshape(
    x::AbstractArray, op1::NonLocalOp, op2::NonLocalOp, op3::NonLocalOp, op4::LocalReshape, ops::AnyOp...
)
    return Rewrap.reshape(x, (op1, op2, op3, op4, ops...))
end

@constprop function Base.reshape(
    x::AbstractArray, op1::NonLocalOp, op2::NonLocalOp, op3::NonLocalOp, op4::NonLocalOp, op5::LocalReshape, ops::AnyOp...
)
    return Rewrap.reshape(x, (op1, op2, op3, op4, op5, ops...))
end


