const AnyOp = Union{LocalReshape,ColonOrEllipsis}

"""
    Rewrap.reshape(x, ops...)
    Rewrap.reshape(x, ops::Tuple)

Reshape the array `x` using the given operations, which can include
`:` (Base.Colon) and `..` (EllipsisNotation.Ellipsis).

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

@constprop function Rewrap.reshape(x::AbstractArray, ops::Tuple{LocalReshape,Vararg{LocalReshape}})
    r = Reshape(ops, Val(ndims(x)))
    r(x)
end

@constprop function Rewrap.reshape(x::AbstractArray, ops::Tuple{AnyOp,Vararg{AnyOp}})
    count(op -> op isa ColonOrEllipsis, ops) > 1 && throw(ArgumentError("At most one Colon or Ellipsis is allowed"))
    ops′ = map(ops) do op
        if op isa Colon
            Merge(..)
        elseif op isa Ellipsis
            Keep(..)
        else
            op
        end
    end
    return Rewrap.reshape(x, ops′)
end

@constprop function Rewrap.reshape(x::AbstractArray, op1::AnyOp, ops::AnyOp...)
    return Rewrap.reshape(x, (op1, ops...))
end

## Base.reshape

"""
    Base.reshape(x, ops...)
    Base.reshape(x, ops::Tuple)

Reshape the array `x` using the given operations, which can include
`:` (Base.Colon) and `..` (EllipsisNotation.Ellipsis), but there
must be at least one `LocalReshape`.

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

@constprop function Base.reshape(x::AbstractArray, ops::Tuple{LocalReshape,Vararg{LocalReshape}})
    return Rewrap.reshape(x, ops)
end

@constprop function Base.reshape(
    x::AbstractArray,
    ops::Union{
        Tuple{ColonOrEllipsis,LocalReshape,Vararg{LocalReshape}},
        Tuple{LocalReshape,Vararg{AnyOp}}
    }
)
    return Rewrap.reshape(x, ops)
end

@constprop function Base.reshape(
    x::AbstractArray, op1::LocalReshape, ops::Union{LocalReshape,ColonOrEllipsis}...
)
    return Rewrap.reshape(x, (op1, ops...))
end

@constprop function Base.reshape(
    x::AbstractArray, op1::ColonOrEllipsis, op2::LocalReshape, ops::LocalReshape...
)
    return Rewrap.reshape(x, (op1, op2, ops...))
end


