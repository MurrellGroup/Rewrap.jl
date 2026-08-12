module Rewrap

using EllipsisNotation; export ..
const Ellipsis = typeof(..)

const IntOrColon = Union{Int,Colon}
const IntOrEllipsis = Union{Int,Ellipsis}
const ColonOrEllipsis = Union{Colon,Ellipsis}

abstract type AxisOp{N,M} end

ndims_in(::Type{<:AxisOp{N}}) where N = N
ndims_in(::Type{<:AxisOp{..}}, ::Val{N}) where N = N

ndims_out(::Type{<:AxisOp{N,M}}) where {N,M} = M
ndims_out(::Type{<:AxisOp{N,M}}, ::Val{N}) where {N,M} = M
ndims_out(::Type{<:AxisOp{..,M}}, ::Val{N}) where {N,M} = M
ndims_out(::Type{<:AxisOp{..,..}}, ::Val{N}) where N = N

ndims_in(op::Type{<:AxisOp}, x::AbstractArray) = ndims_in(op, Val(ndims(x)))
ndims_out(op::Type{<:AxisOp}, x::AbstractArray) = ndims_out(op, Val(ndims(x)))

ndims_in(op::AxisOp, args...) = ndims_in(typeof(op), args...)
ndims_out(op::AxisOp, args...) = ndims_out(typeof(op), args...)

abstract type GlobalAxisOp{N,M} <: AxisOp{N,M} end

macro constprop(ex, setting::Symbol=:aggressive)
    :(Base.@constprop $setting $ex)
end

include("Reshape/Reshape.jl")
export Reshape
export Keep
export Merge
export Split, Split1
export Resqueeze, Squeeze, Unsqueeze

include("enhanced-base/enhanced-base.jl")

include("Permute.jl")
export Permute

include("Reduce.jl")
export Reduce

include("Repeat.jl")
export Repeat

end
