"""
    layout(x::AbstractArray) → Plan

Describe `x` as a `Plan` over its underlying parent. For a plain `Array`,
this is the identity Plan. Wrapper types (`SubArray`, `PermutedDimsArray`,
`ReinterpretArray`) get specialized methods in later phases.
"""
function layout end

# Generic AbstractArray = identity Plan over itself.
layout(x::AbstractArray{<:Any,N}) where {N} =
    Plan{N}(ntuple(Slice, Val(N)))

# SubArray: walk parentindices, build one Step per parent dim.
# Returns `nothing` if any index type isn't expressible as Slice/Drop —
# caller falls back to a wrapper-specific path (or Base.reshape).
function layout(x::SubArray)
    inds = parentindices(x)
    P_ndims = length(inds)
    steps = _layout_subarray_steps(inds, 1)
    steps === nothing && return nothing
    return Plan{P_ndims}(steps)
end

@inline _layout_subarray_steps(::Tuple{}, ::Int) = ()
@inline function _layout_subarray_steps(inds::Tuple, cursor::Int)
    s = _step_for_index(first(inds), cursor)
    s === nothing && return nothing
    rest = _layout_subarray_steps(Base.tail(inds), cursor + 1)
    rest === nothing && return nothing
    return (s, rest...)
end

# Per-index Step. Refused types return `nothing`.
# `Base.Slice` (the canonical `:` index) and `Base.OneTo` (`1:n` over a whole
# dim) lower to a full-extent `Slice`; explicit unit ranges become
# `SliceRange`. `compose` recognizes the full-extent case via type dispatch
# and lets `Group` / `Spread` compose through them.
@inline _step_for_index(idx::Integer, p::Int) = Drop(p, Int(idx))
@inline _step_for_index(::Base.Slice, p::Int) = Slice(p)
@inline _step_for_index(::Base.OneTo, p::Int) = Slice(p)
@inline _step_for_index(idx::AbstractUnitRange{<:Integer}, p::Int) =
    SliceRange(p, Int(first(idx)) - 1, length(idx))
@inline _step_for_index(_, ::Int) = nothing

"""
    root_parent(x::AbstractArray) → AbstractArray

Peel through wrapper layers to the underlying storage that `layout(x)`
described its `Plan` against. For unwrapped arrays this is `x` itself.
"""
function root_parent end

root_parent(x::AbstractArray) = x
root_parent(x::SubArray) = parent(x)
root_parent(x::PermutedDimsArray) = parent(x)
# ReinterpretArray method lands later in Phase 3.

# PermutedDimsArray: output dim i = parent dim perm[i], full extent.
function layout(x::PermutedDimsArray{T,N,perm}) where {T,N,perm}
    return Plan{N}(ntuple(i -> Slice(perm[i]), Val(N)))
end
