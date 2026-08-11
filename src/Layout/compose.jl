# Plan composition. `compose(outer, inner)` returns the Plan that applies
# `inner` then `outer`, expressed against `inner`'s parent. Returns
# `nothing` when the composition is structurally impossible (caller falls
# back to a generic path).

"""
    compose(outer::Plan, inner::Plan) → Union{Plan, Nothing}

Compose `outer` after `inner`. Phase 2 supports `inner` consisting of
`AbstractSlice` and `Drop` steps (i.e., `layout(::SubArray)` shapes plus
`layout(::AbstractArray)`). Returns `nothing` if `inner` contains
unsupported step types or if `outer`'s structure doesn't fit through
`inner`'s cropping (alignment refusal).
"""
function compose(outer::Plan, inner::Plan)
    output_steps = _filter_non_drop(inner.steps)
    drops = _filter_drop(inner.steps)

    composed = _compose_outer(outer.steps, output_steps)
    composed === nothing && return nothing

    # Append inner's Drops so the composed Plan accounts for every parent dim
    # of the inner plan. Drops contribute no output, so trailing position is
    # semantically harmless.
    return Plan{parent_ndims(inner)}((composed..., drops...))
end

# Recursively walk outer.steps, calling _compose_step on each.
@inline _compose_outer(::Tuple{}, ::Tuple) = ()
@inline function _compose_outer(outer_steps::Tuple, inner_outputs::Tuple)
    h = _compose_step(first(outer_steps), inner_outputs)
    h === nothing && return nothing
    rest = _compose_outer(Base.tail(outer_steps), inner_outputs)
    rest === nothing && return nothing
    return (h..., rest...)
end

# Filter inner steps: keep only those that contribute an output dim
# (i.e., not Drop). Drops carry no output dim contribution.
@inline _filter_non_drop(::Tuple{}) = ()
@inline function _filter_non_drop(steps::Tuple)
    h = first(steps)
    rest = _filter_non_drop(Base.tail(steps))
    return h isa Drop ? rest : (h, rest...)
end

@inline _filter_drop(::Tuple{}) = ()
@inline function _filter_drop(steps::Tuple)
    h = first(steps)
    rest = _filter_drop(Base.tail(steps))
    return h isa Drop ? (h, rest...) : rest
end

# ── Per-step composition rules ───────────────────────────────────────────

# Slice ∘ AbstractSlice : outer is full-extent, so the result is the inner
# step itself (already references the correct parent_dim).
@inline function _compose_step(o::Slice, inner::Tuple)
    s = inner[o.parent_dim]
    s isa AbstractSlice || return nothing
    return (s,)
end

# SliceRange ∘ AbstractSlice : compose offsets; outer's length wins.
@inline function _compose_step(o::SliceRange, inner::Tuple)
    s = inner[o.parent_dim]
    s isa AbstractSlice || return nothing
    inner_offset = s isa SliceRange ? s.offset : 0
    return (SliceRange(s.parent_dim, o.offset + inner_offset, o.length),)
end

# Group ∘ k consecutive Slices over consecutive parent dims, where the
# first k-1 are full extent and the last one may carry a UnitRange-style
# crop. The last-only crop is the unique pattern that keeps the merged
# flat axis contiguous in column-major storage.
@inline function _compose_step(o::Group, inner::Tuple)
    first_inner = inner[o.parent_start]
    first_inner isa AbstractSlice || return nothing
    expected_pd = first_inner.parent_dim

    # All but the last grouped slice MUST be full-extent.
    for k in 0:o.nparents - 2
        s = inner[o.parent_start + k]
        s isa Slice || return nothing
        s.parent_dim == expected_pd + k || return nothing
    end

    last_k = o.nparents - 1
    last_s = inner[o.parent_start + last_k]
    last_s isa AbstractSlice || return nothing
    last_s.parent_dim == expected_pd + last_k || return nothing

    if last_s isa SliceRange
        return (GroupRange(first_inner.parent_dim, o.nparents,
                           last_s.offset, last_s.length),)
    else
        return (Group(first_inner.parent_dim, o.nparents),)
    end
end

# GroupRange ∘ ... : outer carries cropping; inner's last slice must be
# full-extent (both cropped → would need re-alignment, refused).
@inline function _compose_step(o::GroupRange, inner::Tuple)
    first_inner = inner[o.parent_start]
    first_inner isa AbstractSlice || return nothing
    expected_pd = first_inner.parent_dim

    for k in 0:o.nparents - 2
        s = inner[o.parent_start + k]
        s isa Slice || return nothing
        s.parent_dim == expected_pd + k || return nothing
    end

    last_k = o.nparents - 1
    last_s = inner[o.parent_start + last_k]
    last_s isa Slice || return nothing
    last_s.parent_dim == expected_pd + last_k || return nothing

    return (GroupRange(first_inner.parent_dim, o.nparents,
                       o.last_offset, o.last_length),)
end

# Spread ∘ AbstractSlice : push any inner cropping into a SpreadRange.
@inline function _compose_step(o::Spread, inner::Tuple)
    s = inner[o.parent_dim]
    s isa AbstractSlice || return nothing
    if s isa SliceRange
        return (SpreadRange(s.parent_dim, o.sizes, s.offset, s.length),)
    else
        return (Spread(s.parent_dim, o.sizes),)
    end
end

# SpreadRange ∘ Slice : outer keeps its cropping, inner must be full-extent
# (both cropped → refuse, would need re-alignment math).
@inline function _compose_step(o::SpreadRange, inner::Tuple)
    s = inner[o.parent_dim]
    s isa Slice || return nothing
    return (SpreadRange(s.parent_dim, o.sizes, o.parent_offset, o.parent_length),)
end

# Singleton passthrough.
@inline _compose_step(::Singleton, ::Tuple) = (Singleton(),)

# Drop ∘ AbstractSlice : the inner slice's length must be 1 (verified at
# materialize time). Composed integer index = inner.offset + outer.index.
@inline function _compose_step(o::Drop, inner::Tuple)
    s = inner[o.parent_dim]
    s isa AbstractSlice || return nothing
    inner_offset = s isa SliceRange ? s.offset : 0
    return (Drop(s.parent_dim, inner_offset + o.index),)
end

# Identity-Plan check (kept from Phase 1 for the AbstractArray fast path).
@inline function _is_identity_plan(p::Plan{N}) where {N}
    nsteps(p) == N || return false
    return _is_identity_steps(p.steps, 1)
end

@inline _is_identity_steps(::Tuple{}, ::Int) = true
@inline _is_identity_steps(::Tuple{Vararg{Step}}, ::Int) = false
@inline function _is_identity_steps(steps::Tuple{Slice, Vararg{Step}}, cursor::Int)
    s = first(steps)
    s.parent_dim == cursor || return false
    return _is_identity_steps(Base.tail(steps), cursor + 1)
end
