abstract type Step end

# ── Slice family ─────────────────────────────────────────────────────────
# Output axis derived from a single parent dim.

abstract type AbstractSlice <: Step end

"""
    Slice(parent_dim)

Output axis = parent dim `parent_dim` in full extent.
"""
struct Slice <: AbstractSlice
    parent_dim::Int
end

"""
    SliceRange(parent_dim, offset, length)

Output axis = `length` elements of parent dim `parent_dim` starting at
element `offset` (0-based).
"""
struct SliceRange <: AbstractSlice
    parent_dim::Int
    offset::Int
    length::Int
end

# ── Group family ─────────────────────────────────────────────────────────
# One output axis = product of consecutive parent dims (the encoding for
# `Merge`). The first `nparents - 1` grouped dims must always be full extent
# for the merge to be expressible as a `view`-compatible flat range.

abstract type AbstractGroup <: Step end

"""
    Group(parent_start, nparents)

Merge `nparents` consecutive parent dims starting at `parent_start` into one
output axis.
"""
struct Group <: AbstractGroup
    parent_start::Int
    nparents::Int
end

"""
    GroupRange(parent_start, nparents, last_offset, last_length)

Like `Group`, but the *last* grouped parent dim is cropped to `last_length`
elements starting at `last_offset` (0-based) — the only crop pattern that
preserves contiguity in column-major storage.
"""
struct GroupRange <: AbstractGroup
    parent_start::Int
    nparents::Int
    last_offset::Int
    last_length::Int
end

# ── Spread family ────────────────────────────────────────────────────────
# Split one parent dim into multiple output axes (the encoding for `Split`).

abstract type AbstractSpread <: Step end

"""
    Spread(parent_dim, sizes)

`length(sizes)` output axes formed by splitting parent dim `parent_dim` into
the given `sizes`. At most one entry of `sizes` may be `Colon`.
"""
struct Spread <: AbstractSpread
    parent_dim::Int
    sizes::Tuple{Vararg{Union{Int,Colon}}}
end

"""
    SpreadRange(parent_dim, sizes, parent_offset, parent_length)

Like `Spread`, but the parent dim is first cropped to the element-range
`parent_offset:parent_offset+parent_length-1` (0-based offset).
"""
struct SpreadRange <: AbstractSpread
    parent_dim::Int
    sizes::Tuple{Vararg{Union{Int,Colon}}}
    parent_offset::Int
    parent_length::Int
end

# ── Standalone steps ─────────────────────────────────────────────────────

"""
    Singleton()

Output axis = inserted size-1 dim. References no parent dim (the encoding
for `Unsqueeze`).
"""
struct Singleton <: Step end

"""
    Drop(parent_dim, index = 1)

Parent dim `parent_dim` consumed with no corresponding output axis.
`index` is the parent-dim coordinate to use when materializing as a `view`
— `1` for `Squeeze` (assumes the dim has size 1), or the original integer
for an integer-indexed `SubArray`.
"""
struct Drop <: Step
    parent_dim::Int
    index::Int
end
Drop(parent_dim::Int) = Drop(parent_dim, 1)

"""
    Plan{ParentN, StepsT}(steps)

A transformation from a `ParentN`-dim parent array to an output array.
`steps` is a tuple of `Step` values; the type structure (which `Step`
subtypes, in which order) is compile-time, while numerical fields ride along
as runtime data the compiler folds when call-site constants permit.
"""
struct Plan{ParentN, StepsT<:Tuple{Vararg{Step}}}
    steps::StepsT
end

Plan{ParentN}(steps::Tuple{Vararg{Step}}) where {ParentN} =
    Plan{ParentN, typeof(steps)}(steps)

parent_ndims(::Plan{N}) where {N} = N
parent_ndims(::Type{<:Plan{N}}) where {N} = N

steps(p::Plan) = p.steps
nsteps(p::Plan) = length(p.steps)

# True iff every step references a parent dim from a partition of 1:ParentN
# with each parent dim used exactly once. Sanity check; not enforced at
# construction (cheap to skip), called by tests.
function is_well_formed(p::Plan{N}) where {N}
    seen = falses(N)
    for s in p.steps
        if s isa AbstractSlice
            (1 <= s.parent_dim <= N) || return false
            seen[s.parent_dim] && return false
            seen[s.parent_dim] = true
        elseif s isa AbstractGroup
            (s.nparents >= 1) || return false
            for k in 0:s.nparents-1
                pd = s.parent_start + k
                (1 <= pd <= N) || return false
                seen[pd] && return false
                seen[pd] = true
            end
        elseif s isa AbstractSpread
            (1 <= s.parent_dim <= N) || return false
            seen[s.parent_dim] && return false
            seen[s.parent_dim] = true
            count(x -> x isa Colon, s.sizes) <= 1 || return false
        elseif s isa Drop
            (1 <= s.parent_dim <= N) || return false
            seen[s.parent_dim] && return false
            seen[s.parent_dim] = true
        elseif s isa Singleton
            # no parent dim
        else
            return false
        end
    end
    return all(seen)
end
