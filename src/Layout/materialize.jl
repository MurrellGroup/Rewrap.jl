# Materialize a Plan against a parent array → minimum wrapper chain.
#
# Decision tree (Phase 2):
#   no view, no reshape   → parent itself
#   reshape only          → Base.reshape(parent, shape)
#   view only             → view(parent, indices...)
#   view + reshape needed → Base.reshape(view(parent, indices...), shape)
#                           (or the reverse, depending on Plan structure)

"""
    materialize(plan::Plan, parent::AbstractArray) → AbstractArray

Realize `plan` as the minimum wrapper chain over `parent`.
"""
function materialize(plan::Plan{N}, parent::AbstractArray{<:Any,N}) where {N}
    # Permuted-parent_dim path covers both pure permutation (no reshape) and
    # reshape-then-permute. Handles PDA inputs uniformly.
    if _has_nontrivial_parent_dim_order(plan)
        return _materialize_permuted(plan, parent)
    end

    needs_v = _needs_view(plan, parent)
    needs_r = _needs_reshape(plan)

    if !needs_v && !needs_r
        return parent
    elseif !needs_v && needs_r
        return Base.reshape(parent, output_shape(plan, parent))
    elseif needs_v && !needs_r
        inds = _view_indices(plan, parent)
        return view(parent, inds...)
    else
        return _materialize_via_view_reshape(plan, parent)
    end
end

function materialize(plan::Plan{Pn}, parent::AbstractArray{<:Any,N}) where {Pn,N}
    throw(DimensionMismatch(
        "Plan expects $Pn parent dims; root parent has $N. Plan: $plan"
    ))
end

# ── needs-view / needs-reshape predicates ────────────────────────────────

@inline function _needs_view(plan::Plan, parent::AbstractArray)
    for s in plan.steps
        if s isa SliceRange
            (s.offset != 0 || s.length != size(parent, s.parent_dim)) && return true
        elseif s isa Drop
            return true
        elseif s isa SpreadRange
            (s.parent_offset != 0 ||
             s.parent_length != size(parent, s.parent_dim)) && return true
        elseif s isa GroupRange
            return true
        end
    end
    return false
end

@inline function _needs_reshape(plan::Plan)
    for s in plan.steps
        (s isa AbstractGroup || s isa AbstractSpread || s isa Singleton) && return true
    end
    return false
end

# ── view-only path ───────────────────────────────────────────────────────
# Build an N-tuple of view indices (one per parent dim) from the Plan's
# AbstractSlice / Drop steps. Other step types in this Plan would have
# triggered the view+reshape path, so this is only hit when the Plan is
# pure view-shaped.

@inline function _view_indices(plan::Plan{N}, parent::AbstractArray{<:Any,N}) where {N}
    # Vector{Any} is not type-stable but is the simplest correct construction
    # for arbitrary parent_dim orderings. Optimization (per-Plan-type @generated
    # specialization) deferred.
    inds = Vector{Any}(undef, N)
    for p in 1:N
        inds[p] = Base.Slice(Base.OneTo(size(parent, p)))
    end
    for s in plan.steps
        if s isa SliceRange
            if s.offset == 0 && s.length == size(parent, s.parent_dim)
                # leave default Base.Slice
            else
                inds[s.parent_dim] = (s.offset + 1):(s.offset + s.length)
            end
        elseif s isa Drop
            inds[s.parent_dim] = s.index
        end
    end
    return Tuple(inds)
end

# ── permuted parent_dim path ─────────────────────────────────────────────
# When the Plan's steps reference parent dims out of natural (1..N) order —
# typically because `inner` was a `PermutedDimsArray` layout — sort the
# steps by parent dim, reshape the parent in that natural order, then
# `PermutedDimsArray` to the user's requested ordering.

@inline _step_first_parent_dim(s::AbstractSlice)  = s.parent_dim
@inline _step_first_parent_dim(s::AbstractGroup)  = s.parent_start
@inline _step_first_parent_dim(s::AbstractSpread) = s.parent_dim
@inline _step_first_parent_dim(s::Drop)           = s.parent_dim
@inline _step_first_parent_dim(::Singleton)       = 0  # no parent contribution

@inline _step_output_rank(::AbstractSlice)   = 1
@inline _step_output_rank(::AbstractGroup)   = 1
@inline _step_output_rank(s::AbstractSpread) = length(s.sizes)
@inline _step_output_rank(::Singleton)       = 1
@inline _step_output_rank(::Drop)            = 0

@inline function _has_nontrivial_parent_dim_order(plan::Plan)
    last_pd = 0
    for s in plan.steps
        pd = _step_first_parent_dim(s)
        pd == 0 && continue
        pd >= last_pd || return true
        last_pd = pd
    end
    return false
end

function _materialize_permuted(plan::Plan{N}, parent::AbstractArray{<:Any,N}) where {N}
    n = nsteps(plan)

    # Sort step indices by their first parent_dim (Singletons go to the
    # end via the N+1 sort key).
    sort_keys = [_step_sort_key(plan.steps[i], N) for i in 1:n]
    order = sortperm(sort_keys)

    # Steps in natural (parent-dim sorted) order.
    natural_steps_vec = [plan.steps[order[i]] for i in 1:n]

    # Natural-order shape: each step contributes its output dims.
    natural_shape = _shape_from_step_iter(natural_steps_vec, parent)

    # Output ranks per natural-position step.
    natural_ranks = [_step_output_rank(plan.steps[order[i]]) for i in 1:n]

    # Cumulative starting positions in natural-order output.
    natural_starts = Vector{Int}(undef, n)
    cum = 1
    for i in 1:n
        natural_starts[i] = cum
        cum += natural_ranks[i]
    end

    # Build perm: for user step i, its output dims start at the corresponding
    # natural position; concatenate.
    perm_parts = Int[]
    for user_i in 1:n
        nat_idx = findfirst(==(user_i), order)
        for j in 0:natural_ranks[nat_idx] - 1
            push!(perm_parts, natural_starts[nat_idx] + j)
        end
    end
    perm = Tuple(perm_parts)

    # `Base.reshape` doesn't always return the array unchanged when the
    # shape matches (varies by version), so guard the no-op explicitly to
    # preserve `parent(result) === parent` for the wrapper-elision tests.
    natural_reshaped = natural_shape == size(parent) ? parent :
                       Base.reshape(parent, natural_shape)
    return _maybe_permute(natural_reshaped, perm)
end

@inline _step_sort_key(s::AbstractSlice,  ::Int) = s.parent_dim
@inline _step_sort_key(s::AbstractGroup,  ::Int) = s.parent_start
@inline _step_sort_key(s::AbstractSpread, ::Int) = s.parent_dim
@inline _step_sort_key(s::Drop,           ::Int) = s.parent_dim
@inline _step_sort_key(::Singleton,       N::Int) = N + 1   # Singletons at end

# Build natural shape from a Vector of steps (runtime-typed; OK because this
# path is for non-AbstractArray inputs where Einops's @inferred doesn't apply).
function _shape_from_step_iter(steps_vec, parent::AbstractArray)
    out = Int[]
    for s in steps_vec
        chunks = _shape_for_step(s, parent)
        append!(out, chunks)
    end
    return Tuple(out)
end

@inline function _maybe_permute(x::AbstractArray{<:Any,N}, perm::NTuple{N,Int}) where {N}
    perm == ntuple(identity, Val(N)) ? x : PermutedDimsArray(x, perm)
end

# ── view + reshape path ──────────────────────────────────────────────────
# Each step contributes `(parent_shape_chunk, ind_chunk)`. After
# `reshape(parent, full_shape)` and `view(_, full_inds...)`, the SubArray
# has the desired output shape — preserving storage sharing where possible.
#
# Singleton steps are unique: they produce no parent contribution but add
# a size-1 output dim. To avoid an outer reshape (which would lose
# SubArray-ness), they're folded as `Base.OneTo(1)` indices on a
# size-1 reshape dim.

function _materialize_via_view_reshape(plan::Plan, parent::AbstractArray)
    parent_shape, inds = _walk_view_reshape(plan.steps, parent, (), ())
    reshaped = Base.reshape(parent, parent_shape)
    return view(reshaped, inds...)
end

@inline _walk_view_reshape(::Tuple{}, ::AbstractArray, parent_shape::Tuple, inds::Tuple) =
    (parent_shape, inds)
@inline function _walk_view_reshape(steps::Tuple, parent::AbstractArray,
                                    parent_shape::Tuple, inds::Tuple)
    ps, is = _step_chunks(first(steps), parent)
    return _walk_view_reshape(Base.tail(steps), parent,
                              (parent_shape..., ps...), (inds..., is...))
end

# Per-step chunks: (parent_shape_contribution, inds_contribution).

@inline function _step_chunks(s::Slice, parent::AbstractArray)
    full = size(parent, s.parent_dim)
    return ((full,), (Base.Slice(Base.OneTo(full)),))
end

@inline function _step_chunks(s::SliceRange, parent::AbstractArray)
    full = size(parent, s.parent_dim)
    if s.offset == 0 && s.length == full
        return ((full,), (Base.Slice(Base.OneTo(full)),))
    end
    1 <= s.offset + 1 && s.offset + s.length <= full || throw(BoundsError(
        "SliceRange at parent dim $(s.parent_dim): " *
        "offset $(s.offset), length $(s.length) out of bounds 1:$full"
    ))
    return ((full,), ((s.offset + 1):(s.offset + s.length),))
end

@inline function _step_chunks(s::Group, parent::AbstractArray)
    merged = 1
    @inbounds for k in 0:s.nparents - 1
        merged *= size(parent, s.parent_start + k)
    end
    return ((merged,), (Base.Slice(Base.OneTo(merged)),))
end

@inline function _step_chunks(s::GroupRange, parent::AbstractArray)
    # Group always materializes as ONE flat parent dim of size `merged`;
    # `last_offset`/`last_length` carve a UnitRange of that flat dim.
    merged = 1
    @inbounds for k in 0:s.nparents - 1
        merged *= size(parent, s.parent_start + k)
    end
    prefix = 1
    @inbounds for k in 0:s.nparents - 2
        prefix *= size(parent, s.parent_start + k)
    end
    flat_offset = s.last_offset * prefix
    flat_length = s.last_length * prefix
    return ((merged,), ((flat_offset + 1):(flat_offset + flat_length),))
end

@inline function _step_chunks(s::Spread, parent::AbstractArray)
    full = size(parent, s.parent_dim)
    sizes_resolved = _resolve_split_sizes(s.sizes, full)
    return (sizes_resolved,
            ntuple(i -> Base.Slice(Base.OneTo(sizes_resolved[i])),
                   length(sizes_resolved)))
end

@inline function _step_chunks(s::SpreadRange, parent::AbstractArray)
    full = size(parent, s.parent_dim)
    extent = s.parent_length
    sizes_resolved = _resolve_split_sizes(s.sizes, extent)
    sizes_prod = 1
    for sz in sizes_resolved
        sizes_prod *= sz
    end

    # Cropped Spread: introduce a leftover factor in the reshape and
    # index it down. Alignment checks throw matching the legacy code's
    # error contracts.
    full % sizes_prod == 0 || throw(DimensionMismatch(
        "Cannot reshape SubArray view: parent dimension $(s.parent_dim) " *
        "($full) is not divisible by Spread block size $sizes_prod"
    ))
    leftover = full ÷ sizes_prod

    s.parent_offset % sizes_prod == 0 || throw(DimensionMismatch(
        "Cannot reshape SubArray view: parent_offset $(s.parent_offset) " *
        "not aligned to Spread block size $sizes_prod"
    ))
    extent % sizes_prod == 0 || throw(DimensionMismatch(
        "Cannot reshape SubArray view: cropped extent $extent " *
        "not aligned to Spread block size $sizes_prod"
    ))

    n_super = extent ÷ sizes_prod
    leftover_offset = s.parent_offset ÷ sizes_prod

    leftover_ind = n_super == 1 ?
        leftover_offset + 1 :
        ((leftover_offset + 1):(leftover_offset + n_super))

    parent_shape_contrib = (sizes_resolved..., leftover)
    inds_contrib = (
        ntuple(i -> Base.Slice(Base.OneTo(sizes_resolved[i])),
               length(sizes_resolved))...,
        leftover_ind,
    )
    return (parent_shape_contrib, inds_contrib)
end

# Singleton: zero parent dims + zero output dims contribution to the
# reshape/view, but the output needs a size-1 dim. Encode as a parent
# contribution of `1` (which the reshape can introduce) and a `:` index.
@inline _step_chunks(::Singleton, ::AbstractArray) =
    ((1,), (Base.Slice(Base.OneTo(1)),))

@inline function _step_chunks(s::Drop, parent::AbstractArray)
    full = size(parent, s.parent_dim)
    1 <= s.index <= full || throw(BoundsError(
        "Drop at parent dim $(s.parent_dim) uses index $(s.index); " *
        "valid range 1:$full"
    ))
    return ((full,), (s.index,))
end

# ── output_shape ─────────────────────────────────────────────────────────

"""
    output_shape(plan::Plan, parent::AbstractArray) → Tuple{Vararg{Int}}

The shape of the array that `materialize(plan, parent)` would produce. Each
`Step` contributes 0, 1, or `length(sizes)` dims:

  - `Slice` → 1 dim (full parent extent)
  - `SliceRange` → 1 dim (cropped length)
  - `Group` → 1 dim (product of grouped parent extents)
  - `GroupRange` → 1 dim (product * cropped last extent)
  - `Spread` → `length(sizes)` dims (with at most one `Colon` resolved)
  - `SpreadRange` → `length(sizes)` dims (sizes resolved against cropped extent)
  - `Singleton` → 1 dim of size 1
  - `Drop` → 0 dims (asserts the parent dim has size 1 *or* a valid integer index)
"""
@inline output_shape(plan::Plan, parent::AbstractArray) =
    _shape_for_steps(plan.steps, parent)

@inline _shape_for_steps(::Tuple{}, ::AbstractArray) = ()
@inline function _shape_for_steps(steps::Tuple, parent::AbstractArray)
    h = _shape_for_step(first(steps), parent)
    return (h..., _shape_for_steps(Base.tail(steps), parent)...)
end

@inline function _shape_for_step(s::Slice, parent::AbstractArray)
    return (size(parent, s.parent_dim),)
end

@inline function _shape_for_step(s::SliceRange, ::AbstractArray)
    return (s.length,)
end

@inline function _shape_for_step(s::Group, parent::AbstractArray)
    p = 1
    @inbounds for k in 0:s.nparents - 1
        p *= size(parent, s.parent_start + k)
    end
    return (p,)
end

@inline function _shape_for_step(s::GroupRange, parent::AbstractArray)
    p = 1
    @inbounds for k in 0:s.nparents - 2
        p *= size(parent, s.parent_start + k)
    end
    return (p * s.last_length,)
end

@inline function _shape_for_step(s::Spread, parent::AbstractArray)
    return _resolve_split_sizes(s.sizes, size(parent, s.parent_dim))
end

@inline function _shape_for_step(s::SpreadRange, ::AbstractArray)
    return _resolve_split_sizes(s.sizes, s.parent_length)
end

@inline _shape_for_step(::Singleton, ::AbstractArray) = (1,)

@inline function _shape_for_step(s::Drop, parent::AbstractArray)
    # Drop is valid if either the parent dim has size 1 (Squeeze case) or
    # the integer index is within bounds (SubArray IntIndex case).
    sz = size(parent, s.parent_dim)
    if s.index == 1
        # Either Squeeze (sz must be 1) or SubArray-derived (already
        # validated when SubArray was constructed). Don't enforce sz==1
        # here because compose may have folded a SubArray's IntIndex into
        # this Drop with index=1, and the parent dim's size may exceed 1.
        # The view emitted by materialize uses index 1, which is always
        # bounds-valid for sz >= 1.
        sz >= 1 || throw(DimensionMismatch(
            "Drop at parent dim $(s.parent_dim): size $sz < 1"
        ))
    else
        1 <= s.index <= sz || throw(BoundsError(
            "Drop at parent dim $(s.parent_dim) uses index $(s.index); valid range 1:$sz"
        ))
    end
    return ()
end

# Resolve a sizes tuple containing at most one Colon against the input
# extent. Returns a tuple of Ints.
@inline function _resolve_split_sizes(sizes::Tuple{Vararg{Union{Int,Colon}}},
                                      input_extent::Int)
    known_prod = 1
    colon_idx = 0
    @inbounds for i in 1:length(sizes)
        si = sizes[i]
        if si isa Colon
            colon_idx == 0 || throw(ArgumentError(
                "Spread can have at most one Colon in sizes"
            ))
            colon_idx = i
        else
            si > 0 || throw(ArgumentError(
                "Spread sizes must be positive; got sizes[$i] = $si"
            ))
            known_prod *= si
        end
    end
    if colon_idx == 0
        known_prod == input_extent || throw(DimensionMismatch(
            "Spread sizes product $known_prod ≠ input extent $input_extent"
        ))
        return _ints(sizes)
    end
    input_extent % known_prod == 0 || throw(DimensionMismatch(
        "Spread sizes incompatible with input extent: $input_extent " *
        "not divisible by $known_prod"
    ))
    missing_size = input_extent ÷ known_prod
    missing_size > 0 || throw(ArgumentError(
        "Spread inferred non-positive Colon size: $missing_size"
    ))
    return ntuple(i -> i == colon_idx ? missing_size : sizes[i]::Int,
                  length(sizes))
end

@inline _ints(t::Tuple{Vararg{Int}}) = t
@inline _ints(t::Tuple) = ntuple(i -> t[i]::Int, length(t))
