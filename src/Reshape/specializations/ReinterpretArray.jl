# ReinterpretArray dispatch. The Layout IR is element-count-agnostic; it
# can't natively express "this dim's stride is scaled by sizeof(S)/sizeof(T)".
# We keep the wrapper outside the IR: modify the user's ops to apply
# directly to `parent(x)` (accounting for the ratio dim per `check` mode),
# run them through Layout's normal Reshape path, then `reinterpret(...)`
# the result. The choice between `reinterpret(T, …)` and
# `reinterpret(reshape, T, …)` follows whether the ratio dim is preserved
# in the output (Keep on dim 1 in check=true mode preserves it).
#
# Mode reference:
#   `check=false`:  ndims(x) == ndims(parent). Dim 1 of x = parent dim 1
#                   with stride scaled by sizeof(S)/sizeof(T).
#   `check=true` (sizeof(T) < sizeof(S)): ndims(x) = ndims(parent)+1, with
#                   x's dim 1 = the ratio. Other x dims = parent dims 1..N.

function (r::Reshape{OpsT,N,M})(
    x::Base.ReinterpretArray{T,N,S,A,check},
) where {OpsT,N,M,T,S,A,check}
    # All-Keep: identity on x.
    all_keep(r.ops) && return x

    parent_ops = _ra_modify_ops(r.ops, Val(check), T, S)
    parent_ops === nothing && return invoke(r, Tuple{AbstractArray{T,N}}, x)

    parent_N = ndims(A)
    reshaped_parent = Reshape(parent_ops, Val(parent_N))(parent(x))

    # In check=true mode, a leading `Keep` preserves the ratio dim — so we
    # re-add it via `reinterpret(reshape, T, …)`. All other cases (check=false
    # everywhere; check=true with a leading Merge that absorbs the ratio dim)
    # produce a flat-first-dim result, which is exactly `reinterpret(T, …)`.
    if check && first(r.ops) isa Keep
        # `Base.reshape` is qualified because Rewrap exports a `reshape` of
        # its own (in enhanced-base/), which would otherwise shadow Base's
        # — and `reinterpret`'s special second-arg form keys off the
        # function-value identity, not the symbol.
        return reinterpret(Base.reshape, T, reshaped_parent)
    else
        return reinterpret(T, reshaped_parent)
    end
end

@inline all_keep(::Tuple{}) = true
@inline all_keep(ops::Tuple) = first(ops) isa Keep && all_keep(Base.tail(ops))

# check=true: the first op must consume the ratio dim cleanly.
@inline _ra_modify_ops(ops::Tuple, ::Val{true}, ::Type, ::Type) =
    _ra_check_true(first(ops), Base.tail(ops))

@inline _ra_check_true(::Keep{1}, rest::Tuple) = rest
@inline _ra_check_true(::Merge{1}, rest::Tuple) = rest
@inline _ra_check_true(::Merge{N}, rest::Tuple) where {N} = (Merge(N - 1), rest...)
@inline _ra_check_true(_, ::Tuple) = nothing

# check=false: dim 1 of x is parent dim 1 with stride scaled by ratio. Keep
# and Merge starting at dim 1 pass through unchanged. Split(1, sizes) needs
# the first size divisible by ratio.
@inline _ra_modify_ops(ops::Tuple, ::Val{false}, ::Type{T}, ::Type{S}) where {T, S} =
    _ra_check_false(first(ops), Base.tail(ops), T, S)

@inline _ra_check_false(::Keep{1}, rest::Tuple, ::Type, ::Type) =
    (Keep(1), rest...)
@inline _ra_check_false(::Merge{N}, rest::Tuple, ::Type, ::Type) where {N} =
    (Merge(N), rest...)

@inline function _ra_check_false(op::Split{1,Mout,SizesT}, rest::Tuple,
                                  ::Type{T}, ::Type{S}) where {Mout, SizesT, T, S}
    sizes = op.sizes
    first_sz = sizes[1]
    first_sz isa Int || return nothing            # Colon-first → refuse
    ratio = sizeof(S) ÷ sizeof(T)
    first_sz % ratio == 0 || return nothing       # alignment fails
    new_sizes = (first_sz ÷ ratio, Base.tail(sizes)...)
    return (Split(1, new_sizes), rest...)
end

@inline _ra_check_false(_, ::Tuple, ::Type, ::Type) = nothing
