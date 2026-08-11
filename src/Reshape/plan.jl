# Lower a resolved Reshape op tuple to a Layout.Plan. Per-op step generation
# dispatches on Keep / Merge / Split / Resqueeze types defined in
# LocalReshape.jl. Used by the Layout-backed `(r::Reshape)(x::AbstractArray)`
# application in specializations/AbstractArray.jl.

# Build a Plan{N} from the (resolved) ops tuple. By the time we reach this
# function, the Reshape constructor's @generated has normalized away
# Ellipsis (..) and lone-Colon / Int-Colon shorthands.
@inline function _ops_to_plan(ops::Tuple, ::Val{N}) where {N}
    Layout.Plan{N}(_steps_for_ops(ops, 1))
end

@inline _steps_for_ops(::Tuple{}, ::Int) = ()
@inline function _steps_for_ops(ops::Tuple, cursor::Int)
    h = _step_for_op(first(ops), cursor)
    return (h..., _steps_for_ops(Base.tail(ops), cursor + _ndims_consumed(first(ops)))...)
end

@inline _ndims_consumed(::Keep{N}) where {N} = N
@inline _ndims_consumed(::Merge{N}) where {N} = N
@inline _ndims_consumed(::Split{N}) where {N} = N
@inline _ndims_consumed(::Resqueeze{N}) where {N} = N

@inline _step_for_op(::Keep{N}, cursor::Int) where {N} =
    ntuple(i -> Layout.Slice(cursor + i - 1), Val(N))

# Merge(0) is degenerate — it consumes zero parent dims and produces one
# size-1 output dim. That's exactly `Singleton`. Encoding it as
# `Group(cursor, 0)` would underflow output_shape's `last_full` lookup.
@inline _step_for_op(::Merge{0}, ::Int) = (Layout.Singleton(),)
@inline _step_for_op(::Merge{N}, cursor::Int) where {N} =
    (Layout.Group(cursor, N),)

@inline function _step_for_op(s::Split{N,M,T}, cursor::Int) where {N,M,T}
    N == 1 || error(
        "Split with N=$N input dims is not yet supported in the Layout IR " *
        "(Phase 1). Tracking issue: extend Spread to take a parent range."
    )
    return (Layout.Spread(cursor, s.sizes),)
end

@inline _step_for_op(::Resqueeze{N,M}, cursor::Int) where {N,M} = (
    ntuple(i -> Layout.Drop(cursor + i - 1), Val(N))...,
    ntuple(_ -> Layout.Singleton(),          Val(M))...,
)
