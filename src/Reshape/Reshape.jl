include("LocalReshape.jl")
include("utils.jl")

"""
    Reshape(ops, ::Val{N})
    Reshape(ops, ::AbstractArray{<:Any,N})

Given a tuple of operations and the input array dimensionality,
construct a `Reshape` object that can be used to reshape arrays.

See also [`Rewrap.reshape`](@ref) and [`Base.reshape`](@ref).
"""
struct Reshape{OpsT,N,M} <: GlobalAxisOp{N,M}
    ops::OpsT
end

# The Reshape constructor is a parser: it normalizes user shorthand
# (`:` → `Merge(..)`, contiguous Int/`:` runs → `Split`, bare `..` →
# `Keep(..)`), resolves any remaining `..` against the array's `N`, and
# emits a fully-typed `Reshape{ResolvedOpsT, N, M}`. Three small passes:
#
#   _normalize_shorthand → eliminates `:` / Int-Colon / bare `..`
#   _infer_ellipsis_n    → determines what `..` resolves to (or 0)
#   _resolve_op_types    → produces final (op_types, op_exprs)

@generated function Reshape(ops::OpsT, ::Val{N}) where {OpsT<:Tuple,N}
    op_types, op_exprs = _normalize_shorthand(OpsT.parameters)
    ellipsis_n         = _infer_ellipsis_n(op_types, N)
    resolved_types, resolved_exprs, M =
        _resolve_op_types(op_types, op_exprs, ellipsis_n, N)

    resolved_opsT = Tuple{resolved_types...}
    resolved_ops_expr = Expr(:tuple, resolved_exprs...)
    return :(Reshape{$resolved_opsT,$N,$M}($resolved_ops_expr))
end

# Pass 1 — Replace user shorthand with explicit ellipsis ops.
# `:` (lone) → `Merge(..)`; contiguous Int/Colon run → `Split(.., (sizes,))`;
# bare `..` → `Keep(..)`. Returns parallel vectors of (typeT, exprT) where
# `exprT` reads the value at the right position from the original `ops`.
function _normalize_shorthand(raw_op_types)
    _, kind = _preprocess_op_types(raw_op_types)

    op_types = Any[]
    op_exprs = Any[]

    if kind === nothing
        for (k, opT) in enumerate(raw_op_types)
            push!(op_types, is_ellipsis(opT) ? Keep{..} : opT)
            push!(op_exprs, is_ellipsis(opT) ? :(Keep(..)) : :(ops[$k]))
        end
    elseif kind.first === :lone_colon
        idx = kind.second
        for (k, opT) in enumerate(raw_op_types)
            if k == idx
                push!(op_types, Merge{..})
                push!(op_exprs, :(Merge(..)))
            elseif is_ellipsis(opT)
                push!(op_types, Keep{..})
                push!(op_exprs, :(Keep(..)))
            else
                push!(op_types, opT)
                push!(op_exprs, :(ops[$k]))
            end
        end
    else
        (start, stop) = kind.second
        split_type = _build_split_type(raw_op_types, start, stop)
        split_expr = _build_split_expr(start, stop)
        for (k, opT) in enumerate(raw_op_types)
            if k == start
                push!(op_types, split_type)
                push!(op_exprs, split_expr)
            elseif k > start && k <= stop
                continue                            # already absorbed into split
            elseif is_ellipsis(opT)
                push!(op_types, Keep{..})
                push!(op_exprs, :(Keep(..)))
            else
                push!(op_types, opT)
                push!(op_exprs, :(ops[$k]))
            end
        end
    end

    return op_types, op_exprs
end

# Pass 2 — Walk normalized ops to compute what `..` resolves to.
# Returns the integer `ellipsis_n` (0 if no Ellipsis input was found and
# the fixed input count covers `N` exactly).
function _infer_ellipsis_n(op_types, N::Int)
    ellipsis_seen = false
    fixed_in = 0
    has_ellipsis_out = false

    for opT in op_types
        n_in = ndims_in(opT)
        if is_ellipsis(n_in)
            ellipsis_seen && throw(ArgumentError("At most one op can have Ellipsis input"))
            ellipsis_seen = true
        else
            fixed_in += n_in
        end
        has_ellipsis_out |= is_ellipsis(ndims_out(opT))
    end

    if ellipsis_seen
        n = N - fixed_in
        n >= 0 || throw(ArgumentError("Ops need $fixed_in dims but array has $N"))
        return n
    end

    has_ellipsis_out && throw(ArgumentError("Ellipsis output requires an Ellipsis input"))
    fixed_in == N || throw(ArgumentError("Ops consume $fixed_in dims but array has $N"))
    return 0
end

# Pass 3 — Replace each op's `..` with the resolved Int. Returns parallel
# vectors of resolved (types, exprs) plus the produced output rank `M`.
function _resolve_op_types(op_types, op_exprs, ellipsis_n::Int, N::Int)
    resolved_types = Any[]
    resolved_exprs = Any[]
    consumed = 0
    produced = 0

    for (k, opT) in enumerate(op_types)
        nin  = is_ellipsis(ndims_in(opT))  ? ellipsis_n : ndims_in(opT)
        mout = is_ellipsis(ndims_out(opT)) ? ellipsis_n : ndims_out(opT)
        consumed += nin
        produced += mout

        if opT <: Keep
            push!(resolved_types, Keep{nin})
            push!(resolved_exprs, :(Keep($nin)))
        elseif opT <: Merge
            push!(resolved_types, Merge{nin})
            push!(resolved_exprs, :(Merge($nin)))
        elseif opT <: Split
            T0 = opT.parameters[3]
            M0 = opT.parameters[2]
            mout == M0 || throw(ArgumentError("Split output rank cannot be ellipsis-resolved"))
            push!(resolved_types, Split{nin, M0, T0})
            push!(resolved_exprs, :(Split($nin, $(op_exprs[k]).sizes)))
        elseif opT <: Resqueeze
            push!(resolved_types, Resqueeze{nin, mout})
            push!(resolved_exprs, :(Resqueeze($nin => $mout)))
        else
            throw(ArgumentError("Unknown reshape op type: $opT"))
        end
    end

    consumed == N || throw(ArgumentError("Ops consume $consumed dims but array has $N"))
    return resolved_types, resolved_exprs, produced
end

Reshape(ops, ::AbstractArray{<:Any,N}) where {N} = Reshape(ops, Val(N))

include("plan.jl")
include("specializations/AbstractArray.jl")
include("specializations/PermutedDimsArray.jl")
include("specializations/SubArray.jl")
include("specializations/ReinterpretArray.jl")
