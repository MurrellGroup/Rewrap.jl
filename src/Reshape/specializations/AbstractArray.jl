function _abstractarray_reshape_codegen(op_types::Core.SimpleVector, N::Int)
    ellipsis_idx = nothing
    fixed_in = 0
    has_ellipsis_out = false
    for (k, op) in enumerate(op_types)
        n_in = ndims_in(op)
        if is_ellipsis(n_in)
            ellipsis_idx !== nothing && throw(ArgumentError("At most one op can have Ellipsis input"))
            ellipsis_idx = k
        else
            fixed_in += n_in
        end
        has_ellipsis_out |= is_ellipsis(ndims_out(op))
    end

    if ellipsis_idx !== nothing
        ellipsis_n = N - fixed_in
        ellipsis_n >= 0 || throw(ArgumentError("Ops need $fixed_in dims but array has $N"))
    else
        has_ellipsis_out && throw(ArgumentError("Ellipsis output requires an Ellipsis input"))
        fixed_in == N || throw(ArgumentError("Ops consume $fixed_in dims but array has $N"))
        ellipsis_n = 0
    end

    shape_parts = Any[]
    checks = Any[]
    in_dim = 0

    for (k, op) in enumerate(op_types)
        n_in = is_ellipsis(ndims_in(op)) ? ellipsis_n : ndims_in(op)
        m_out = is_ellipsis(ndims_out(op)) ? ellipsis_n : ndims_out(op)

        if op <: Keep
            push!(shape_parts, Expr(:..., :(keep_sizes(x, $in_dim, Val($n_in)))))

        elseif op <: Merge
            push!(shape_parts, _dimprod_expr(:x, in_dim, n_in))

        elseif op <: Split
            input_prod = _dimprod_expr(:x, in_dim, n_in)
            push!(shape_parts, Expr(:..., :(split_sizes(ops[$k], $input_prod))))

        elseif op <: Resqueeze
            for j in 1:n_in
                pos = in_dim + j
                push!(checks, :(size(x, $pos) == 1 || throw(DimensionMismatch(
                    string("Resqueeze expects size-1 dim at position ", $pos, ", got ", size(x, $pos))
                ))))
            end
            push!(shape_parts, Expr(:..., :(ones_sizes(Val($m_out)))))
        end

        in_dim += n_in
    end

    shape_tuple = Expr(:tuple, shape_parts...)

    if isempty(checks)
        return :(Base.reshape(x, $shape_tuple))
    end

    return quote
        $(checks...)
        Base.reshape(x, $shape_tuple)
    end
end

@generated function (r::Reshape{OpsT,N,M})(x::AbstractArray{<:Any,N}) where {OpsT,N,M}
    op_types = OpsT.parameters
    body = _abstractarray_reshape_codegen(op_types, N)
    return quote
        ops = r.ops
        $body
    end
end

# Generic fallback for non-AbstractArray types that opt in via `supports_fallback`.
# Uses the same codegen, which is built from `size`, `Base.reshape`, and the
# `keep_sizes`/`split_sizes` helpers — all of which are type-agnostic.
#
# The `supports_fallback` opt-in is checked in this plain (non-`@generated`)
# wrapper rather than inside the generator: types opt in from downstream packages
# and extensions (e.g. cuTile via `Rewrap.supports_fallback(::Type{<:Tile})`),
# whose methods are defined in a later world than this `Reshape` definition. A
# `@generated` generator only sees methods up to its own definition world, so
# querying `supports_fallback` there would never observe those opt-ins and would
# always take the throw branch. As a normal call it resolves at the caller's
# world (with backedges) and constant-folds away for opted-in types.
#
# `@inline` is load-bearing for accelerator backends (e.g. cuTile): they lower by
# walking optimized IR and mapping each call to an intrinsic, and do not recurse
# into an un-inlined functor. Inlining exposes the inner `Base.reshape(x, dims)`
# (which such backends map to a native reshape) instead of an opaque `Reshape`
# functor invoke.
@inline function (r::Reshape{OpsT,N,M})(x) where {OpsT,N,M}
    supports_fallback(x) || throw(MethodError(r, (x,)))
    return _reshape_fallback(r, x)
end

@inline @generated function _reshape_fallback(r::Reshape{OpsT,N,M}, x) where {OpsT,N,M}
    op_types = OpsT.parameters
    body = _abstractarray_reshape_codegen(op_types, N)
    return quote
        ops = r.ops
        $body
    end
end
