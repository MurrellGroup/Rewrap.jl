# Generic AbstractArray fallback for Reshape application. Lowers the resolved
# op tuple to a Layout.Plan via `_ops_to_plan`, then materializes against the
# array. The specializations for SubArray / PermutedDimsArray /
# ReinterpretArray below this still match first when applicable; they
# `invoke` back into this method for cases they can't fold.

function (r::Reshape{OpsT,N,M})(x::AbstractArray{<:Any,N}) where {OpsT,N,M}
    op_plan = _ops_to_plan(r.ops, Val(N))
    # AbstractArray's `layout` is identity (no cropping anywhere), so the
    # composed Plan's parent extents come straight from `x` and the only
    # operation needed is a single `Base.reshape`. Bypassing `materialize`'s
    # view/reshape branch keeps inference type-stable end-to-end —
    # important for downstream `@inferred` users like Einops.
    return Base.reshape(x, Layout.output_shape(op_plan, x))
end
