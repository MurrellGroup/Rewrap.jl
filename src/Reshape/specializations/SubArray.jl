# SubArray dispatch routes through the Layout IR. `layout(::SubArray)` reads
# `parentindices` into `Slice` / `Drop` steps; `compose` folds the Reshape
# op-plan over them; `materialize` emits the minimum wrapper chain.
#
# When the IR can't express the result (unsupported index type, refused
# composition), we fall back to `Base.reshape` on the SubArray — which
# produces a `Base.ReshapedArray` (loses wrapper-elision but is
# correctness-preserving).

function (r::Reshape{OpsT,N,M})(x::SubArray{T,N,P,I}) where {OpsT,N,M,T,P,I}
    arr_plan = Layout.layout(x)
    if arr_plan !== nothing
        op_plan = _ops_to_plan(r.ops, Val(N))
        composed = Layout.compose(op_plan, arr_plan)
        if composed !== nothing
            return Layout.materialize(composed, Layout.root_parent(x))
        end
    end
    return invoke(r, Tuple{AbstractArray{T,N}}, x)
end
