# PermutedDimsArray dispatch routes through the Layout IR.
# `layout(::PermutedDimsArray)` reads the permutation as `Slice` steps with
# permuted parent_dim references; `compose` rebinds outer steps through them;
# `materialize` detects the non-natural parent_dim ordering and emits
# `PermutedDimsArray(reshape(parent, natural_shape), perm)` (or just
# `PermutedDimsArray(parent, perm)` when no reshape is needed).

function (r::Reshape{OpsT,N,M})(
    x::PermutedDimsArray{T,N,perm,iperm,P},
) where {OpsT,N,M,T,perm,iperm,P}
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
