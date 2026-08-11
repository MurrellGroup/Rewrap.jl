"""
    Repeat(repeats::NTuple{K,Int})

Repeats an array along multiple dimensions.

Can be applied to arrays with at most `K` dimensions.
"""
struct Repeat{K} <: GlobalAxisOp{..,..}
    repeats::NTuple{K,Int}
    function Repeat{K}(repeats::NTuple{K,Int}) where K
        all(>=(0), repeats) || throw(ArgumentError("Repeats must be non-negative"))
        return new{K}(repeats)
    end
end
Repeat(repeats::NTuple{K,Int}) where K = Repeat{K}(repeats)

function (op::Repeat{K})(x::AbstractArray{<:Any,N}) where {N,K}
    K <= N || throw(ArgumentError("Repeat dimensions must be less than or equal to the number of dimensions of the array"))
    return repeat(x, op.repeats...)
end

# Push repeats to the parent to avoid iterating fallbacks (e.g. CUDA scalar
# indexing). `iperm` maps parent-dims → output-dims, so we remap repeat
# factors accordingly.
function (op::Repeat{K})(
    x::PermutedDimsArray{T,N,perm,iperm,P},
) where {K,T,N,perm,iperm,P}
    K <= N || throw(ArgumentError(
        "Repeat dimensions must be less than or equal to the number of dimensions of the array"
    ))
    parent_repeats = ntuple(d -> iperm[d] <= K ? op.repeats[iperm[d]] : 1, Val(N))
    y = repeat(parent(x), parent_repeats...)
    return PermutedDimsArray(y, perm)
end
