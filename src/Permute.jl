is_identity_perm(perm::NTuple{N,Int}) where N = perm === ntuple(identity, Val(N))

"""
    Permute(perm::NTuple{N,Int})

Permute the axes of an array according to the given permutation.
The resulting array may be a copy, or the same array if the permutation is the identity.

Can be applied to arrays with `N` dimensions.
"""
struct Permute{perm,N} <: GlobalAxisOp{N,N} end

function Permute(perm::NTuple{N,Int}) where N
    all(in(perm), 1:N) || throw(ArgumentError("Permutation must be a permutation of 1:$N"))
    return Permute{perm,N}()
end

function (::Permute{perm,N})(x::AbstractArray{<:Any,N}) where {perm,N}
    is_identity_perm(perm) ? x : permutedims(x, perm)
end

@generated function (::Permute{perm2,N})(x::PermutedDimsArray{T,N,perm1}) where {perm2,N,T,perm1}
    is_identity_perm(perm2) && return :(x)
    perm_total = ntuple(i -> perm1[perm2[i]], Val(N))
    if is_identity_perm(perm_total)
        return :(parent(x))
    end
    return :(PermutedDimsArray(parent(x), $(QuoteNode(perm_total))))
end

using LinearAlgebra: Transpose, Adjoint
const AdjOrTrans{T,P} = Union{Transpose{T,P},Adjoint{T,P}}

@generated function (::Permute{perm2,2})(x::AdjOrTrans{<:Real,P}) where {perm2,P}
    is_identity_perm(perm2) ? :(x) : :(parent(x))
end
