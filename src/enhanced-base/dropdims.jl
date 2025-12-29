"""
    dropdims(x; dims)

Drop the specified dimensions from the array `x`.

```jldoctest
julia> x = [1 3 5; 2 4 6;;;]
2×3×1 Array{Int64, 3}:
[:, :, 1] =
 1  3  5
 2  4  6

julia> y = view(x, :, 1:2, :)
2×2×1 view(::Array{Int64, 3}, :, 1:2, :) with eltype Int64:
[:, :, 1] =
 1  3
 2  4

julia> Rewrap.dropdims(y; dims=3)
2×2 view(::Matrix{Int64}, :, 1:2) with eltype Int64:
 1  3
 2  4
```
"""
@constprop function dropdims(
    x::AbstractArray{<:Any,N}; dims::Union{Int,Tuple{Vararg{Int}}}
) where N
    dims′ = dims isa Int ? (dims,) : dims
    ops = ntuple(i -> i in dims′ ? Squeeze() : Keep(), N)
    return reshape(x, ops)
end
