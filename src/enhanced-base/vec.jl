"""
    vec(x)

Flatten the array `x` into a vector.

```jldoctest
julia> x = [1 3 5; 2 4 6];

julia> Rewrap.vec(view(x, :, 1:2))
4-element view(::Vector{Int64}, 1:4) with eltype Int64:
 1
 2
 3
 4

julia> Rewrap.vec(view(x, 1:2, :)) # not contiguous!
6-element reshape(view(::Matrix{Int64}, 1:2, :), 6) with eltype Int64:
 1
 2
 3
 4
 5
 6
```
"""
vec(x) = reshape(x, Merge(..))
