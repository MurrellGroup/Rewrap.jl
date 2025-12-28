abstract type LocalAxisOp{N,M} <: AxisOp{N,M} end

# scans like a cursor, consuming N axes, and emitting M axes
abstract type LocalReshape{N,M} <: LocalAxisOp{N,M} end

"""
    Keep(N = 1)

Keep the first `N` dimensions. `N` can be an integer or ellipsis (`..`).
`..` is semantically equivalent, and gets converted to `Keep(..)`
when passed to `reshape`.

```jldoctest
julia> x = reshape(collect(1:24), 3, 4, 2);

julia> y = view(x, 2:3, :, :);

julia> reshape(y, Keep(), :)
2×8 view(::Matrix{Int64}, 2:3, :) with eltype Int64:
 2  5  8  11  14  17  20  23
 3  6  9  12  15  18  21  24
```
"""
struct Keep{N} <: LocalReshape{N,N} end
Keep(N::IntOrEllipsis = 1) = Keep{N}()

"""
    Merge(N)

Merge the first `N` dimensions into one. `N` can be an integer or ellipsis (`..`).
`:` is semantically equivalent, and gets converted to `Merge(..)`
when passed to `reshape`.

```jldoctest
julia> x = reshape(collect(1:8), 2, 4);

julia> y = view(x, :, 2:3);

julia> reshape(y, Merge(..))
4-element view(::Vector{Int64}, 3:6) with eltype Int64:
 3
 4  
 5
 6
```
"""
struct Merge{N} <: LocalReshape{N,1} end
Merge(N::IntOrEllipsis) = Merge{N}()

"""
    Split(N, sizes)
    Split{N}(sizes...)
    Split(sizes)

Split the first `N` dimensions into `M` dimensions, with sizes given by a tuple
of integers and at most one colon (`:`).
This can be interpreted as a local reshape operation on the `N` dimensions,
and doesn't have many of the compile time guarantees of the other operations.

```jldoctest
julia> x = reshape(collect(1:16), 8, 2);

julia> y = view(x, 1:4, :);

julia> reshape(y, Split(1, (2, :)), :)
2×2×2 view(::Array{Int64, 3}, :, 1:2, :) with eltype Int64:
[:, :, 1] =
 1  3
 2  4

[:, :, 2] =
  9  11
 10  12
```
"""
struct Split{N,M,T<:NTuple{M,IntOrColon}} <: LocalReshape{N,M}
    sizes::T
end

Split(N::IntOrEllipsis, sizes::T) where {M,T<:NTuple{M,IntOrColon}} =
    Split{N,M,T}(sizes)

Split{N}(sizes::IntOrColon...) where N = Split(N, sizes)

Split(sizes::Tuple{Vararg{IntOrColon}}) = Split(1, sizes)

"""
    Split1(sizes...)

Split the first dimension into `length(sizes)` dimensions, with sizes given by a tuple
of integers and at most one colon (`:`).
This is a shortcut for `Split(1, sizes)` or `Split{1}(sizes...)`.
"""
const Split1 = Split{1}

"""
    Resqueeze(N => M)

Turn `N` singleton dimensions into `M` singleton dimensions.

See also [`Squeeze`](@ref) and [`Unsqueeze`](@ref).

```jldoctest
julia> x = reshape(collect(1:6), 3, 1, 2);

julia> y = view(x, 1:2, :, :);

julia> reshape(y, Keep(), Resqueeze(1 => 2), Keep())
2×1×1×2 reshape(view(::Array{Int64, 3}, 1:2, :, :), 2, 1, 1, 2) with eltype Int64:
[:, :, 1, 1] =
 1
 2

[:, :, 1, 2] =
 4
 5
```
"""
struct Resqueeze{N,M} <: LocalReshape{N,M} end
Resqueeze((N,M)::Pair{<:IntOrEllipsis,Int}) = Resqueeze{N,M}()

"""
    Squeeze(N = 1)

Remove `N` singleton dimensions.

See also [`Resqueeze`](@ref) and [`Unsqueeze`](@ref).

```jldoctest
julia> x = reshape(collect(1:6), 3, 1, 2);

julia> y = view(x, 1:2, :, :);

julia> reshape(y, Keep(), Squeeze(..), Keep())
2×2 view(::Matrix{Int64}, 1:2, :) with eltype Int64:
 1  4
 2  5
```
"""
const Squeeze{N} = Resqueeze{N,0}
Squeeze(N::IntOrEllipsis = 1) = Resqueeze(N => 0)

"""
    Unsqueeze(M = 1)

Add `M` singleton dimensions.

See also [`Resqueeze`](@ref) and [`Squeeze`](@ref).

```jldoctest
julia> x = reshape(collect(1:6), 3, 2);

julia> y = view(x, 1:2, :);

julia> reshape(y, Keep(), Unsqueeze(1), Keep())
2×1×2 view(::Array{Int64, 3}, 1:2, :, :) with eltype Int64:
[:, :, 1] =
 1
 2

[:, :, 2] =
 4
 5
```
"""
const Unsqueeze{M} = Resqueeze{0,M}
Unsqueeze(M::Int = 1) = Resqueeze(0 => M)
