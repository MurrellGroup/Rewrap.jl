# Rewrap.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/Rewrap.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Rewrap.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Rewrap.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Rewrap.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Rewrap.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Rewrap.jl)

Rewrap makes lazy wrappers more tractable, and double wrappers less common, by using compile-time type information about the structure of operations to generate custom code at no runtime cost.

The primitives in this package support downstream packages such as [Einops.jl](https://github.com/MurrellGroup/Einops.jl), whose declarative patterns can leverage these rewrapping optimizations.

## Motivation

Julia infamously struggles with "double wrappers", particularly on GPUs, triggering generic fallbacks that use scalar indexing. This can make users particularly wary when working with views and lazy permutations. In the case of `reshape`, the *structure* of the new shape relative to the old shape is completely neglected, and an array may for example become a reshape of a view:

```julia
julia> x = rand(3, 4, 2);

julia> reshape(x, size(x, 1), :) isa Array # same type, no copy
true

julia> y = view(x, 1:2, :, :);

julia> reshape(y, size(y, 1), :) isa Base.ReshapedArray
true
```

We use `size(y, 1)` in our reshape, but despite preserving the first dimension (the one dimension only partially sliced) it evaluates to an integer at runtime, and Julia has no way of knowing that it represents preserving the first dimension. The size could in theory be constant-propagated [if the size wasn't dynamic](https://github.com/JuliaArrays/FixedSizeArrays.jl), [or if the size is embedded in the type](https://github.com/JuliaArrays/StaticArrays.jl), but even then, integers alone are not useful once passed through `reshape`.

Rewrap provides types like `Keep`, `Merge`, and `Split` that encode reshape structure at compile-time, enabling rewrapping optimizations.

```julia
julia> using Rewrap

julia> reshape(y, Keep(), :) isa SubArray
true
```

As a more complex example, we can "split" the first dimension into two:

```julia
julia> x = reshape(collect(1:24), 12, 2);

julia> y = view(x, 1:6, :)
6×2 view(::Matrix{Int64}, 1:6, :) with eltype Int64:
 1  13
 2  14
 3  15
 4  16
 5  17
 6  18

julia> z = reshape(y, Split(1, (2, :)), :)
2×3×2 view(::Array{Int64, 3}, :, 1:3, :) with eltype Int64:
[:, :, 1] =
 1  3  5
 2  4  6

[:, :, 2] =
 13  15  17
 14  16  18

julia> reshape(z, :, Keep()) # undo
6×2 view(::Matrix{Int64}, 1:6, :) with eltype Int64:
 1  13
 2  14
 3  15
 4  16
 5  17
 6  18
```

The view is commuted past the reshape: the parent array gets reshaped first, then re-viewed with adjusted indices.

## Features

- `..` (from [EllipsisNotation.jl](https://github.com/SciML/EllipsisNotation.jl)) is replaced with `Keep(..)` when passed to `reshape`.
- `:` can be used like normal, but under the hood it gets replaced by `Merge(..)` when passed to `reshape`.

## Limitations

- Operations won't always rewrap, and may reshape silently if the operation is possible but not without double-wrapping:
```julia
julia> reshape(z, Keep(), :)
2×6 reshape(view(::Array{Int64, 3}, :, 1:3, :), 2, 6) with eltype Int64:
 1  3  5  13  15  17
 2  4  6  14  16  18
```
- Direct arguments of reshape can not be integers when an axis operation is present.
- `..` and `:` alone won't use Rewrap.jl, as defining such methods would be type piracy. In these cases, `Keep(..)` and `Merge(..)` should be used instead.

## Installation

```julia
using Pkg
Pkg.add("Rewrap")
```

## Contributing

At the moment, Rewrap explicitly defines optimizations in big codegen monoliths for generated function specializations, making the source code hard to parse. Ideally it would use a more modular approach. If you have any ideas or suggestions, please feel free to open an issue or a pull request.
