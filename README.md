# Rewrap.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/Rewrap.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Rewrap.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Rewrap.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Rewrap.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Rewrap.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Rewrap.jl)

Rewrap makes lazy wrappers more tractable, and double wrappers less common, by using compile-time type information about the structure of operations to generate custom code at no runtime cost.

The primitives in this package support downstream packages such as [Einops.jl](https://github.com/MurrellGroup/Einops.jl), whose declarative patterns can leverage these rewrapping optimizations.

## Motivation

Julia infamously struggles with "double wrappers", particularly on GPUs, triggering generic fallbacks that use scalar indexing. This can make users wary when working with views and lazy permutations. In the case of `reshape`, the *structure* of the new shape relative to the old shape is completely neglected, and an array may for example become a reshape of a view:

```julia
julia> x = rand(3, 4, 2);

julia> reshape(x, size(x, 1), :) isa Array # same type, no copy
true

julia> y = view(x, 1:2, :, :);

julia> reshape(y, size(y, 1), :) isa Base.ReshapedArray
true
```

We use `size(y, 1)` in our reshape, but despite preserving the first dimension (the one dimension only partially sliced) it evaluates to an integer at runtime, and Julia has no way of knowing that it represents preserving the first dimension.

The size could in theory be constant-propagated [if the size wasn't dynamic](https://github.com/JuliaArrays/FixedSizeArrays.jl), or [if the size is embedded in the type](https://github.com/JuliaArrays/StaticArrays.jl). But integers lack *intent*, and are hard to track.

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

### Local Reshape Operations

These operations are passed to `reshape` and consume/emit a specific number of dimensions:

| Operation | Description |
|-----------|-------------|
| `Keep(N)` | Keep `N` dimensions unchanged (default: 1). `..` becomes `Keep(..)`. |
| `Merge(N)` | Merge `N` dimensions into one. `:` becomes `Merge(..)`. |
| `Split(N, sizes)` | Split `N` dimensions into multiple, with sizes as a tuple of integers and at most one `:`. |
| `Squeeze(N)` | Remove `N` singleton dimensions (default: 1). |
| `Unsqueeze(M)` | Add `M` singleton dimensions (default: 1). |
| `Resqueeze(N => M)` | Turn `N` singleton dimensions into `M` singleton dimensions. |

### Global Axis Operations

These are callable structs that transform arrays:

| Operation | Description |
|-----------|-------------|
| `Reshape(ops, x)` | Apply a tuple of local reshape operations to array `x`. |
| `Permute(perm)` | Permute axes, unwrapping existing `PermutedDimsArray` when possible. |
| `Reduce(f; dims)` | Reduce over dimensions, unwrapping lazy permutes when only reduced dims were permuted. |
| `Repeat(repeats)` | Repeat array along dimensions, pushing through `PermutedDimsArray` to avoid scalar indexing. |

### Enhanced Base Functions

Rewrap also provides optimized versions of common operations:

- `Rewrap.reshape(x, ops...)` — reshape with full Rewrap semantics (no type piracy concerns)
- `Base.reshape(x, ops...)` — reshape with Rewrap semantics (must include a `LocalReshape`)
- `Rewrap.dropdims(x; dims)` — drop singleton dimensions while preserving wrapper types
- `Rewrap.vec(x)` — flatten to vector, preserving views when possible

## Limitations

- Operations won't always rewrap, and may reshape silently if the operation is possible but not without double-wrapping:
```julia
julia> reshape(z, Keep(), :)
2×6 reshape(view(::Array{Int64, 3}, :, 1:3, :), 2, 6) with eltype Int64:
 1  3  5  13  15  17
 2  4  6  14  16  18
```
- `..` and `:` alone won't use Rewrap.jl, as defining such methods would be type piracy. In these cases, `Keep(..)` and `Merge(..)` should be used instead.
- In the `Split` example above, certain divisibility requirements are imposed on the parent dimension.

## Installation

```julia
using Pkg
Pkg.add("Rewrap")
```

## Contributing

Rewrap uses generated functions with compile-time type analysis to produce specialized code for each reshape pattern. We welcome ideas for making the implementation more modular — please feel free to open an issue or pull request.
