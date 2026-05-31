using Republic: @public

@public supports_fallback

"""
    supports_fallback(x) -> Bool
    supports_fallback(::Type) -> Bool

Whether `x` (or its type) opts into Rewrap's generic, type-agnostic execution path.

Rewrap's global axis operations (`Permute`, `Reduce`, `Repeat`, `Reshape`) have
specialized methods for `AbstractArray` and a handful of lazy wrappers. For any
other type, the generic fallback — built from `size`, `permutedims`, `repeat`,
`Base.reshape`, and reduction functions — is only used if the type declares it is
compatible by defining

    Rewrap.supports_fallback(::Type{<:MyType}) = true

Types that do not opt in get a `MethodError`, exactly as if no generic method
existed. `AbstractArray`s are supported by default.
"""
supports_fallback(x) = supports_fallback(typeof(x))
supports_fallback(::Type) = false
supports_fallback(::Type{<:AbstractArray}) = true
