# Helpers for the Reshape constructor's parser pass + the few legacy
# specializations that still inspect op-tuple types directly.
# (Most of the value-side helpers from the pre-Layout codebase —
# `split_sizes`, `keep_sizes`, `ones_sizes`, `_dimprod_expr` — were dead
# after the SubArray/PDA/AbstractArray rewrites and have been deleted.)

is_ellipsis(::Type{Ellipsis}) = true
is_ellipsis(::Ellipsis) = true
is_ellipsis(_) = false

function _ops_has_ellipsis(op_types)
    for op in op_types
        (is_ellipsis(ndims_in(op)) || is_ellipsis(ndims_out(op))) && return true
    end
    return false
end

function _ops_total_in(op_types)
    total = 0
    for op in op_types
        n = ndims_in(op)
        is_ellipsis(n) && return nothing
        total += n
    end
    return total
end

function _is_intcolon(T::Type)
    T <: Int && return true
    T <: Colon && return true
    return false
end

# Detect user shorthand in the raw op-tuple type:
#   - lone `:` (single Colon, no other Int/Colon) → produced as `Merge(..)`
#   - contiguous Int/Colon run (≥ 2 entries, or 1 entry that's an Int) →
#     fused into a single `Split(.., (sizes,))`
# Returns `(op_types, kind)` where `kind` is one of
#   `nothing`                       — no shorthand to rewrite
#   `:lone_colon  => idx`           — Colon at position `idx` becomes Merge
#   `:split_sequence => (start, stop)` — that range becomes one Split
function _preprocess_op_types(op_types)
    n = length(op_types)
    n == 0 && return op_types, nothing

    runs = Tuple{Int,Int}[]
    i = 1
    while i <= n
        if _is_intcolon(op_types[i])
            start = i
            while i <= n && _is_intcolon(op_types[i])
                i += 1
            end
            push!(runs, (start, i - 1))
        else
            i += 1
        end
    end

    has_ellipsis = any(T -> is_ellipsis(T), op_types)

    isempty(runs) && return op_types, nothing

    if length(runs) == 1 && runs[1][1] == runs[1][2] && op_types[runs[1][1]] <: Colon
        has_ellipsis && throw(ArgumentError("At most one Colon or Ellipsis is allowed"))
        return op_types, :lone_colon => runs[1][1]
    end

    length(runs) == 1 || throw(ArgumentError("Int/Colon must form a single contiguous sequence"))
    has_ellipsis && throw(ArgumentError("Cannot mix Ellipsis (..) with Int/Colon sequence"))

    start, stop = runs[1]
    colon_count = count(i -> op_types[i] <: Colon, start:stop)
    colon_count <= 1 || throw(ArgumentError("Split can have at most one Colon in sizes"))

    return op_types, :split_sequence => runs[1]
end

function _build_split_type(op_types, start::Int, stop::Int)
    M = stop - start + 1
    size_types = Any[op_types[i] for i in start:stop]
    T = Tuple{size_types...}
    return Split{.., M, T}
end

function _build_split_expr(start::Int, stop::Int)
    sizes_expr = Expr(:tuple, [:(ops[$i]) for i in start:stop]...)
    return :(Split(.., $sizes_expr))
end
