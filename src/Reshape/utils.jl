is_ellipsis(::Type{Ellipsis}) = true
is_ellipsis(::Ellipsis) = true
is_ellipsis(_) = false

@inline function _split_sizes_dynamic(sizes::NTuple{M,IntOrColon}, input_prod::Int) where {M}
    known_prod = 1
    colon_idx = 0
    @inbounds for i in 1:M
        si = sizes[i]
        if si isa Colon
            colon_idx == 0 || throw(ArgumentError("Split can have at most one Colon in sizes"))
            colon_idx = i
        else
            si > 0 || throw(ArgumentError("Split sizes must be positive; got sizes[$i] = $si"))
            known_prod *= si
        end
    end

    if colon_idx == 0
        return sizes::NTuple{M,Int}
    end

    input_prod % known_prod == 0 || throw(DimensionMismatch(
        "Split sizes incompatible with input size: input_prod=$input_prod not divisible by known_prod=$known_prod"
    ))
    missing = input_prod ÷ known_prod
    missing > 0 || throw(ArgumentError("Split inferred non-positive Colon size: $missing"))
    return ntuple(i -> i == colon_idx ? missing : sizes[i], Val(M))
end

@generated function split_sizes(op::Split{N,M,T}, input_prod::Int) where {N,M,T}
    # Fast path when the Colon position is known from the tuple type T (e.g. Tuple{Int,Colon}).
    if !(T <: Tuple)
        return :(_split_sizes_dynamic(op.sizes, input_prod))
    end

    tps = T.parameters
    if length(tps) != M || any(tp -> tp isa Core.TypeofVararg, tps)
        return :(_split_sizes_dynamic(op.sizes, input_prod))
    end

    colon_pos = 0
    for (i, tp) in enumerate(tps)
        if tp <: Colon
            colon_pos == 0 || return :(throw(ArgumentError("Split can have at most one Colon in sizes")))
            colon_pos = i
        end
    end

    if colon_pos == 0
        # No Colon in the type; just validate positivity at runtime and return sizes.
        checks = Any[]
        prod_expr = :(1)
        for i in 1:M
            push!(checks, :(op.sizes[$i] > 0 || throw(ArgumentError("Split sizes must be positive"))))
            prod_expr = :($prod_expr * op.sizes[$i])
        end
        return quote
            $(checks...)
            $prod_expr == input_prod || throw(DimensionMismatch(
                "Split sizes incompatible with input size"
            ))
            return op.sizes
        end
    end

    # If any non-colon position isn't an Int type, fall back (can't make return type concrete).
    for (i, tp) in enumerate(tps)
        i == colon_pos && continue
        tp <: Int || return :(_split_sizes_dynamic(op.sizes, input_prod))
    end

    known_checks = Any[]
    known_prod_expr = :(1)
    for i in 1:M
        i == colon_pos && continue
        push!(known_checks, :(op.sizes[$i] > 0 || throw(ArgumentError("Split sizes must be positive"))))
        known_prod_expr = :($known_prod_expr * op.sizes[$i])
    end

    tuple_expr = Expr(:tuple, (i == colon_pos ? :missing : :(op.sizes[$i]) for i in 1:M)...)

    return quote
        $(known_checks...)
        known_prod = $known_prod_expr
        input_prod % known_prod == 0 || throw(DimensionMismatch(
            "Split sizes incompatible with input size: input_prod=$input_prod not divisible by known_prod=$known_prod"
        ))
        missing = input_prod ÷ known_prod
        missing > 0 || throw(ArgumentError("Split inferred non-positive Colon size: $missing"))
        return $tuple_expr
    end
end

@inline keep_sizes(x::AbstractArray, offset::Int, ::Val{n}) where {n} =
    ntuple(i -> size(x, offset + i), Val(n))

@inline ones_sizes(::Val{n}) where {n} =
    ntuple(_ -> 1, Val(n))

@inline function _dimprod_expr(xsym::Symbol, in_dim::Int, n_in::Int)
    n_in == 0 && return 1
    n_in == 1 && return Expr(:call, :size, xsym, in_dim + 1)
    return Expr(:call, :*, (Expr(:call, :size, xsym, in_dim + j) for j in 1:n_in)...)
end

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
