function _subarray_reshape_codegen(T, N::Int, M::Int, op_types::Core.SimpleVector, index_types::Core.SimpleVector)
    fallback() = :(invoke(r, Tuple{AbstractArray{$T,$N}}, x))

    _ops_has_ellipsis(op_types) && return fallback()

    total_in = _ops_total_in(op_types)
    (total_in === nothing || total_in != N) && return fallback()

    subdim_to_parent = [i for (i, t) in enumerate(index_types) if !(t <: Integer)]
    length(subdim_to_parent) == N || return fallback()

    parent_ops = Any[]
    view_inds = Any[]
    pd = 1
    subdim = 0

    function drain_integer_dims_to!(target_pd::Int)
        while pd < target_pd
            index_types[pd] <: Integer || return false
            push!(parent_ops, :(Keep(1)))
            push!(view_inds, :(inds[$pd]))
            pd += 1
        end
        return true
    end

    function require_consecutive_slices!(n_in::Int)
        expected_pd = subdim_to_parent[subdim + 1]
        drain_integer_dims_to!(expected_pd) || return false
        pd == expected_pd || return false
        for j in 0:(n_in - 1)
            subdim_to_parent[subdim + 1 + j] == expected_pd + j || return false
            index_types[expected_pd + j] <: Base.Slice || return false
        end
        return true
    end

    function consume_view_dim_keep!()
        subdim += 1
        expected_pd = subdim_to_parent[subdim]
        drain_integer_dims_to!(expected_pd) || return false
        pd == expected_pd || return false
        push!(parent_ops, :(Keep(1)))
        push!(view_inds, :(inds[$pd]))
        pd += 1
        return true
    end

    while pd <= length(index_types) && index_types[pd] <: Integer
        push!(parent_ops, :(Keep(1)))
        push!(view_inds, :(inds[$pd]))
        pd += 1
    end

    for (k, opT) in enumerate(op_types)
        n_in = ndims_in(opT)
        m_out = ndims_out(opT)

        if opT <: Keep
            for _ in 1:n_in
                consume_view_dim_keep!() || return fallback()
            end

        elseif opT <: Merge
            if n_in == 0
                push!(parent_ops, :(ops[$k]))
                push!(view_inds, :(:))
            elseif n_in == 1
                consume_view_dim_keep!() || return fallback()
            else
                expected_pd = subdim_to_parent[subdim + 1]
                drain_integer_dims_to!(expected_pd) || return fallback()
                pd == expected_pd || return fallback()

                for j in 0:(n_in - 2)
                    subdim_to_parent[subdim + 1 + j] == expected_pd + j || return fallback()
                    index_types[expected_pd + j] <: Base.Slice || return fallback()
                end

                last_j = n_in - 1
                subdim_to_parent[subdim + 1 + last_j] == expected_pd + last_j || return fallback()
                last_idx_type = index_types[expected_pd + last_j]

                push!(parent_ops, :(ops[$k]))

                if last_idx_type <: Base.Slice
                    push!(view_inds, :(:))
                elseif last_idx_type <: Base.OneTo
                    last_pd = expected_pd + last_j
                    prod_parts = [:(size(parent(x), $(expected_pd + j))) for j in 0:(n_in - 2)]
                    prod_expr = length(prod_parts) == 1 ? prod_parts[1] : Expr(:call, :*, prod_parts...)
                    push!(view_inds, :(Base.OneTo(length(inds[$last_pd]) * $prod_expr)))
                elseif last_idx_type <: UnitRange{<:Integer}
                    last_pd = expected_pd + last_j
                    prod_parts = [:(size(parent(x), $(expected_pd + j))) for j in 0:(n_in - 2)]
                    prod_expr = length(prod_parts) == 1 ? prod_parts[1] : Expr(:call, :*, prod_parts...)
                    push!(view_inds, :(let r = inds[$last_pd], prod = $prod_expr
                        ((first(r) - 1) * prod + 1):(last(r) * prod)
                    end))
                else
                    return fallback()
                end

                pd += n_in
                subdim += n_in
            end

        elseif opT <: Split
            n_in == 0 && return fallback()
            if n_in == 1 && index_types[pd] <: UnitRange{<:Integer}
                expected_pd = subdim_to_parent[subdim + 1]
                drain_integer_dims_to!(expected_pd) || return fallback()
                if pd == expected_pd
                    pd_idx = pd
                    subdim += 1

                    if m_out == 1
                        push!(parent_ops, :(Keep(1)))
                        push!(view_inds, :(let r = inds[$pd_idx]
                            sizes = ops[$k].sizes
                            want = sizes[1]
                            want isa Int || throw(ArgumentError("Split expects an Int size"))
                            length(r) == want || throw(DimensionMismatch(
                                string("Split expects a UnitRange of length ", want, ", got length ", length(r))
                            ))
                            r
                        end))
                        pd += 1
                        continue
                    end

                    # m_out >= 2: commute view ∘ Split(1, sizes) to Split(1, (n, sizes[2:end-1]..., :)) ∘ view
                    # with runtime alignment assertions (throws on failure).
                    tuple_parts = Any[:n]
                    if m_out > 2
                        for j in 2:(m_out - 1)
                            push!(tuple_parts, :(sizes[$j]))
                        end
                    end
                    push!(tuple_parts, :(:))
                    sizes_tuple_expr = Expr(:tuple, tuple_parts...)

                    push!(parent_ops, :(let sizes = ops[$k].sizes
                        n = sizes[1]
                        n isa Int || throw(ArgumentError("Split expects an Int size in first position"))
                        n > 0 || throw(ArgumentError("Split sizes must be positive; got n=$n"))
                        $(m_out > 2 ? :(for j in 2:$(m_out - 1)
                            sj = sizes[j]
                            sj isa Int || throw(ArgumentError("Split expects Int sizes in intermediate positions"))
                            sj > 0 || throw(ArgumentError("Split sizes must be positive; got sizes[$j]=$sj"))
                        end) : nothing)
                        Split(1, $sizes_tuple_expr)
                    end))

                    for _ in 1:(m_out - 1)
                        push!(view_inds, :(:))
                    end

                    push!(view_inds, :(let r = inds[$pd_idx]
                        sizes = ops[$k].sizes
                        n = sizes[1]
                        n isa Int || throw(ArgumentError("Split expects an Int size in first position"))
                        n > 0 || throw(ArgumentError("Split sizes must be positive; got n=$n"))

                        length(r) % n == 0 || throw(DimensionMismatch(
                            string("UnitRange must select whole Split blocks of size ", n, "; got ", r)
                        ))
                        (first(r) - 1) % n == 0 || throw(DimensionMismatch(
                            string("UnitRange must select whole Split blocks of size ", n, "; got ", r)
                        ))
                        last(r) % n == 0 || throw(DimensionMismatch(
                            string("UnitRange must select whole Split blocks of size ", n, "; got ", r)
                        ))

                        b1 = (first(r) - 1) ÷ n + 1
                        b2 = last(r) ÷ n
                        nb = length(r) ÷ n

                        middle_prod = 1
                        if $m_out > 2
                            for j in 2:$(m_out - 1)
                                sj = sizes[j]
                                sj isa Int || throw(ArgumentError("Split expects Int sizes in intermediate positions"))
                                middle_prod *= sj
                            end
                        end

                        last_size = sizes[$m_out]
                        if last_size isa Int
                            expected_nb = middle_prod * last_size
                            nb == expected_nb || throw(DimensionMismatch(
                                string(
                                    "Split expects ", expected_nb,
                                    " block(s) but UnitRange ", r,
                                    " selects ", nb,
                                    " block(s) with block size ", n,
                                )
                            ))
                        elseif !(last_size isa Colon)
                            throw(ArgumentError("Split expects last size to be Int or Colon"))
                        end

                        if middle_prod == 1
                            b1:b2
                        else
                            (b1 - 1) % middle_prod == 0 || throw(DimensionMismatch(
                                string(
                                    "UnitRange ", r,
                                    " is not aligned to full super-tiles of size ", middle_prod,
                                    " (in blocks of size ", n, ")"
                                )
                            ))
                            b2 % middle_prod == 0 || throw(DimensionMismatch(
                                string(
                                    "UnitRange ", r,
                                    " is not aligned to full super-tiles of size ", middle_prod,
                                    " (in blocks of size ", n, ")"
                                )
                            ))
                            ((b1 - 1) ÷ middle_prod + 1):(b2 ÷ middle_prod)
                        end
                    end))

                    pd += 1
                    continue
                end
            end

            require_consecutive_slices!(n_in) || return fallback()
            push!(parent_ops, :(ops[$k]))
            for _ in 1:m_out
                push!(view_inds, :(:))
            end
            pd += n_in
            subdim += n_in

        elseif opT <: Resqueeze
            if n_in == 0
                push!(parent_ops, :(ops[$k]))
                for _ in 1:m_out
                    push!(view_inds, :(:))
                end
            elseif m_out == 0
                require_consecutive_slices!(n_in) || return fallback()
                push!(parent_ops, :(ops[$k]))
                pd += n_in
                subdim += n_in
            else
                return fallback()
            end
        else
            return fallback()
        end
    end

    subdim == N || return fallback()

    while pd <= length(index_types)
        index_types[pd] <: Integer || return fallback()
        push!(parent_ops, :(Keep(1)))
        push!(view_inds, :(inds[$pd]))
        pd += 1
    end

    Nparent = length(index_types)
    parent_ops_tuple = Expr(:tuple, parent_ops...)
    view_inds_tuple = Expr(:tuple, view_inds...)

    return quote
        ops = r.ops
        inds = parentindices(x)
        parent_ops = $parent_ops_tuple
        parent_r = resolve(parent_ops, Val($Nparent))
        rp = parent_r(parent(x))
        view(rp, $view_inds_tuple...)
    end
end

@generated function (r::Reshape{OpsT,N,M})(
    x::SubArray{T,N,P,I}
) where {OpsT,N,M,T,P,I}
    _subarray_reshape_codegen(T, N, M, OpsT.parameters, I.parameters)
end
