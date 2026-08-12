# Minimal non-AbstractArray tensor implementing the whole-array primitives the
# generic methods are built from, delegating to a wrapped Array (Base semantics).
struct DuckTensor{T,N}
    data::Array{T,N}
end
Base.size(x::DuckTensor) = size(x.data)
Base.size(x::DuckTensor, d::Integer) = size(x.data, d)
Base.ndims(x::DuckTensor) = ndims(x.data)
Base.permutedims(x::DuckTensor, perm) = DuckTensor(permutedims(x.data, perm))
Base.reshape(x::DuckTensor, dims::Tuple{Vararg{Int}}) = DuckTensor(reshape(x.data, dims))
Base.repeat(x::DuckTensor, counts::Integer...) = DuckTensor(repeat(x.data, counts...))
Base.sum(x::DuckTensor; dims) = DuckTensor(sum(x.data; dims))

# A type implementing none of the primitives.
struct NoPrimsTensor end

@testset "Generic (non-AbstractArray) execution" begin
    A = collect(reshape(1.0:24.0, 2, 3, 4))

    @testset "Permute" begin
        y = Permute((3, 1, 2))(DuckTensor(A))
        @test y isa DuckTensor
        @test y.data == permutedims(A, (3, 1, 2))

        x = DuckTensor(A)
        @test Permute((1, 2, 3))(x) === x
    end

    @testset "Reshape" begin
        y = Rewrap.reshape(DuckTensor(A), (Merge(2), Keep(1)))
        @test y isa DuckTensor
        @test y.data == reshape(A, 6, 4)

        z = Rewrap.reshape(DuckTensor(A), (Split(1, (1, 2)), Keep(2)))
        @test z.data == reshape(A, 1, 2, 3, 4)
    end

    @testset "Repeat" begin
        M = collect(reshape(1.0:6.0, 2, 3))
        y = Repeat((1, 2))(DuckTensor(M))
        @test y isa DuckTensor
        @test y.data == repeat(M, 1, 2)
    end

    @testset "Reduce" begin
        y = Reduce(sum; dims=(2,))(DuckTensor(A))
        @test y isa DuckTensor
        @test y.data == sum(A, dims=2)

        x = DuckTensor(A)
        @test Reduce(sum; dims=())(x) === x
    end

    @testset "missing primitive fails loudly" begin
        # Types without the required primitive error at the primitive itself
        # (here: `permutedims(::NoPrimsTensor, perm)` and `ndims(::NoPrimsTensor)`).
        @test_throws MethodError Permute((2, 1))(NoPrimsTensor())
        @test_throws MethodError Repeat((2,))(NoPrimsTensor())
        @test_throws MethodError Reduce(sum; dims=(1,))(NoPrimsTensor())
    end
end
