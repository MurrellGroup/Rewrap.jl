@testset "Reshape" begin

    @testset "Equivalence" begin
        A = reshape(collect(1:24), 4, 3, 2)

        @test reshape(A, Keep(3)) == A
        @test reshape(A, Keep(..)) == A
        @test reshape(A, Merge(2), Keep(1)) == reshape(A, 12, 2)
        @test reshape(A, Split(1, (2, 2)), Keep(2)) == reshape(A, 2, 2, 3, 2)
        @test reshape(A, Split(1, (2, :)), Keep(2)) == reshape(A, 2, 2, 3, 2)
        @test reshape(A, Split(1, (1, 4)), Keep(..)) == reshape(A, 1, 4, 3, 2)
    end

    @testset "Colon notation" begin
        A = reshape(collect(1:24), 4, 3, 2)
        @test reshape(A, Keep(1), :) == reshape(A, 4, :)
    end

    @testset "Ellipsis notation" begin
        A = reshape(collect(1:24), 4, 3, 2)
        @test reshape(A, Split(1, (2, :)), ..) == reshape(A, 2, :, 3, 2)
    end

    @testset "Resqueeze" begin
        A = reshape(collect(1:24), 4, 3, 2)
        x = reshape(A, 1, 4, 3, 2)

        y = reshape(x, Squeeze(1), Keep(..))
        @test y == A
        z = reshape(A, Unsqueeze(1), Keep(..))
        @test z == x
    end

    @testset "Split errors" begin
        A = reshape(collect(1:12), 6, 2)
        @test_throws DimensionMismatch reshape(A, Split(1, (4, :)), Keep(..))
        @test_throws ArgumentError reshape(A, Split(1, (0, :)), Keep(..))
        @test_throws ArgumentError reshape(A, Split(1, (2, :, :)), Keep(..))
    end

    @testset "SubArray wrapper elision" begin
        A = reshape(collect(1:24), 4, 3, 2)
        x = view(A, 1:2, :, :)
        y = reshape(x, Split(1, (1, 2)), Keep(..))
        @test y == reshape(x, 1, 2, 3, 2)
        @test y isa SubArray
        @test _shares_storage(y, parent(x))

        B = reshape(collect(1:16), 8, 2)
        x2 = view(B, 2:3, :)
        @test_throws DimensionMismatch reshape(x2, Split(1, (2, :)), Keep(..))

        C = reshape(collect(1:24), 4, 3, 2)
        x3 = view(C, 1:2, :, :)
        y3 = reshape(x3, Split(1, (2,)), Keep(..))
        @test y3 == x3
        @test y3 isa SubArray
        @test _shares_storage(y3, parent(x3))

        D = reshape(collect(1:32), 8, 2, 2)
        x4 = view(D, 1:4, :, :)
        y4 = reshape(x4, Split(1, (2, 1, 2)), Keep(..))
        @test y4 == reshape(x4, 2, 1, 2, 2, 2)
        @test y4 isa SubArray
        @test _shares_storage(y4, parent(x4))
    end

    @testset "PermutedDimsArray reshape wrapper elision" begin
        A = reshape(collect(1:24), 2, 3, 4)
        x = PermutedDimsArray(A, (3, 1, 2))
        @test reshape(x, Keep(3)) == x
        @test reshape(x, Keep(), :) == PermutedDimsArray(reshape(A, :, 4), (2, 1))
        @test _shares_storage(A, x)
    end
end


