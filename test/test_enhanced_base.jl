@testset "Enhanced Base" begin

    @testset "dropdims" begin
        A = reshape(collect(1:24), 4, 1, 3, 1, 2)

        @test Rewrap.dropdims(A; dims=2) == reshape(A, 4, 3, 1, 2)
        @test Rewrap.dropdims(A; dims=4) == reshape(A, 4, 1, 3, 2)
        @test Rewrap.dropdims(A; dims=(2, 4)) == reshape(A, 4, 3, 2)
        @test Rewrap.dropdims(A; dims=(4, 2)) == reshape(A, 4, 3, 2)

        B = reshape(collect(1:6), 1, 2, 3, 1)
        @test Rewrap.dropdims(B; dims=(1, 4)) == reshape(B, 2, 3)

        x = view(reshape(collect(1:12), 4, 1, 3), :, 1:1, :)
        y = Rewrap.dropdims(x; dims=2)
        @test y == reshape(x, 4, 3)
        @test _shares_storage(y, parent(x))
    end

    @testset "vec" begin
        A = [1 3 5; 2 4 6]
        @test Rewrap.vec(A) == [1, 2, 3, 4, 5, 6]

        x = view(A, :, 1:2)
        y = Rewrap.vec(x)
        @test y == [1, 2, 3, 4]
        @test y isa SubArray
        @test _shares_storage(y, parent(x))

        x2 = view(A, 1:2, :)
        y2 = Rewrap.vec(x2)
        @test y2 isa Base.ReshapedArray
        @test y2 == [1, 2, 3, 4, 5, 6]

        B = reshape(collect(1:24), 2, 3, 4)
        @test Rewrap.vec(B) == collect(1:24)
        @test Rewrap.vec(B) isa Array
    end

end

