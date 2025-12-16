@testset "Repeat" begin

    @testset "PermutedDimsArray" begin
        A = reshape(collect(1:24), 2, 3, 4)
        p = (1, 3, 2)
        x = PermutedDimsArray(A, p)
        y = Repeat((2,))(x)
        @test y isa PermutedDimsArray
        @test y == repeat(Array(x), 2)
    end

    @testset "basic correctness" begin
        A = reshape(collect(1:6), 2, 3)
        y = Repeat((2, 1))(A)
        @test y == repeat(A, 2, 1)
    end

end
