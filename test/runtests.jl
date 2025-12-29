using Rewrap
using LinearAlgebra
using Test

include("utils.jl")

@testset "Rewrap.jl" begin
    include("test_reshape.jl")
    include("test_permute.jl")
    include("test_repeat.jl")
    include("test_reduce.jl")
    include("test_enhanced_base.jl")
end
