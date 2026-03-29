using Test

@testset "aqua" include("aqua.jl")
@testset "pad" include("pad.jl")
@testset "kernel" include("kernel.jl")
@testset "convolve" include("convolve.jl")
@testset "interpolate" include("interpolate.jl")
@testset "project" include("project.jl")
@testset "pythonic_api" include("pythonic_api.jl")
