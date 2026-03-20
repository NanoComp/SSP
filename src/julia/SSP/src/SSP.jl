module SSP

include("definitions.jl")
public init, init!, solve, solve!, adjoint_solve, adjoint_solve!

include("pad.jl")
include("kernel.jl")
include("convolve.jl")
include("interpolate.jl")
include("project.jl")

end
