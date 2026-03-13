module SSP

include("definitions.jl")
public init, solve, solve!, adjoint_solve, adjoint_solve!

include("pad.jl")
include("kernel.jl")
include("convolve.jl")
include("project.jl")

end
