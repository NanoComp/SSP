module SSP

include("definitions.jl")
public init, init!, solve, solve!, adjoint_solve, adjoint_solve!

include("pad.jl")
public Pad
include("kernel.jl")
public Kernel
include("convolve.jl")
public Convolve
include("interpolate.jl")
public Interpolate
include("project.jl")
public Project
include("constrain.jl")
public Constrain

include("pythonic_api.jl")
public conic_filter, ssp1_linear, ssp1, ssp2

end
