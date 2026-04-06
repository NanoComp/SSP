"""
    ssp_constrained_paper.jl

Replicates Fig. 1 from [1].

[1]: G. Romano, R. Arrieta, and S. G. Johnson, [“Differentiating through binarized topology changes: Second-order subpixel-smoothed projection,”](http://arxiv.org/abs/2601.10737) arXiv.org e-Print archive, 2601.10737, January 2026.
"""

using SSP: conic_filter, ssp2, constraint_void, constraint_solid
using Plots

const Nx = Ny = 1024
const grid = (
    range(0, 1, length=Nx),
    range(0, 1, length=Ny),
)

# filter
const conic_radius = 0.3125
# SSP projection
const beta = Inf64
const eta = 0.5 
# Geometric constraints
const constraint_threshold = 1e-8   # ϵ

function latent_to_projected_width(h, conic_radius)
    # Eq. (A9) from [1]
    hnorm = h / conic_radius
    if 0 <= hnorm < 2-sqrt(2)
        return 0.0
    elseif 2-sqrt(2) <= hnorm < 1.0
        return sqrt(4 * hnorm - hnorm^2 - 2.0)
    elseif hnorm >= 1.0
        return float(hnorm)
    else
        throw(DomainError(hnorm))
    end
end

function get_honey(w, target_length)
    # w: width of latent design
    # target_length: target length for the constraint
    # return: value of constraint
    design_vars = [Float64(abs(y - 0.5) ≤ w/2) for (x, y) in Iterators.product(grid...)]
    rho_filtered = conic_filter(design_vars, conic_radius, grid)
    rho_projected = ssp2(rho_filtered, beta, eta, grid)
    gs = constraint_solid(rho_filtered, rho_projected, grid, target_length, conic_radius; constraint_threshold)
    # 'constraint_solid' returns (g-ϵ)/ϵ but we want g
    gs = gs*constraint_threshold + constraint_threshold  # de-normalize constraint value
    return gs
end

# compute constraint vs. latent strip width h
# for various target lengths
hrange = range(0.0, 2.0, length=50) .* conic_radius    # latent strip widths
data = Dict()
for target_length in [0.25, 0.5, 1.0, 1.5, 1.75] .* conic_radius
    println("Processing target length: ", target_length)
    data[target_length] = [get_honey(h,target_length) for h in hrange]
end

# we're interested in plotting constraint vs. *projected* strip width w(h)
plot(xlabel="Normalized width w/radius", 
     ylabel="Constraint Value", 
     yscale=:log10,
     xlims=(-0.1, 2.0),
     ylims=(1e-32, 1e0),
     xticks=[0.25*i for i in 0:8],
     yticks=[10^(i) for i in -32.0:4.0:0.0],
     legend=:bottomleft)

for target_length in keys(data)
    # for a semilogy plot get rid of nonpositive values
    data_target_length = data[target_length]
    mask = data_target_length .> 0.0
    data_mask = data_target_length[mask]
    hmask = hrange[mask]  # latent strip widths
    wmask = latent_to_projected_width.(hmask, conic_radius)  # projected strip widths

    plot!(wmask, data_mask, label="target length / radius: $(target_length/conic_radius)")
end
savefig("constraint_plot.png")