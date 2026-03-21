function conic_filter_withsolver(data, radius, grid)
    kernel = Kernel.conickernel(grid, radius)

    padprob = Pad.PaddingProblem(;
        data,
        boundary = Pad.BoundaryPadding(size(kernel) .- 1, size(kernel) .- 1),
    )
    padalg = Pad.DefaultPaddingAlgorithm()
    padsolver = init(padprob, padalg)
    padsol = solve!(padsolver)

    convprob = Convolve.DiscreteConvolutionProblem(;
        data = padsol.value,
        kernel,
    )

    convalg = Convolve.FFTConvolution()
    convsolver = init(convprob, convalg)
    convsol = solve!(convsolver)

    depadprob = Pad.PaddingProblem(;
        data = convsol.value,
        boundary = Pad.Inner(size(kernel) .- 1, size(kernel) .- 1),
    )
    depadalg = Pad.DefaultPaddingAlgorithm()
    depadsolver = init(depadprob, depadalg)
    depadsol = solve!(depadsolver)

    return depadsol.value, (padsolver, convsolver, depadsolver)
end

conic_filter(args...) = conic_filter_withsolver(args...)[1]

function ssp_withsolver(alg, rho_filtered, beta, eta, grid)
    target_points = vec(collect(Iterators.product(grid...)))
    prob = Project.ProjectionProblem(;
        rho_filtered,
        grid,
        target_points,
        beta,
        eta,
    )
    solver = init(prob, alg)
    sol = solve!(solver)
    return reshape(sol.value, size(rho_filtered)), solver
end

ssp1_linear(args...; kws...) = ssp_withsolver(Project.SSP1_linear(; kws...), args...)[1]
ssp1(args...; kws...) = ssp_withsolver(Project.SSP1(; kws...), args...)[1]
ssp2(args...; kws...) = ssp_withsolver(Project.SSP2(; kws...), args...)[1]
