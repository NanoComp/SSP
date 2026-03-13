module Convolve

using FFTW: plan_fft!, plan_bfft!
import SSP: init, solve!, adjoint_solve!

Base.@kwdef struct DiscreteConvolutionProblem{D,K}
    data::D
    kernel::K
end

mutable struct DiscreteConvolutionSolver{D,K,A,C}
    data::D
    const kernel::K
    alg::A
    cacheval::C
end

Base.@kwdef struct FFTConvolution{F,P}
    factors::F=(2,3,5,7)
    plan_kws::P=(;)
end

function init(prob::DiscreteConvolutionProblem, alg::FFTConvolution)

    (; data, kernel) = prob

    N = size(data) # number of target points
    K = size(kernel) # support of kernel
    transformsize = N .+ K .- 1
    fftsize = map(x -> nextprod(alg.factors, x), transformsize)

    fftkernel = zeros(ComplexF64, fftsize...)
    fftconv = zeros(ComplexF64, fftsize...)

    plan_fw = plan_fft!(fftconv; alg.plan_kws...)
    plan_bk = plan_bfft!(fftconv; alg.plan_kws...)

    output = similar(data)
    adj_data = similar(data)
    cacheval = (; fftsize, fftkernel, fftconv, plan_fw, plan_bk, output, adj_data)
    return DiscreteConvolutionSolver(data, kernel, alg, cacheval)
end

function solve!(solver::DiscreteConvolutionSolver)
    conv_solve!(solver, solver.alg)
end

function conv_solve!(solver, ::FFTConvolution)
    (; data, kernel, cacheval) = solver
    (; fftsize, fftkernel, fftconv, plan_fw, plan_bk, output, adj_data) = cacheval

    fill!(fftkernel, zero(eltype(fftkernel)))
    copy!(view(fftkernel, axes(kernel)...), kernel)
    plan_fw * fftkernel

    fill!(fftconv, zero(eltype(fftconv)))
    copy!(view(fftconv, axes(data)...), data)
    plan_fw * fftconv

    fftconv .*= fftkernel ./ prod(fftsize)

    plan_bk * fftconv

    N = size(data) # number of target points
    K = size(kernel) # support of kernel
    target_indices = map((n, k) -> k÷2+1:n+k÷2, N, K)

    elt = eltype(data) <: Real && eltype(kernel) <: Real ? real : identity
    output .= elt.(view(fftconv, target_indices...))

    return (; value = output, tape = nothing)
end

function adjoint_solve!(solver::DiscreteConvolutionSolver, adj_output, tape)
    adjoint_conv_solve!(solver, solver.alg, adj_output, tape)
end

function adjoint_conv_solve!(solver, ::FFTConvolution, adj_output, tape)
    (; data, kernel, cacheval) = solver
    (; fftsize, fftkernel, fftconv, plan_fw, plan_bk, output, adj_data) = cacheval

    fill!(fftkernel, zero(eltype(fftkernel)))
    copy!(view(fftkernel, axes(kernel)...), kernel)
    plan_fw * fftkernel

    N = size(data) # number of target points
    K = size(kernel) # support of kernel
    target_indices = map((n, k) -> k÷2+1:n+k÷2, N, K)

    elt = eltype(data) <: Real && eltype(kernel) <: Real ? real : identity
    fill!(fftconv, zero(eltype(fftconv)))
    view(fftconv, target_indices...) .= elt.(adj_output)
    plan_fw * fftconv

    fftconv .*= conj.(fftkernel) ./ prod(fftsize)

    plan_bk * fftconv

    adj_data .= elt.(view(fftconv, axes(data)...))

    return (; data=adj_data, kernel=nothing)
end
end
