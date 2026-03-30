module Convolve

using FFTW: plan_fft!, plan_bfft!, plan_rfft, plan_brfft, mul!
import SSP: init!, solve!, adjoint_solve!

public DiscreteConvolutionProblem, FFTConvolution

"""
    DiscreteConvolutionProblem(; data, kernel)

Performs a discrete convolution of arrays `data` and `kernel`, returning an array of the same size as `data`.
Here, the data are implicitly zero-padded.
The kernel is assumed to be constant and is not allowed to change after `init`.
"""
Base.@kwdef struct DiscreteConvolutionProblem{D,K}
    data::D
    kernel::K
end

function Base.copy(prob::DiscreteConvolutionProblem)
    newprob = DiscreteConvolutionProblem(;
        data = copy(prob.data),
        kernel = copy(prob.kernel),
    )
    return newprob
end

mutable struct DiscreteConvolutionSolver{D,A,C}
    data::D
    alg::A
    cacheval::C
end

"""
    FFTConvolution(; factors=(2,3,5,7), plan_kws=(;))

Calculate a discrete convolution by extending it to a periodic convolution accelerated via FFTs.
Uses FFTW.jl and pads the transform to the next 7-smooth number by default, controlled by the `factors` keyword.
Additional plan keywords can be passed as a named tuple to `plan_kws`.
"""
Base.@kwdef struct FFTConvolution{F,P}
    factors::F=(2,3,5,7)
    plan_kws::P=(;)
end

function init!(prob::DiscreteConvolutionProblem, alg::FFTConvolution)

    (; data, kernel) = prob
    @assert isreal(zero(eltype(data)))
    @assert isreal(zero(eltype(kernel)))

    N = size(data) # number of target points
    K = size(kernel) # support of kernel
    transformsize = N .+ K .- 1
    fftsize = map(x -> nextprod(alg.factors, x), transformsize)

    signal = zeros(float(promote_type(eltype(data), eltype(kernel))), fftsize...)

    plan_fw = plan_rfft(signal; alg.plan_kws...)
    fftconv = plan_fw * signal
    
    # precompute the fft of the kernel
    fill!(signal, zero(eltype(signal)))
    copy!(view(signal, axes(kernel)...), kernel)
    fftkernel = plan_fw * signal

    plan_bk = plan_brfft(fftconv, fftsize[1]; alg.plan_kws...)
    conv = plan_bk * fftconv

    output = similar(data)
    adj_data = similar(data)
    cacheval = (; fftsize, signal, K, fftkernel, fftconv, conv, plan_fw, plan_bk, output, adj_data)
    return DiscreteConvolutionSolver(data, alg, cacheval)
end

function solve!(solver::DiscreteConvolutionSolver)
    conv_solve!(solver, solver.alg)
end

function conv_solve!(solver, ::FFTConvolution)
    (; data, cacheval) = solver
    (; fftsize, signal, K, fftkernel, fftconv, conv, plan_fw, plan_bk, output, adj_data) = cacheval

    fill!(signal, zero(eltype(signal)))
    copy!(view(signal, axes(data)...), data)
    mul!(fftconv, plan_fw, signal)

    fftconv .*= fftkernel ./ prod(fftsize)

    mul!(conv, plan_bk, fftconv)

    N = size(data) # number of target points
    target_indices = map((n, k) -> k÷2+1:n+k÷2, N, K)

    elt = eltype(data) <: Real ? real : identity
    output .= elt.(view(conv, target_indices...))

    return (; value = output, tape = nothing)
end

function adjoint_solve!(solver::DiscreteConvolutionSolver, adj_output, tape)
    adjoint_conv_solve!(solver, solver.alg, adj_output, tape)
end

function adjoint_conv_solve!(solver, ::FFTConvolution, adj_output, tape)
    (; data, cacheval) = solver
    (; fftsize, signal, K, fftkernel, fftconv, conv, plan_fw, plan_bk, output, adj_data) = cacheval

    N = size(data) # number of target points
    target_indices = map((n, k) -> k÷2+1:n+k÷2, N, K)

    elt = eltype(data) <: Real ? real : identity
    fill!(conv, zero(eltype(conv)))
    view(conv, target_indices...) .= elt.(adj_output.value)
    mul!(fftconv, plan_fw, conv)

    fftconv .*= conj.(fftkernel) ./ prod(fftsize)

    mul!(signal, plan_bk, fftconv)

    adj_data .= elt.(view(signal, axes(data)...))

    return (; data=adj_data, kernel=nothing)
end

end
