module SSPChainRulesCoreExt

import ChainRulesCore: rrule, NoTangent
using SSP: conic_filter, ssp1_linear, ssp1, ssp2, Project, conic_filter_withsolver, ssp_withsolver, conic_filter_rrule, ssp_rrule

function rrule(::typeof(conic_filter), data, radius, grid)
    rho_filtered, solvers = conic_filter_withsolver(data, radius, grid)
    function conic_filter_pullback(adj_rho_filtered)
        adj_data = conic_filter_rrule(adj_rho_filtered, solvers...)
        return NoTangent(), adj_data, NoTangent(), NoTangent()
    end
    return rho_filtered, conic_filter_pullback
end

function _ssp_rrule(alg, rho_filtered, beta, eta, grid)
    rho_projected, solver = ssp_withsolver(alg, rho_filtered, beta, eta, grid)
    function ssp_pullback(adj_rho_projected)
        adj_rho_filtered = ssp_rrule(adj_rho_projected, solver)
        return NoTangent(), adj_rho_filtered, NoTangent(), NoTangent(), NoTangent()
    end
    return rho_projected, ssp_pullback
end
function rrule(::typeof(ssp1_linear), rho_filtered, beta, eta, grid; kws...)
    _ssp_rrule(Project.SSP1_linear(; kws...), rho_filtered, beta, eta, grid)
end
function rrule(::typeof(ssp1), rho_filtered, beta, eta, grid; kws...)
    _ssp_rrule(Project.SSP1(; kws...), rho_filtered, beta, eta, grid)
end
function rrule(::typeof(ssp2), rho_filtered, beta, eta, grid; kws...)
    _ssp_rrule(Project.SSP2(; kws...), rho_filtered, beta, eta, grid)
end

end