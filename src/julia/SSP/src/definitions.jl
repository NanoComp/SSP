# Define a CommonSolve-like interface for adjoint problems

function init! end

function init(prob, alg)
    newprob = copy(prob)
    solver = init!(newprob, alg)
    return solver
end

function solve! end

function solve(prob, alg)
    solver = init(prob, alg)
    sol = solve!(solver)
    return sol
end
function adjoint_solve! end

function adjoint_solve(prob, alg, adj_sol, tape)
    solver = init(prob, alg)
    adj_prob = adjoint_solve!(solver, adj_sol, tape)
    return adj_prob
end