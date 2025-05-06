# data augmentation for branch history goes here


function BranchHistoryLogProbability(branch::Branch, model::MkModel)
    n_transitions = length(branch.states) - 1

    # p = exp^{-λ * t) * λ^ntransitions
    lp = (-1.0) * model.λ * sum(branch.times)
    lp += n_transitions * model.λ
    return(lp)
end

function logHistoryProbability(node::Root, model::MkModel)
    branches = get_branches(node)

    lnl = 0.0
    for branch in branches
        lnl += BranchHistoryLogProbability(branch, model)
    end
    return(lnl)
end

logHistoryProbability(tree, model)

