# data augmentation for branch history goes here


function BranchHistoryLogProbability(branch::Branch, model::MkModel)
    n_transitions = length(branch.states) - 1

    # p = exp^{-λ * t) * λ^ntransitions
    lp = (-1.0) * model.λ *(model.k-1) * sum(branch.times)
    lp += n_transitions * (model.λ)
    return(lp)
end

function BranchHistoryLogProbability(branch::Branch, model::ARDModel)
    n_transitions = length(branch.states) - 1

    # p = exp^{-λ * t) * λ^ntransitions
    lp = 0.0

    
    ## order is now from old to young
    states = reverse(branch.states)
    times = reverse(branch.times)

    ## probability of no events in the episodes
    for (time, state) in zip(times, states)
        state_index = findfirst(model.state_space .== state)
        lp += model.Q[state_index,state_index] * time
    end

    ## probability (density) of the change events
    if n_transitions > 0
        for i in 1:n_transitions
            old_state = states[i]
            young_state = states[i+1]
            
            old_index = findfirst(model.state_space .== old_state)
            young_index = findfirst(model.state_space .== young_state)

            lp += model.Q[young_index, old_index]
        end
    end

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

tree2 = deepcopy(tree);

logHistoryProbability(tree, model)

updateNode(tree2, model)
logHistoryProbability(tree2, model)

Q = [
    -0.05 0.025 0.025
    0.03 -0.05 0.025
    0.02 0.025 -0.05
]

λ = 0.025
model = MkModel(["Gr", "Br", "MF"], λ, 3)
model2 = ARDModel(["Gr", "Br", "MF"], Q)

transition_probability(model, 0.5)
transition_probability(model2, 0.5)

BranchHistoryLogProbability(tree.left, model)
BranchHistoryLogProbability(tree.left, model2)


q0 = BranchHistoryLogProbability(tree.left, model) + BranchHistoryLogProbability(tree.right, model)
q1 = BranchHistoryLogProbability(tree2.left, model) + BranchHistoryLogProbability(tree2.right, model)

#π0 = branchLogProbability(tree.left, model) + branchLogProbability(tree.right, model)
#π1 = branchLogProbability(tree2.left, model) + branchLogProbability(tree2.right, model)

p0 = logHistoryProbability(tree, model)
p1 = logHistoryProbability(tree2, model)

acceptance_probability = exp(p1 - p0 + q0 - q1)
#hastings_ratio = exp(p1 - p0 + π0 - π1)




sampleNode(tree2, model); treeplot(tree2)

treeplot(tree)