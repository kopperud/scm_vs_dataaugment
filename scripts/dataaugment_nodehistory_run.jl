using StochasticCharacterMap
using CairoMakie
using ProgressMeter
using Distributions
using DataStructures

## read character history
tree = readsimmap("data/ungulates_simmap.tre")

## plot the character history
treeplot(tree)

λ = 0.05
model = MkModel(["Gr", "Br", "MF"], λ, 3)

## analytical marginal probabilities
S = ancestral_state_probabilities(model, tree)


##########################################
##
## code to implement the data augmentation
##
##########################################

function nodeProbability(node::Node, model::MkModel)
    # A is this node
    # R is the parent node
    # B and C are child nodes

    R = node.inbounds.inbounds
    A = node
    B = node.left.outbounds
    C = node.right.outbounds

    R_index = findfirst(model.state_space .== R.state)
    B_index = findfirst(model.state_space .== B.state)
    C_index = findfirst(model.state_space .== C.state)

    pA_given_R = transition_probability(model, sum(A.inbounds.times))
    pB_given_A = transition_probability(model, sum(A.left.times))
    pC_given_A = transition_probability(model, sum(A.right.times))

    pA = zeros(model.k)
    for a in 1:model.k
        pA[a] = pA_given_R[a,R_index] * pB_given_A[B_index,a] * pC_given_A[C_index,a]
    end
    pA = pA ./ sum(pA)
    return(pA)
end

function nodeProbability(node::Root, model::MkModel)
    # A is this node
    # B and C are child nodes

    A = node
    B = node.left.outbounds
    C = node.right.outbounds

    B_index = findfirst(model.state_space .== B.state)
    C_index = findfirst(model.state_space .== C.state)

    pB_given_A = transition_probability(model, sum(A.left.times))
    pC_given_A = transition_probability(model, sum(A.right.times))

    ## assume that the root prior is uniform
    prior_A = ones(model.k) ./ model.k
    
    pA = zeros(model.k)
    for a in 1:model.k
        pA[a] = prior_A[a] * pB_given_A[B_index,a] * pC_given_A[C_index,a]
    end
    pA = pA ./ sum(pA)
    return(pA)
end

function sampleNode(node::T, model::MkModel) where {T <: InternalNode}
    p = nodeProbability(node, model)

    d = Distributions.Categorical(p)
    new_state = model.state_space[rand(d)]

    return(new_state)
end

function updateNode(node::Root, model::MkModel)
    previous_state = node.state

    new_state = sampleNode(node, model)

    if previous_state != new_state
        node.state = new_state

        oldest_state = new_state

        ## left branch
        left_branch = node.left
        left_node = left_branch.outbounds
        left_state = left_node.state
        redraw_branch!(left_branch, model, oldest_state, left_state)

        ## right branch
        right_branch = node.right
        right_node = right_branch.outbounds
        right_state = right_node.state
        redraw_branch!(right_branch, model, oldest_state, right_state)     
    end
   
    nothing
end

function updateNode(node::Node, model::MkModel)
    previous_state = node.state
    
    new_state = sampleNode(node, model)

    if new_state != previous_state
        node.state = new_state

        oldest_state = new_state

        ## parent branch
        parent_branch = node.inbounds
        parent_node = parent_branch.inbounds
        parent_state = parent_node.state
        redraw_branch!(parent_branch, model, parent_state, node.state)

        ## left branch
        left_branch = node.left
        left_node = left_branch.outbounds
        left_state = left_node.state
        redraw_branch!(left_branch, model, oldest_state, left_state)

        ## right branch
        right_branch = node.right
        right_node = right_branch.outbounds
        right_state = right_node.state
        redraw_branch!(right_branch, model, oldest_state, right_state)     
    end

    nothing
end

function branchLogProbability(branch::Branch, model::MkModel)
    old_state = branch.states[end]
    young_state = branch.states[1]
    old_state_index = argmax(old_state .== model.state_space)
    young_state_index = argmax(young_state .== model.state_space)
    t = sum(branch.times)
    P = transition_probability(model, t)
    p = P[young_state_index, old_state_index]
    return(log(p))
end

function logProbability(node::Root, model::MkModel)
    branches = get_branches(node)
    
    lnl = 0.0
    for branch in branches
        lnl += branchLogProbability(branch, model)
    end

    return(lnl)
end


## run a data augmentation analysis

tree2 = stochastic_character_map(tree, model)
nodes = get_nodes(tree2)

n_iterations = 10_000
weight = 100

chain = String[]
lnls = Float64[]
@showprogress for i in 1:n_iterations
    for j in 1:weight
        node = rand(nodes)
        updateNode(node, model)
    end

    lnl = logProbability(tree2, model)
    push!(chain, tree2.state)
    push!(lnls, lnl)
end

## plot the root probability
fig = Figure(size = (900, 300))
ax = Axis(fig[1,1], xlabel = "iteration", ylabel = "log P(Z = z|λ)")
lines!(ax, 1:1_000, lnls[1:1_000])
fig


## do some tests of what the metropolis-hastings
## ratio would be, if we were to calculate it
q = nodeProbability(tree2, model)
p0 = logProbability(tree2, model)

updateNode(tree2, model)
p1 = logProbability(tree2, model)

acceptance_probability = exp(p1 - p0) * q[1] / q[3]

