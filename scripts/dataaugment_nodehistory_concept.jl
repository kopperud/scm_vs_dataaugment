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


S = ancestral_state_probabilities(model, tree)

## dummy tree
#
#                 (B)
#                 /
#                /
# (R) ------- (A) 
#               \ 
#                \
#                (C)
#
#
#
# Joint probability
# P(r,a,b,c) = P(r) * P(a|r) * P(b|a) * P(c|a)
#
# Probability of A conditional on observing R, B and C
#
# P(A = a|R,B,C) = α * sum_{r,b,c} P(r) * P(a|r) P(b|a) * P(c|a) * P(b) * P(c)
#

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


