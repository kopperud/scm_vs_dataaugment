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

exp(0.5 .* Q)

tree2 = stochastic_character_map(tree, model)

## make some new samples
stochastic_character_maps = Root[]
root_states = String[]

@showprogress for i in 1:50_000
    scm = stochastic_character_map(tree, model)
    root_state = scm.state
    push!(stochastic_character_maps, scm)
    push!(root_states, root_state)
end


## summarize the root state
counts = Dict{String, Int64}()
for root_state in model.state_space
    counts[root_state] = 0
end

for root_state in root_states
    counts[root_state] += 1
end

S = ancestral_state_probabilities(model, tree)


branches = get_branches(tree2);
length(branches)

trees = Root[]
for _ in 1:10
    tree2 = stochastic_character_map(tree, model)
    push!(trees, tree2)
end

function branch_history_logpdf(tree, model)
    branches = get_branches(tree);

    lnl = 0.0
    for branch in branches
        lnl += -sum(branch.times)
        n_events = length(branch.times) -1
        lnl += n_events * log(model.λ)
    end
    return(lnl)
end

function branch_history_nchanges(tree)
    branches = get_branches(tree);

    n = 0
    for branch in branches
        n_events = length(branch.times) - 1
        n += n_events
    end
    return(n)
end

lnls = [branch_history_logpdf(tr, model) for tr in stochastic_character_maps]
nchanges = [branch_history_nchanges(tr) for tr in stochastic_character_maps]

hist(lnls, bins = 48, color = :gray)
hist(nchanges, bins = 48, color = :gray)

P = StochasticCharacterMap.transition_probability(model, 0.5)
unique(lnls)

