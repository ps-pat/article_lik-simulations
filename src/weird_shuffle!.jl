using Random: AbstractRNG, GLOBAL_RNG, ltm52, randexp

function weird_shuffle!(rng::AbstractRNG, arg, classes)
    n = nleaves(arg)

    ## Draw leaves to permute.
    leaves_filtered = Iterators.filter(v -> Moosh.isleaf(arg, v),
                                       highest_under(arg, randexp(rng) / log(n)))
    idx = setdiff(Moosh.leaves(arg), leaves_filtered)

    for class ∈ classes
        shuffle!(rng, view(class, findall(v -> v ∈ idx, class)))
    end

    classes
end
