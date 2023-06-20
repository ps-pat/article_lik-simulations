using Moosh

using RandomNumbers.PCG: PCGStateOneseq

using StatsBase: sample, mean

using Distributions: Bernoulli

using JLSO

pop_hap2(N, maf) =
    vcat([Sequence([one(UInt)], 1) for _ ∈ 1:(N * maf)],
         [Sequence([zero(UInt)], 1) for _ ∈ 1:(N * (1 - maf))])

function pop_pheno2(rng, haplotypes, penetrance::NTuple{2, Real})
    phenotypes = mapreduce(vcat, haplotypes) do haplotype
        ps = first(haplotype) ? last(penetrance) : first(penetrance)
        dists = Bernoulli.(ps)
        rand.(rng, dists)
    end
    phenotypes = convert(Vector{Union{Missing, Bool}}, phenotypes)
end

function pop_pheno2(rng, haplotypes, models::Dict)
    ret = Dict{keytype(models), Vector{Union{Missing, Bool}}}()

    for (model, penetrance) ∈ models
        ret[model] = pop_pheno2(rng, haplotypes, penetrance)
    end

    ret
end

function sample_pop(rng, n, haplotypes, phenotypes)
    idx = sample(rng, eachindex(haplotypes), n, replace = false)

    new_phenotypes = deepcopy(phenotypes)
    map!(φs -> getindex(φs, idx), values(new_phenotypes))

    (;haplotypes = haplotypes[idx],
     phenotypes = new_phenotypes)
end

function append_sample!(new_haplotype, sam)
    push!(sam.haplotypes, new_haplotype)
    map!(φs -> push!(φs, missing), values(sam.phenotypes))

    sam
end

export pure_coal2
"""
    pure_coal2

# Empirical Complexity
## Function of `n`
Linear. With `n_is = 4` and `n_mcmc = 10`:
- 1 -> 17s
- 10 -> 41s
- 100 -> 281s

## Function of `n_is`
With `n = 10` and `n_mcmc = 10`:
- 4 -> 41s
- 8 -> 75s
- 12 -> 96s
"""
function pure_coal2(rng, sample_prop, models, path = nothing;
                    N = 1_000_000, maf = 5e-2, μ = 1e-1,
                    M = 1000, n_is = 1000, n_mcmc = 1000)
    ## Generate population.
    pop_hap = pop_hap2(N, maf)
    pop_phenos = pop_pheno2(rng, pop_hap, models)

    prevalences = Dict{keytype(pop_phenos), Float64}()
    for (model, φs) ∈ pop_phenos
        prevalences[model] = mean(φs)
    end

    ## Simulation study.
    n = round(Int, sample_prop * N)
    dens_tree = CoalMutDensity(n + 1, μ, 1)

    res = map(1:M) do _
        sam_base = sample_pop(rng, n, pop_hap, pop_phenos)

        sams = (;wild = append_sample!(Sequence(zeros(UInt, 1), 1),
                                       deepcopy(sam_base)),
                derived = append_sample!(Sequence(ones(UInt, 1), 1),
                                         deepcopy(sam_base)))

        map(sams) do sam
            ret = Dict{keytype(models), IsChain}()
            for model ∈ keys(models)
                prevalence, φs = prevalences[model], sam.phenotypes[model]
                joint = compute_joint(rng,
                                      sam.haplotypes,
                                      dens_tree,
                                      FrechetCoalDensity(φs, pars = Dict(:p => prevalence)),
                                      n_is = n_is, n_mcmc = n_mcmc)

                ret[model] = joint
            end

            ret
        end
    end

    ret = (;res_pheno2 = res,
           prevalence_pheno2 = prevalences,
           pop_pheno2 = pop_phenos)

    if !isnothing(path)
        pairs = map((var, val) -> var => val, keys(ret), values(ret))
        JLSO.save(path, pairs...)
    end

    ret
end
