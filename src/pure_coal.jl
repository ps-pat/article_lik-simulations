using Moosh

using RandomNumbers.PCG: PCGStateSetseq

using StatsBase: sample, mean

using Distributions: Bernoulli

using JLSO

using Base.Threads

using DataFrames

pop_hap2(N, maf) =
    vcat([Sequence([one(UInt)], 1) for _ ∈ 1:(N * maf)],
         [Sequence([zero(UInt)], 1) for _ ∈ 1:(N * (1 - maf))])

"""
    pop_pheno2(rng, haplotypes, penetrance)
    pop_pheno2(rng, haplotypes, models)

# Notes
- This function assumes that all the haplotypes with the derived
allele are at the beginning of `haplotypes`.
"""
function pop_pheno2(rng, haplotypes, penetrance::Union{NamedTuple{S, T}, T}) where {S, T<:NTuple{2, Real}}
    N = length(haplotypes)

    ret = Vector{Union{Missing, Bool}}(undef, N)
    fill!(ret, false)

    N_derived = findlast(s -> first(s), haplotypes)
    N_wild = N - N_derived

    ncases_wild = ceil(Int, first(penetrance) * N_wild)
    ncases_derived = ceil(Int, last(penetrance) * N_derived)

    ret[range(N_derived + 1, length = ncases_wild)] .= true
    ret[range(1, length = ncases_derived)] .= true

    ret
end

function pop_pheno2(rng, haplotypes, models::Dict)
    ret = Dict{Symbol, Any}()

    ret[:ηs] = haplotypes
    ret[:scenarios] = (Tuple ∘ keys)(models)
    ret[:N] = length(haplotypes)

    for (model, penetrance) ∈ models
        φs = pop_pheno2(rng, haplotypes, penetrance)
        ret[model] = Dict{Symbol, Any}(:φs => φs,
                                       :prevalence => mean(φs),
                                       :penetrance => penetrance)
    end

    ret
end

function sample_pop(rng, n, pop)
    ret = deepcopy(pop)
    ret[:n] = n

    ηs = ret[:ηs]
    idx = sample(rng, eachindex(ηs), n, replace = false)

    ret[:ηs] = getindex(ret[:ηs], idx)

    for scenario ∈ ret[:scenarios]
        ret[scenario][:φs] = getindex(ret[scenario][:φs], idx)
        delete!(ret[scenario], :prevalence)
    end

    ret
end

function compute_likelihood(rng, fφs, ηs, nb_is;
                            seq_length = 1,
                            Ne = 1000,
                            μ_loc = 5e-5)
    n = length(ηs)
    res = zeros(BigFloat, 2)

    for _ ∈ 1:nb_is
        arg = Arg(
            ηs,
            seq_length = seq_length,
            effective_popsize = Ne,
            μ_loc = μ_loc,
            positions = [0])
        buildtree!(rng, arg)

        ftree = CoalMutDensity(n, mut_rate(arg), seq_length)

        res .+= exp.(log.(fφs(arg))
                     .+ ftree(arg, logscale = true) .- arg.logprob)
    end

    res ./ sum(res)
end

export pure_coal2
"""
    pure_coal2
"""
function pure_coal2(rng, sample_prop, models, path = nothing;
                    N = 1_000_000, maf = 5e-2, μ = 1e-1,
                    M = 1000, n_is = 1000, α = t -> 1 - exp(-t))
    ## Generate population.
    pop_hap = pop_hap2(N, maf)
    pop_phenos = pop_pheno2(rng, pop_hap, models)

    ## Simulation study.
    n = round(Int, sample_prop * N)

    ret = Dict(:pop => pop_phenos,
               :samples => Vector{Dict{Symbol, Any}}(undef, M))

    seed = rand(rng, Int)
    @threads for k ∈ 1:M
        rng_local = PCGStateSetseq((seed, k))

        sam = sample_pop(rng_local, n, pop_phenos)

        ## Remove the phenotype of a random individual.
        derived_idx = findall(seq -> first(seq), sam[:ηs])
        possible_stars = isodd(k) ?
            derived_idx : setdiff(1:n, derived_idx)

        istar = (first ∘ sample)(rng_local, possible_stars, 1)
        ηstar = sam[:ηs][istar]
        sam[:star] = istar

        for scenario ∈ sam[:scenarios]
            sam[scenario][:φs][istar] = missing

            fφs = FrechetCoalDensity(
                sam[scenario][:φs],
                α = α,
                pars = Dict(:p => pop_phenos[scenario][:prevalence]))

            sam[scenario][:lik] =
                compute_likelihood(rng_local, fφs, sam[:ηs], n_is)

            ret[:samples][k] = sam
        end
    end

    isnothing(path) || JLSO.save(path, sims => ret)

    ret
end

export todf
function todf(results)
    ## From simulations.
    sims = DataFrame(Scenario = Symbol[],
                     Status = Symbol[],
                     prob = BigFloat[])

    for sam ∈ results[:samples]
        status = first(sam[:ηs][sam[:star]]) ? :derived : :wild

        for scenario ∈ sam[:scenarios]
            push!(sims, (scenario, status, last(sam[scenario][:lik])))
        end
    end

    ## True values.
    truth = empty(sims)

    for (scenario, status) ∈ Iterators.product(results[:pop][:scenarios],
                                               (:derived, :wild))
        prob = results[:pop][scenario][:penetrance][status]

        push!(truth, (scenario, status, prob))
    end

    Dict(:simulated => sims, :truth => truth)
end
