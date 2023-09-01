using StatsBase: sample, mean

using DataFrames

using Random: Xoshiro

using Distributed

using Moosh

using ProgressMeter

using RandomNumbers.PCG: PCGStateSetseq

using JLSO

@everywhere begin
    using Moosh

    using ProgressMeter

    using RandomNumbers.PCG: PCGStateSetseq

    using JLSO
end

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
    ret = empty(pop)

    ret[:N] = pop[:N]
    ret[:scenarios] = pop[:scenarios]
    ret[:n] = n

    idx = sample(rng, 1:pop[:N], n, replace = false)

    ret[:ηs] = getindex(pop[:ηs], idx)

    for scenario ∈ pop[:scenarios]
        ret[scenario] = Dict{Symbol, Any}()
        ret[scenario][:φs] = getindex(pop[scenario][:φs], idx)
        ret[scenario][:penetrance] = pop[scenario][:penetrance]
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
                    α = (t, λ) -> 1 - exp(-t / λ),
                    M = 1000, n_is = 1000)
    ## Generate population.
    pop_hap = pop_hap2(N, maf)
    pop_phenos = pop_pheno2(rng, pop_hap, models)

    ## Simulation study.
    n = round(Int, sample_prop * N)
    seed = rand(rng, Int)

    batchsize = ceil(Int, M ÷ nworkers())

    res = @showprogress pmap(1:M, batch_size = batchsize) do k
        GC.gc()

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
        end

        sam
    end

    ret = Dict(:pop => pop_phenos, :samples => res)

    isnothing(path) || JLSO.save(path, :simulation => ret)

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

export study1

"""
    study1()

Execute the first simulation study.
"""
function study1(; kwargs...)
    rng = Xoshiro(42)
    sample_prop = 1e-3
    scenarios = Dict(:full => (wild = 0.05, derived = 1.0),
                     :high => (wild = 0.05, derived = 0.75),
                     :low => (wild = 0.05, derived = 0.2))
    path = "study1.jlso"

    pure_coal2(rng, sample_prop, scenarios, path; kwargs...)
end
