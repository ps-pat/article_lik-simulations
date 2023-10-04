using StatsBase: sample, mean, var

using DataFrames

using Random: Xoshiro, shuffle!

using Moosh

using RandomNumbers.PCG: PCGStateSetseq

using JLD2: jldsave

using SpecialFunctions: gamma

import MPI

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

        ## Probability of the tree.
        ftree = CoalMutDensity(n, mut_rate(arg), seq_length)
        log_weight = ftree(arg, logscale = true) .- arg.logprob

        res .+= exp.(log.(fφs(arg)) .+ log_weight)
    end

    res ./ sum(res)
end

export pure_coal2
"""
    pure_coal2
"""
function pure_coal2(rng, sample_prop, models, path = nothing;
                    N = 1_000_000, maf = 5e-2, μ = 1e-1,
                    α = t -> -expm1(-t),
                    M = 1000, n_is = 1000)
    ## MPI setup.
    MPI.Init()
    comm = MPI.COMM_WORLD
    worldsize = MPI.Comm_size(comm)

    seed = rand(rng, Int)
    batchsize = ceil(Int, M ÷ MPI.Comm_size(comm))
    n = round(Int, sample_prop * N)
    pop_phenos = pop_pheno2(rng, pop_hap2(N, maf), models)
    rank = MPI.Comm_rank(comm)

    istars_local = Vector{Int}(undef, batchsize)
    idx_local = Matrix{Int}(undef, n, batchsize)
    liks_local = Matrix{Float64}(undef, length(pop_phenos[:scenarios]), batchsize)
    iterations = range(batchsize * rank + 1, length = batchsize)
    for (idx, k) in enumerate(iterations)
        @info "Simulation" k
        GC.gc()

        rng_local = PCGStateSetseq((seed, k))

        sam_idx = sample(rng_local, 1:N, n, replace = false)
        sam_ηs = getindex(pop_phenos[:ηs], sam_idx)
        derived_idx = findall(seq -> first(seq), sam_ηs)
        while isempty(derived_idx)
            sam_idx = sample(rng_local, 1:N, n, replace = false)
            sam_ηs = getindex(pop_phenos[:ηs], sam_idx)
            derived_idx = findall(seq -> first(seq), sam_ηs)
        end

        ## Remove the phenotype of a random individual.
        possible_stars = isodd(k) ?
            derived_idx : setdiff(1:n, derived_idx)

        istar = (first ∘ sample)(rng_local, possible_stars, 1)

        for (i, scenario) ∈ enumerate(pop_phenos[:scenarios])
            sam_φs = getindex(pop_phenos[scenario][:φs], sam_idx)
            sam_φs[istar] = missing
            p = pop_phenos[scenario][:prevalence]

            fφs = FrechetCoalDensity(sam_φs, pars = Dict(:p => p))

            liks_local[i, idx] =
                last(compute_likelihood(rng_local, fφs, sam_ηs, n_is))
        end

        istars_local[idx] = istar
        idx_local[:, idx] .= sam_idx

        @info "Simulation completed" k
    end

    @info "Iterations finished" iterations

    ## Put it all together & save results.
    if iszero(rank)
        istars = similar(istars_local, worldsize * batchsize)
        istars_buff = MPI.UBuffer(istars, batchsize)

        sampled_idx = similar(idx_local, n, worldsize * batchsize)
        sampled_idx_buff =
            MPI.UBuffer(sampled_idx, (first ∘ size)(sampled_idx) * batchsize)

        liks = similar(liks_local, length(pop_phenos[:scenarios]), worldsize * batchsize)
        fill!(liks, 42)
        liks_buff = MPI.UBuffer(liks, (first ∘ size)(liks) * batchsize)
    else
        istars_buff = MPI.UBuffer(nothing)
        sampled_idx_buff = MPI.UBuffer(nothing)
        liks_buff = MPI.UBuffer(nothing)
    end

    MPI.Gather!(idx_local, sampled_idx_buff, comm)
    MPI.Gather!(istars_local, istars_buff, comm)
    MPI.Gather!(liks_local, liks_buff, comm)

    if iszero(rank) && !isnothing(path)
        ## Fill `res`.
        @info "Compiling Results..."
        res = [Dict{Symbol, Any}() for _ ∈ range(1, length = worldsize * batchsize)]
        for k ∈ eachindex(res)
            res[k][:N] = N
            res[k][:n] = n
            res[k][:scenarios] = pop_phenos[:scenarios]
            res[k][:star] = istars[k]
            res[k][:ηs] = getindex(pop_phenos[:ηs], sampled_idx[:,k])

            for (i, scenario) ∈ enumerate(res[k][:scenarios])
                res[k][scenario] = Dict{Symbol, Any}()
                res[k][scenario][:penetrance] =
                    pop_phenos[scenario][:penetrance]
                res[k][scenario][:φs] =
                    getindex(pop_phenos[scenario][:φs], sampled_idx[:,k])
                res[k][scenario][:prob_φ] = liks[i, k]
            end
        end

        @info "Writing results..."
        jldsave(path, simulation = Dict(
            :samples => vcat(res...),
            :pop => pop_phenos))
    end

    MPI.Finalize()
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
            push!(sims, (scenario, status, sam[scenario][:prob_φ]))
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
function study1(path = "study1.data"; sample_prop = 1e-3, kwargs...)
    rng = Xoshiro(42)
    scenarios = Dict(:full => (wild = 0.05, derived = 1.0),
                     :high => (wild = 0.05, derived = 0.75),
                     :low => (wild = 0.05, derived = 0.2))

    pure_coal2(rng, sample_prop, scenarios, path; kwargs...)
end
