using StatsBase: sample, mean, var

using DataFrames

using Random: Xoshiro, shuffle!

using Moosh

using RandomNumbers.PCG: PCGStateSetseq

using JLD2: jldsave

using SpecialFunctions: gamma

import MPI

export pop_hap2
pop_hap2(N, maf) =
    vcat([Sequence([one(UInt)], 1) for _ ∈ 1:(N * maf)],
         [Sequence([zero(UInt)], 1) for _ ∈ 1:(N * (1 - maf))])

export pop_pheno2
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

function sample2(rng, n, φs, ::Nothing)
    N = length(φs)
    sample(rng, 1:N, n, replace = false)
end

function sample2(rng, n, φs, ncases)
    N = length(φs)
    ncontrols = n - ncases
    cases = findall(skipmissing(φs))
    controls = setdiff(1:N, cases)

    [sample(rng, cases, ncases, replace = false);
     sample(rng, controls, ncontrols, replace = false)]
end

export pure_coal2_sim
function pure_coal2_sim(pop_phenos, seed, k, n, ncases, n_is; scale_α = false)
    rng_local = PCGStateSetseq((seed, k))
    nscenarios = length(pop_phenos[:scenarios])

    lik = Vector{Float64}(undef, nscenarios)
    istars = similar(lik, Int)
    idx = similar(istars, n, nscenarios)
    for (i, scenario) ∈ enumerate(pop_phenos[:scenarios])
        ## Draw sample.
        @label draw_sample

        idx[:,i] .= sample2(rng_local, n, pop_phenos[scenario][:φs], ncases)
        sam_ηs = getindex(pop_phenos[:ηs], idx[:,i])
        derived_idx = findall(seq -> first(seq), sam_ηs)

        isodd(k) && isempty(derived_idx) && @goto draw_sample

        ## Remove the phenotype of a random individual.
        possible_stars = isodd(k) ? derived_idx : setdiff(1:n, derived_idx)

        istars[i] = (first ∘ sample)(rng_local, possible_stars, 1)

        sam_φs = getindex(pop_phenos[scenario][:φs], idx[:,i])
        sam_φs[istars[i]] = missing
        p = pop_phenos[scenario][:prevalence]

        fφs = FrechetCoalDensity(sam_φs,
                                 pars = Dict(:p => p),
                                 scale_α = scale_α)

        lik[i] = last(compute_likelihood(rng_local, fφs, sam_ηs, n_is))
    end

    lik, istars, idx
end

export pure_coal2
"""
    pure_coal2
"""
function pure_coal2(rng, sample_prop, models, cases_prop = nothing, path = nothing;
                    N = 1_000_000, maf = 5e-2, μ = 1e-1,
                    α = t -> -expm1(-t),
                    M = 1000, n_is = 1000,
                    scale_α = false)
    ## MPI setup.
    MPI.Init()
    comm = MPI.COMM_WORLD
    worldsize = MPI.Comm_size(comm)

    seed = rand(rng, Int)
    batchsize = ceil(Int, M ÷ MPI.Comm_size(comm))
    n = round(Int, sample_prop * N)
    pop_phenos = pop_pheno2(rng, pop_hap2(N, maf), models)
    ncases = isnothing(cases_prop) ? nothing : round(Int, cases_prop * n)
    nscenarios = length(pop_phenos[:scenarios])
    rank = MPI.Comm_rank(comm)

    liks_local = Matrix{Float64}(undef, nscenarios, batchsize)
    istars_local = similar(liks_local, Int)
    idx_local = similar(istars_local, n, nscenarios, batchsize)

    iterations = range(batchsize * rank + 1, length = batchsize)
    for (idx, k) in enumerate(iterations)
        @info "Simulation" k
        GC.gc()

        res = pure_coal2_sim(pop_phenos,
                             seed, k,
                             n, ncases, n_is, scale_α = scale_α)

        liks_local[:,idx] .= res[1]
        istars_local[:,idx] .= res[2]
        idx_local[:,:,idx] .= res[3]

        @info "Simulation completed" k
    end

    @info "Iterations finished" iterations

    ## Put it all together & save results.
    if iszero(rank)
        liks = similar(liks_local,
                       (first ∘ size)(liks_local),
                       (last ∘ size)(liks_local) * worldsize)
        liks_buff = MPI.UBuffer(liks, (prod ∘ size)(liks_local))

        istars = similar(istars_local,
                         (first ∘ size)(istars_local),
                         (last ∘ size)(istars_local) * worldsize)
        istars_buff = MPI.UBuffer(istars, (prod ∘ size)(istars_local))

        sampled_idx = similar(idx_local,
                              size(idx_local)[1:2]...,
                              (last ∘ size)(idx_local) * worldsize)
        sampled_idx_buff = MPI.UBuffer(sampled_idx, (prod ∘ size)(idx_local))
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

            for (i, scenario) ∈ enumerate(pop_phenos[:scenarios])
                res[k][scenario] = Dict{Symbol, Any}()
                res[k][scenario][:penetrance] =
                    pop_phenos[scenario][:penetrance]
                res[k][scenario][:φs] =
                    getindex(pop_phenos[scenario][:φs], sampled_idx[:, i, k])
                res[k][scenario][:prob_φ] = liks[i, k]
                res[k][scenario][:ηs] = getindex(pop_phenos[:ηs], sampled_idx[:, i, k])
                res[k][scenario][:star] = istars[i, k]
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
                     prob = Float64[])

    for sam ∈ results[:samples]
        for scenario ∈ sam[:scenarios]
            status = first(sam[scenario][:ηs][sam[scenario][:star]]) ?
                :derived : :wild
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
function study1(cases_prop = nothing, path = "study1.data";
                sample_prop = 1e-3, f0 = 0.05, kwargs...)
    rng = Xoshiro(42)
    scenarios = Dict(:full => (wild = f0, derived = 1.0),
                     :high => (wild = f0, derived = 0.75),
                     :low => (wild = f0, derived = 0.2))

    pure_coal2(rng, sample_prop, scenarios, cases_prop, path; kwargs...)
end
