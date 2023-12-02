module Simulations

using PrecompileTools

include("pure_coal.jl")

include("weird_shuffle!.jl")

## Precompilation.
@setup_workload begin
    @compile_workload begin
        study1(nothing, nothing, nothing, M = 1, n_is = 1, sample_prop = 2e-5)
    end
end

end # module Simulations
