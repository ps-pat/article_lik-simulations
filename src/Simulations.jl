module Simulations

using PrecompileTools

include("pure_coal.jl")

include("weird_shuffle!.jl")

## Precompilation.
@setup_workload begin
    @compile_workload begin
        study1(nothing, nothing, M = 1, n_is = 1)
    end
end

end # module Simulations
