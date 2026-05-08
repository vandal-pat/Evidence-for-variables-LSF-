using Pkg
Pkg.activate("/VirtualEnvPath/") 
 
using PyPlot
#plt.style.use("gadfly")
#rc("font", weight="bold")
#rc("axes", labelweight="bold")
#rc("font", family="juliamono")
try
    plt.style.use("gadfly_stylesheet")
    rc("font", family="juliamono")
catch
    nothing
end
rc("font", weight="bold")
rc("axes", labelweight="bold")
using EchelleSpectra, EchelleUtils, EchelleSpectralModeling, Echelle.ishell
using LaTeXStrings
using JLD2
using Infiltrator
using FITSIO
using DelimitedFiles
using Glob
using NaNStatistics



function parameter_correlation_plots_wavelength_knots(path, rvs_all_iters, opt_results, labels, k, iteration)
    pygui(false)
    x1 = [opt_results[k][iteration][l].pbest["λ1"].value for l=1:length(opt_results[k][iteration])]
    x2 = [opt_results[k][iteration][l].pbest["λ2"].value for l=1:length(opt_results[k][iteration])]
    x3 = [opt_results[k][iteration][l].pbest["λ3"].value for l=1:length(opt_results[k][iteration])]
    y = rvs_all_iters[k, :, iteration]
    scatter(x1 .- nanmedian(x1), y, label="λ1")
    xlabel("Wavelength Solution Knots - Median Knot Val")
    ylabel("RV [m/s]")
    xlim(-0.05,0.05)
    ylim(nanmedian(y) - 200, nanmedian(y) + 200) 
    title("Wavelength sol. Knots, λ1, iter$([iteration]), $(labels[k])", fontsize=16, fontweight="bold")
    legend()
    tight_layout()
    savefig("$(path)/wavelength_solution_knots_correlation_$(labels[k])_iter$([iteration]).png")
    plt.close()
end

function parameter_correlation_plots_one_by_one(path, rvs_all_iters, RVS, labels, k, iteration, pname) 
    pygui(false)  
    x = RVS   
    y = rvs_all_iters[k, :, iteration]
    scatter(x, y)
    xlabel(pname) 
    ylabel("RVs [m/s]")
    ylim(nanmedian(y) - 200, nanmedian(y) + 200) 
    title("$pname, $(labels[k]), iter $([iteration]) - subtracted bc_vels", fontsize=16, fontweight="bold")
    tight_layout()
    savefig("$(path)/$(pname)_correlation_$(labels[k])_iter$([iteration]).png")
    plt.close()
end


function EchelleSpectralModeling.load_rvs(path::String, label::String)
    println("Loading in RVs for $label")
    fname = path * label * "/RVs/rvs_$label.jld"
    rvs = jldopen(fname) do f
        do_ccf = "rvsxc" in keys(f)
        if do_ccf
            return f["bjds"], f["bc_vels"], f["rvsfwm"], f["rvsfwmerr"], f["rvsxc"], f["rvsxcerr"]
        else
            return f["bjds"], f["bc_vels"], f["rvsfwm"], f["rvsfwmerr"];
        end
    end
    return rvs
end


function EchelleSpectralModeling.load_rvs(path::String, labels::Vector{String})
    _rvs0 = load_rvs(path, labels[1])
    n_chunks = length(labels)
    n_spec, n_iterations = size(_rvs0[3])
    rvs = Dict{String, Any}()
    do_ccf = length(_rvs0) > 4
    rvs["bjds"] = _rvs0[1]
    rvs["bc_vels"] = _rvs0[2]
    rvs["rvsfwm"] = fill(NaN, (n_chunks, n_spec, n_iterations))
    rvs["rvsfwmerr"] = fill(NaN, (n_chunks, n_spec, n_iterations))
    if do_ccf
        rvs["rvsxc"] = fill(NaN, (n_chunks, n_spec, n_iterations))
        rvs["rvsxcerr"] = fill(NaN, (n_chunks, n_spec, n_iterations))
    end
    for i = 1:length(labels)
        _rvs = load_rvs(path, labels[i])
        rvs["rvsfwm"][i, :, :] .= _rvs[3]
        rvs["rvsfwmerr"][i, :, :] .= _rvs[4]
        if do_ccf
            rvs["rvsxc"][i, :, :] .= _rvs[5]
            rvs["rvsxcerr"][i, :, :] .= _rvs[6]
        end
    end
    return rvs
end



function parse_rms(opt_results)
    n_chunks = length(opt_results)
    n_iterations = length(opt_results[1])
    n_spec = length(opt_results[1][1])
    rms = fill(NaN, (n_chunks, n_spec, n_iterations))
    for i=1:n_chunks
        for j=1:n_spec
            for k=1:n_iterations
                rms[i, j, k] = opt_results[i][k][j].rms
            end
        end
    end
    return rms 
end

function load_ensemble(path, label)
    println("Loading in Ensemble for $label")
    fname = glob("*ensemble*.jld", "$path$label/")[1]
    ensemble = jldopen(fname)["ensemble"]
    return ensemble
end



function parse_param_vals(opt_results, pname)
    n_orders = length(opt_results)
    n_iterations = length(opt_results[1])
    n_spec = length(opt_results[1][1])
    out = [opt_results[i][k][j].pbest[pname].value for i=1:n_orders, j=1:n_spec, k=1:n_iterations]
end


function pearsoncc(x, y)
    xbar = nanmean(x)
    ybar = nanmean(y)
    r = nansum((x .- xbar) .* (y .- ybar)) / sqrt(nansum((x .- xbar).^2) * nansum((y .- ybar).^2))
    if !isfinite(r)
        return NaN
    else
        return r
    end
end

function compute_pccs(opt_results, labels, iteration)
    n_chunks = length(opt_results)
    n_spec = length(opt_results[1][1])
    pnames = collect(keys(opt_results[1][iteration][1].pbest))
    n_pars = length(pnames)
    rs = fill(NaN, (n_chunks, n_pars, n_pars))
    for i=1:n_chunks
        for j=1:n_pars
            pname1 = pnames[j]
            for k=1:n_pars
                pname2 = pnames[k]
                x = [opt_results[i][iteration][l].pbest[pname1].value for l=1:n_spec]
                y = [opt_results[i][iteration][l].pbest[pname2].value for l=1:n_spec]
                rs[i, j, k] = pearsoncc(x, y)
                if rs[i, j, k] > 0.5 && pname1 != pname2
                    println("Order $(labels[i]), Parameters $pname1, $pname2, pcc = $(rs[i, j, k])")
                end
            end
        end
    end
    return rs
end

function sync!(rvs, weights)
    bad = findall(@. ~isfinite(rvs) || ~isfinite(weights) || weights <= 0)
    rvs[bad] .= NaN
    weights[bad] .= 0
end

function plot_lsf_kernels(path, data, models, opt_results, iteration)
    n_chunks = length(opt_results)
    n_spec = length(opt_results[1][1])
    pygui(false)
    mkpath(path * "lsf_kernels")
    for i = 1:n_chunks
        fig = plt.figure(figsize=(8, 5), dpi=200)
        x = Maths.δλ2δv.(models[i].templates["λlsf"], nanmedian(models[i].templates["λ"]))
        for j = 1:n_spec
            kernel = build(models[i].lsf, opt_results[i][iteration][j].pbest, models[i].templates, data[i][j], models[i].sregion)
            plot(x, kernel)
        end
        xlabel("Relative Velocity [m/s]")
        ylabel("LSF, Area Normalized")
        savefig("$path/lsf_kernels/lsf_kernels_$(labels[i]).png")
        plt.close()
    end
end


function compute_rms(data, models, opt_results, iteration)
    n_chunks = length(opt_results)
    n_iterations = length(opt_results[1])
    n_spec = length(opt_results[1][1])
    rms = fill(NaN, (n_chunks, n_spec))
    for i = 1:n_chunks
        for j = 1:n_spec
            data_flux = data[i][j].flux
            pars = opt_results[i][iteration][j].rms 
            _, model_flux = build(models[i], pars, data[i][j])
            residuals = data_flux .- model_flux 
            rms[i, j] = Maths.rmsloss(residuals, mask_worst=20, mask_edges=6) 
        end
    end
    return rms 
end


function parse_rms(opt_results)
    n_chunks = length(opt_results)
    n_iterations = length(opt_results[1])
    n_spec = length(opt_results[1][1])
    rms = fill(NaN, (n_chunks, n_spec, n_iterations))
    for i = 1:n_chunks
        for j = 1:n_spec
            for k = 1:n_iterations
                if opt_results[i][k][j].success
                    rms[i, j, k] = opt_results[i][k][j].rms 
                end
            end
        end
    end
    return rms
end



# Basic info
path = "/Path/"  

do_orders = [220]
#do_orders = [217,218,219,221,222,223,224,225,226,227]  
labels = ["Order$o" for o ∈ do_orders]
iteration = 1

# Load in results

data, models = load_data_models(path, labels)
opt_results = load_opt_results(path, labels)


rvs = load_rvs(path, labels) 


# Numbers
n_spec = length(rvs["bjds"])
n_orders = length(do_orders)
n_iterations = length(opt_results[1])

 # Bin jds
jds_binned, indices = bin_jds(rvs["bjds"], sep=0.5, utc_offset=-10)


rvs_all_iters = copy(rvs["rvsfwm"])  
weights_all_iters = 1 ./ rvs["rvsfwmerr"] .^ 2

xc = copy(rvs["bc_vels"])

k = n_orders 

#RVS = (rvs_all_iters[k,:,iteration] .- xc[:])  


 bad = findall(rvs["rvsfwmerr"] .> 80)
 
 rvs_all_iters[bad] .= NaN


 function hms2decimal(x)
     x = split(x, ':')
     x = parse(Float64, x[1]) + parse(Float64, x[3]) / 60 + parse(Float64, x[3]) / 3600
     return x
 end

 function bin_vals(x, indices)
     n_bins = length(indices)
     xb = fill(NaN, n_bins)
     for i=1:n_bins
         xb[i] = nanmean(x[indices[i]])
     end
     return xb
 end

 sync!(rvs_all_iters, weights_all_iters)
 

 rvs_single1, unc_single1, t_binned1, rvs_binned1, unc_binned1 = combine_rvs(rvs["bjds"], rvs_all_iters[:, :, 1], weights_all_iters[:, :, 1], indices, nσ=5, n_iterations=10)
 


 geomspace(a, b, num) = exp.(range(log(a), log(b), length=num))

 begin
 errorbar(t_binned1 .- 2457000, rvs_binned1, yerr=unc_binned1, marker="o", lw=0, elinewidth=2, markersize=10, mec="grey", label="Iter 1")
 legend()
 xlabel("BJD - 2457000"); ylabel("RV [m/s]"); title("TOI 442, iSHELL data - "); plt.show()
 end




parse_rms(opt_results)