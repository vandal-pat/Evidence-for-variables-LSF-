using Distributed
using Pkg
Pkg.activate("/home/USER-ID/VirtualEnvName/")
using EchelleUtils, EchelleSpectra, EchelleSpectralModeling, Echelle.ishell
#addprocs(15)
using NaNStatistics

@everywhere begin
    using Pkg
    Pkg.activate("/home/USER-ID/VirtualEnvName/")
    using EchelleUtils, EchelleSpectra, EchelleSpectralModeling, Echelle.ishell 
    # Read in reduced spectrum
function EchelleSpectra.read_spec1d!(data::SpecData1D{:ishell}, sregion::SpecRegion1D)
    d = read_fitstable(data.fname; hdu=2, column=replace(sregion.label, "Order" => "")) 
    spec = d[:, 1]
    specerr = d[:, 2]
    reverse!(spec)
    reverse!(specerr)
    data.λ = ishell.get_λsolution_estimate(data, sregion)
    data.spec = spec
    data.specerr = specerr

    s = nanquantile(data.spec, 0.98)
    spec_copy = data.spec ./ s   
    bad = findall( x -> x < 0.15, spec_copy)     
    data.spec[bad] .= NaN
    data.specerr[bad] .= NaN
   
end
end

# Config
spectrograph = "iSHELL"
data_input_path = "/Path/"
filelist = "filelist.txt"
data_files = [data_input_path * fname for fname ∈ eachline(data_input_path * filelist) if !startswith(fname, '#')]
output_path = "/Path/"
star_name = "HD_15906"
do_orders = [220]
templates_path = "/Path/"

# Loop over orders
for order ∈ do_orders
    
    # Model
    pixmin, pixmax = 400, 2048 - 400 
    sregion = SpecRegion1D(pixrange=[400,1648], label="Order$order", mask_mode="pixels")

# Create the model
model = SpectralForwardModel(;sregion, sampling=4,
λsolution=PolyλSolution(deg=2, bounds=[-0.05, 0.05]),
continuum=PolyContinuum(deg=2, bounds= [0.8,1.2]), #[0.9, 1.1]),
lsf=GaussHermiteLSF(deg=0, bounds=[[0.009,0.014], [-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]]),
gascell=GasCell(input_file=templates_path * ishell.GASCELL_FILE, τ_bounds=[ishell.τ_GASCELL, ishell.τ_GASCELL]),

star=AugmentedStar(input_file=templates_path * "XXX.txt", star_name=star_name, rv_abs=-3.697E3),

tellurics=TAPASTellurics(input_file=templates_path * "TAPAS_tellurics_maunakea.jld", τ_water_bounds=[0.05, 5], τ_airmass_bounds=[0.9, 3.0], vel_bounds=[-70.0, -70.0])
)

    # Data
    data = initialize_data(data_files, sregion, spectrograph; norm=0.98)

    # Run RVs for this order
    drive( data, model; output_path, do_ccf=true, ccf_kwargs = (;vel_window_coarse=2_000, vel_step_coarse=50, vel_step_fine=10, vel_window_fine=200, min_data_flux=0.2, max_star_flux=0.99),
    individual_optim_kwargs=(;loss="redχ2", ftol_rel=1E-6, mask_worst=20, mask_edges=6, reset_star_vel=true), n_augment_iters=10, augment_kwargs=(;smooth_width=0, smooth_poly_deg=3, max_val=1.0, remove_contaminants=false, weights=:specerr),
    n_global_iters=0, global_optim_kwargs=(;miniters=1, maxiters=300, ftol=1E-6, n_spec_max=Inf, mask_worst=20, conv_sampling=:lr)
    )

end

rmprocs(workers())