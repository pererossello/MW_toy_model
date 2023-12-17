import numpy as np
from astroquery.gaia import Gaia
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import Distance
from astropy.stats import gaussian_sigma_to_fwhm


def retrieve_gaia_data(dist_min, dist_max, flux_oerror_thresholds, par_oerror_thresholds,
                       radv_error=1, astro_noise=1):
    gaia_data = None
    N = 0
    for flux_oerror, par_oerror in zip(flux_oerror_thresholds, par_oerror_thresholds):
        adql_query = distance_range_query(dist_min, dist_max, flux_oerror=flux_oerror, par_oerror=par_oerror, radv_error=radv_error, astro_noise=astro_noise)
        job = Gaia.launch_job_async(adql_query)
        gaia_data_temp = job.get_results()
        N_temp = len(gaia_data_temp)
        
        # If number of stars meets the criteria, stop and return the data
        if N_temp >= 1000:
            gaia_data = gaia_data_temp
            N = N_temp
            break
        elif N_temp > N:  # Update with the latest best effort if it retrieved more stars
            gaia_data = gaia_data_temp
            N = N_temp
            
    return gaia_data, N, par_oerror

def distance_range_query(min_distance, max_distance, flux_oerror=20, par_oerror=20, radv_error=1,
                         astro_noise=1):
 
    max_par = 1e3/min_distance
    min_par = 1e3/max_distance

    adql_query = f"""
    SELECT dr3.ra, dr3.dec, dr3.ra_error, dr3.dec_error,
    dr3.parallax, dr3.parallax_error, 
    dr3.phot_g_mean_flux, dr3.phot_g_mean_mag, dr3.phot_g_mean_flux_error, 
    dr3.bp_rp, 
    dr3.phot_rp_mean_flux, dr3.phot_rp_mean_mag, dr3.phot_rp_mean_flux_error,
    dr3.phot_bp_mean_flux, dr3.phot_bp_mean_flux_error,
    dr3.astrometric_excess_noise, dr3.astrometric_excess_noise_sig,
    dr3.phot_bp_rp_excess_factor,
    dr3.pm, dr3.pmra, dr3.pmra_error, dr3.pmdec, dr3.pmdec_error, 
    dr3.radial_velocity, dr3.radial_velocity_error,
    dr3.phot_bp_mean_flux_over_error, dr3.phot_rp_mean_flux_over_error,
    dr3.l, dr3.b,
    dr3.non_single_star,
    dr3.ag_gspphot, dr3.ag_gspphot_lower, dr3.ag_gspphot_upper,
    dr3.ebpminrp_gspphot, dr3.ebpminrp_gspphot_lower, dr3.ebpminrp_gspphot_upper
    FROM gaiadr3.gaia_source AS dr3
    WHERE (dr3.parallax BETWEEN {min_par} AND {max_par})
    AND dr3.radial_velocity IS NOT null
    AND dr3.radial_velocity_error < {radv_error}
    AND dr3.non_single_star = 0
    AND dr3.parallax_over_error > {par_oerror}
    AND dr3.phot_bp_mean_flux_over_error > {flux_oerror}
    AND dr3.phot_rp_mean_flux_over_error > {flux_oerror}
    AND dr3.phot_g_mean_flux_error > {flux_oerror}
    AND dr3.astrometric_excess_noise < {astro_noise}
    """

    return adql_query

def distance_query(distance):
 
    par = 1e3/distance

    adql_query = f"""
    SELECT dr3.ra, dr3.dec, dr3.ra_error, dr3.dec_error,
    dr3.parallax, dr3.parallax_error, dr3.parallax_over_error,
    dr3.phot_g_mean_flux, dr3.phot_g_mean_mag, dr3.phot_g_mean_flux_error, 
    dr3.bp_rp, 
    dr3.phot_rp_mean_flux, dr3.phot_rp_mean_mag, dr3.phot_rp_mean_flux_error,
    dr3.phot_bp_mean_flux, dr3.phot_bp_mean_flux_error,
    dr3.astrometric_excess_noise, dr3.astrometric_excess_noise_sig,
    dr3.phot_bp_rp_excess_factor,
    dr3.pm, dr3.pmra, dr3.pmra_error, dr3.pmdec, dr3.pmdec_error, 
    dr3.radial_velocity, dr3.radial_velocity_error,
    dr3.phot_bp_mean_flux_over_error, dr3.phot_rp_mean_flux_over_error,
    dr3.l, dr3.b,
    dr3.non_single_star
    FROM gaiadr3.gaia_source AS dr3
    WHERE dr3.parallax > {par}
    AND dr3.radial_velocity IS NOT null
    AND dr3.radial_velocity_error < 1
    AND dr3.non_single_star = 0
    AND dr3.parallax_over_error > 20
    AND dr3.visibility_periods_used > 5
    AND dr3.ruwe < 1
    AND dr3.phot_bp_mean_flux_over_error > 20
    AND dr3.phot_rp_mean_flux_over_error > 20
    AND dr3.astrometric_excess_noise < 1
    """

    return adql_query


def bayesian_distance_estimator(data):

    parallax_data = data['parallax']
    parallax_error = data['parallax_error']

    def likelihood(parallax, parallax_true, parallax_error):
        return np.exp(-0.5 * ((parallax - parallax_true) / parallax_error)**2)

    def prior(distance):
        # Replace with the chosen prior distribution
        return 1 / distance**2

    def prior(distance, L=1.35*u.kpc):
        return 1/(2*L**3) * distance**2 * np.exp(-distance / L)

    distance_grid = np.linspace(1, 12000, 12000) * u.pc

    parallax_true_grid = Distance(distance_grid).parallax.value

    posterior = np.array([likelihood(parallax_data.value, parallax_true, parallax_error.value) * prior(dist) 
                        for parallax_true, dist in zip(parallax_true_grid, distance_grid)])

    norm = np.trapz(posterior, distance_grid, axis=0)
    normalized_posterior = posterior / norm

    distances_mean = np.trapz(normalized_posterior * distance_grid[:, np.newaxis], distance_grid, axis=0)
    mode_indices = np.argmax(normalized_posterior, axis=0)
    distances_mode = distance_grid[mode_indices]

    cdf = np.cumsum(normalized_posterior, axis=0)
    #cdf /= cdf[-1, :]  # Normalize each column to make it a proper CDF

    # Find indices for median and percentiles
    median_indices = np.argmin(np.abs(cdf.value - 0.5), axis=0)
    lower_bound_indices = np.argmin(np.abs(cdf.value - 0.16), axis=0)
    upper_bound_indices = np.argmin(np.abs(cdf.value - 0.84), axis=0)

    distances_median = distance_grid[median_indices].value
    distances_lower_bound = distance_grid[lower_bound_indices].value
    distances_upper_bound = distance_grid[upper_bound_indices].value

    # distances_mean = distances_mean 
    # distances_mode = distances_mode 
    # distances_median = distances_median * distance_grid.unit
    # distances_lower_bound = distances_lower_bound * distance_grid.unit
    # distances_upper_bound = distances_upper_bound * distance_grid.unit

    data['distance_mean_bayes'] = distances_mean
    data['distance_lower_bound'] = distances_lower_bound
    data['distance_upper_bound'] = distances_upper_bound
    data['distance_median_bayes'] = distances_median
    data['distance_mode_bayes'] = distances_mode
    data['distance_naive'] = (1e3 / data['parallax']).value * u.pc
    data['distance_error_naive'] = (1e3 / data['parallax']**2 * data['parallax_error']).value * u.pc

    return data


