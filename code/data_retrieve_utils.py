import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric, Galactic
from scipy.integrate import nquad
import astropy.coordinates as coord
from astropy.table import QTable, vstack
import astropy.units as u
from astropy.coordinates import Distance

def galcen_dice_to_gal(center, sides, units=u.kpc, fit_val=False):
    x, y, z = center
    dx, dy, dz = sides
    x_edges = np.array([x - dx/2, x + dx/2])
    y_edges = np.array([y - dy/2, y + dy/2])
    z_edges = np.array([z - dz/2, z + dz/2])

    xx, yy, zz = np.meshgrid(x_edges, y_edges, z_edges, indexing='ij')

    galactocentric_coords = SkyCoord(x=xx*units, y=yy*units, z=zz*units, frame=Galactocentric())
    galactic_coords = galactocentric_coords.transform_to(Galactic)
    ll, bb, dd = galactic_coords.l.value, galactic_coords.b.value, galactic_coords.distance.value

    d_lims = np.array([np.min(dd), np.max(dd)])
    b_lims = np.array([np.min(bb), np.max(bb)])
    l_lims = np.array([np.min(ll), np.max(ll)])

    l_min, l_max = l_lims[0], l_lims[1]

    if fit_val:
        volume_square = dx * dy * dz
        a = np.sin(np.deg2rad(b_lims[1] - b_lims[0]))*d_lims[0]
        c = d_lims[1] - d_lims[0]
        l_wrap_bool = (l_max - l_min) > 180
        if not l_wrap_bool:
            b = np.sin(np.deg2rad(l_lims[1] - l_lims[0]))*d_lims[0]
        else:
            b1 = np.sin(np.deg2rad(l_lims[0]))*d_lims[0]
            b2 = np.sin(np.deg2rad(360 - l_lims[1]))*d_lims[0]
            b = b1 + b2
    volume_sphere_el = a*b*c
    print(f'Volume ratio: {volume_square/volume_sphere_el}')

    return l_lims, b_lims, d_lims

def gal_cond_adql(l_lims, b_lims, d_lims):

    p_min, p_max = 1/d_lims[1], 1/d_lims[0]
    b_min, b_max = b_lims[0], b_lims[1]
    l_min, l_max = l_lims[0], l_lims[1]

    adql_str_bp = f"""(parallax BETWEEN {p_min} AND {p_max}) AND 
    (b BETWEEN {b_min} AND {b_max})"""

    l_wrap_bool = (l_max - l_min) > 180

    if not l_wrap_bool:
        adql_str_l = f"(l BETWEEN {l_min} AND {l_max})"
    else:
         l_min1, l_max1 = 0, l_min
         l_min2, l_max2 = l_max, 360
         adql_str_l = f"((l BETWEEN {l_min1} AND {l_max1}) OR (l BETWEEN {l_min2} AND {l_max2}))"

    adql_str = adql_str_bp + " AND " + adql_str_l

    return adql_str


def dice_adql_query(adql_cond_gal, N=None,
                    extra_cond = 'AND radial_velocity IS NOT null'):

    extra = ''
    if N is not None:
        top_get = f"TOP {N}"
        extra = "ORDER BY random_index"

    adql_query = f"""
    SELECT {top_get} ra, dec, l, b, parallax, pmra, pmdec, radial_velocity,
    ra_error, dec_error, parallax_error, pmra_error, pmdec_error, radial_velocity_error,
    parallax_over_error, astrometric_excess_noise,
    phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
    bp_rp,
    phot_g_mean_flux, phot_rp_mean_flux, phot_bp_mean_flux,
    phot_g_mean_flux_error, phot_rp_mean_flux_error, phot_bp_mean_flux_error,
    phot_bp_mean_flux_over_error, 
    phot_bp_rp_excess_factor, phot_rp_mean_flux_over_error,
    ag_gspphot, ag_gspphot_lower, ag_gspphot_upper,
    ebpminrp_gspphot, ebpminrp_gspphot_lower, ebpminrp_gspphot_upper

    FROM gaiadr3.gaia_source
    WHERE {adql_cond_gal} {extra_cond}
    {extra}
    """

    return adql_query


def get_galcen_table(data):
    
    dist = coord.Distance(parallax=u.Quantity(data['parallax']))
    c = coord.SkyCoord(ra=data['ra'],
                    dec=data['dec'],
                    distance=dist,
                    pm_ra_cosdec=data['pmra'],
                    pm_dec=data['pmdec'],
                    radial_velocity=data['radial_velocity'])

    galcen = c.transform_to(coord.Galactocentric())

    galcen_table = QTable([galcen.x, galcen.y, galcen.z,
                           galcen.v_x, galcen.v_y, galcen.v_z], names=('x', 'y', 'z', 'v_x', 'v_y', 'v_z'))
    # convert to kpc
    galcen_table['x'] = galcen_table['x'].to(u.kpc)
    galcen_table['y'] = galcen_table['y'].to(u.kpc)
    galcen_table['z'] = galcen_table['z'].to(u.kpc)

    galcen_table['v_x'] = galcen_table['v_x'].to(u.km / u.s)
    galcen_table['v_y'] = galcen_table['v_y'].to(u.km / u.s)
    galcen_table['v_z'] = galcen_table['v_z'].to(u.km / u.s)

    # add columns to data
    data['x'] = galcen_table['x']
    data['y'] = galcen_table['y']
    data['z'] = galcen_table['z']

    data['v_x'] = galcen_table['v_x']
    data['v_y'] = galcen_table['v_y']
    data['v_z'] = galcen_table['v_z']

    return data, galcen

def preprocess_data(data, distance=None):

    data['distance'] = 1e3 / data['parallax']

    if distance is None:
        distance = data['distance']
    else: 
        distance = data[distance]


    data['g_abs'] = data['phot_g_mean_mag'] - 5 * (np.log10(distance) - 1)
    #data['g_abs'] = data['phot_g_mean_mag'] + 5 * (np.log10(data['parallax']/1e3) + 1)
    #data['g_abs'] = data['phot_g_mean_mag'] + 5 * np.log10(1e3 / distance / 100.0)


    # data['phot_g_mean_mag_error'] = sigma_g_mag = (2.5 / np.log(10)) * (data['phot_g_mean_flux_error'] / data['phot_g_mean_flux'])
    # data['g_abs_error'] = np.sqrt(data['phot_g_mean_mag_error']**2 + 
    #                             (5 / (data['parallax'] * np.log(10)) * data['parallax_error'])**2)

    bp_magnitude = -2.5 * np.log10(data['phot_bp_mean_flux'])
    rp_magnitude = -2.5 * np.log10(data['phot_rp_mean_flux'])
    color_magnitude = bp_magnitude - rp_magnitude 
    cal = data['bp_rp'] - color_magnitude
    color_magnitude = color_magnitude + cal
    data['bp_rp_mag'] = color_magnitude

    # sigma_bp_mag = (2.5 / np.log(10)) * (data['phot_bp_mean_flux_error'] / data['phot_bp_mean_flux'])
    # sigma_rp_mag = (2.5 / np.log(10)) * (data['phot_rp_mean_flux_error'] / data['phot_rp_mean_flux'])
    # sigma_color = np.sqrt(sigma_bp_mag**2 + sigma_rp_mag**2)
    # data['bp_rp_mag_error'] = sigma_color

    return data



def calculate_square_centers_excluding_inner_circle(r, d, inner_radius):
    square_centers = []
    y = -r + d / 2  # Start from the bottom plus half the square's side

    while y <= r - d / 2:  # Iterate over rows within the semicircle
        # Calculate the horizontal range for this row
        x_max = -np.sqrt(r**2 - y**2)
        x = x_max + d / 2

        while x <= -x_max - (d / 2)*1:  # Iterate over columns within the semicircle
            # Exclude squares that fall within the inner circle
            if np.sqrt(x**2 + y**2) >= inner_radius + d/2:
                if x < 0:
                    square_centers.append((x, y))
            x += d

        y += d

    n = (1/2*np.pi*r**2)/(d**2)

    print("Number of squares: ", len(square_centers))

    return square_centers

def calculate_cube_centers_in_hemisphere(r, d, z_lim = 2):
    cube_centers = []
    # Iterate through layers along z
    for z in np.arange(z_lim, r, d):
        # Calculate radius of the circular slice at this z
        slice_radius = np.sqrt(r**2 - z**2)

        # Iterate over x and y within the circle slice
        for x in np.arange(-slice_radius, slice_radius, d):
            for y in np.arange(-slice_radius, slice_radius, d):
                if np.sqrt(x**2 + y**2) < slice_radius:
                    if z > 0:
                        if x < 0: 
                            cube_centers.append((x, y, z))

    return cube_centers



def bayesian_distance_estimator(data):

    parallax_data = data['parallax']
    parallax_error = data['parallax_error']

    def likelihood(parallax, parallax_true, parallax_error):
        return np.exp(-0.5 * ((parallax - parallax_true) / parallax_error)**2)

    def prior(distance):
        # Replace with the chosen prior distribution
        return 1 / distance**2

    def prior(distance, L=1.35*1e3*u.pc):
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

