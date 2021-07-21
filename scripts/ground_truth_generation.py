import numpy as np
import matplotlib.pyplot as plt

def sing(coefficient, x, a): # Singularity functions
    """returns result of the singularity function for a transverse force

    Parameters
    ----------
    x : ndarray <float>
        location data that the function is acting on
    a : float
        critical point of singularity function

    Returns
    -------
    array
        output of singularity function    
    """

    shear_data = []
    moment_data = []
    if isinstance(x, np.ndarray):
        for location in x:
            if location < a:
                shear_data.append(0)
                moment_data.append(0)
            else:
                shear_data.append(coefficient)
                moment_data.append(coefficient*(location - a))
    else:
        raise Exception('incompatible type')
    
    return shear_data, moment_data

def solve_deflection_curves(x, internal_moment_data, flexural_rigidity):
    """Calculates the deflection curve for a cantilevered beam

    Parameters
    ----------
    x : ndarray <float>
        location data which the moment is integrated over
    internal_moment_data : ndarray <float>
        Internal moments associated with each location in the beam
    flexural_rigidity : float
        Flexural reigidity of the beam (constant cross-section)
    
    Returns
    -------
    ndarray <float>
        Slope of the deformed beam
    ndarray <float>
        Deflections of the deformed beam 
    """

    integrand = internal_moment_data / flexural_rigidity
    dx = x[-1] - x[-2]
    theta = [0] # B.C., slope is zero at the base (cantilevered beam) 
    for i in range(len(x)-1):
        additional_area = (dx/2)*(integrand[i]+integrand[i+1])
        theta.append(theta[-1] + additional)
    delta = [0] # B.C., deflection is zero at the base (batilevered beam)
    for i in range(len(x)-1):
        additional_area = (dx/2)*(theta[i]+theta[i+1])
        delta.append(delta[-1] + additional_area)

    return theta, delta

def generate_internal_load_data(x, load_vector):
    """Calculates the internal shear force and bending moment data (cantilevered beam)

    Parameters
    ----------
    x : ndarray <float>
        location data that the function is acting on
    load_vector : ndarray <float>
        Array of transverse forces that correspond to the location data
    
    Returns
    -------
    ndarray <float>
        Array of internal shear force corresponding to each location in "x"
    ndarray <float>
        Array of internal bending moments corresponding to each location in "x"
    """

    pass
