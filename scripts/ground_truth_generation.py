import numpy as np
import matplotlib.pyplot as plt

def deflection(P, x, a, flexural_rigidity): # Singularity functions
    """returns deflection for the considered beam and transverse force

    Parameters
    ----------
    P : Float
        transverse force magnitude
    x : ndarray <float>
        Array of location data
    a: float
        Location of transverse force on beam
    flexural_rigidity : float
        Flexural rigidity of beam

    Returns
    -------
    ndarray
        Array containing beam deflection data
    """

    deflection_data = []
    if isinstance(x, np.ndarray):
        for location in x:
            common_factor = P/(6*flexural_rigidity)
            if location < a:
                deflection_data.append((location**2)*common_factor*(3*a - location))
            else:
                deflection_data.append((a**2)*common_factor*(3*location - a))
    else:
        raise Exception('Incompatible type, only use numpy arrays')
    
    return deflection_data

def generate_deflected_beam(beam, flexural_rigidity):
    """Calculates the internal shear force and bending moment data (cantilevered beam)

    Parameters
    ----------
    input : ndarray <float>
        2 channel array where the first is location data and the second is load data
    
    Returns
    -------
    ndarray <float>
        Array consisting of 3 channels being location data, load data, deflection curve data
    """

    x = beam[0]
    loads = beam[1]
    deflection_curve = np.zeros_like(x)

    # Apply principle of superposition
    for c, coordinate in enumerate(x):
        location_load = loads[c]
        deflection_curve += deflection(location_load, x, coordinate, flexural_rigidity)
    
    return np.stack((x, loads, deflection_curve))


if __name__ == '__main__':
    x = np.linspace(0, 1, 101)
    loads = np.zeros_like(x)
    loads[50] = 2
    loads[75] = -1.125
    beam = np.stack((x, loads))
    # print(generate_deflected_beam(beam, 1e1))
    plt.plot(x, generate_deflected_beam(beam, 1e1)[2])
    plt.ylim([-0.005, 0.005])
    plt.show()