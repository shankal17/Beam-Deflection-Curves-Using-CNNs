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

    z = beam[0]
    force_magnitudes = beam[1]
    force_bearings = beam[2]
    x_deflection_curve, y_deflection_curve = np.zeros_like(z), np.zeros_like(z)
    f_x = np.multiply(force_magnitudes, np.cos(force_bearings))
    f_y = np.multiply(force_magnitudes, np.sin(force_bearings))

    # Apply principle of superposition
    for c, coordinate in enumerate(z):
        location_f_x = f_x[c]
        location_f_y = f_y[c]
        x_deflection_curve += deflection(location_f_x, z, coordinate, flexural_rigidity)
        y_deflection_curve += deflection(location_f_y, z, coordinate, flexural_rigidity)
    
    return np.stack((z, force_magnitudes, force_bearings, x_deflection_curve, y_deflection_curve))

def generate_random_beam(length, flexural_rigidity, num_points, load_probability):
    """Generates beam with random loading condition

    Parameters
    ----------
    length : float
        Length of beam in some unit
    num_points : int
        number of points in beam
    load_probability : float
        Probability that a point will have a load applied to it
    
    Returns
    -------
    ndarray <float>
        Array consisting of 3 channels being location data, load data, deflection curve data
    """

    # Generate height data on beam
    z = np.linspace(0, length, num=num_points)

    # Generate random loads corresponding to each height
    is_loaded = np.array(np.random.rand(z.shape[0]) < load_probability, dtype=np.int8)
    force_magnitudes = is_loaded*np.random.random(z.shape[0])*(np.flip(np.linspace(0.1, 1, z.shape[0]))**2)*75
    # print(1/np.sqrt(np.linspace(0.1, 1, z.shape[0])))
    # force_magnitudes = is_loaded*np.random.random(z.shape[0])
    force_bearings = np.random.random(z.shape[0])*2*np.pi


    beam = np.stack((z, force_magnitudes, force_bearings))
    deflected_beam = generate_deflected_beam(beam, flexural_rigidity)

    return deflected_beam

def plot_deformed_beam(beam, normalize=False, normalize_to=2):
    """Plots deformed beam

    Parameters
    ----------
    beam : ndarray
        Array containing deformed beam information
    normalize : bool, optional
        Determines whether to normalize the deflections or not
    nomalize_to : float, optional
        number to normalize the deflections to
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x_deflection_curve = beam[3]
    y_deflection_curve = beam[4]

    # Normalize if desired
    if normalize:
        x_deflection_curve = (normalize_to/(np.amax(np.abs(x_deflection_curve))+1e-5))*x_deflection_curve
        y_deflection_curve = (normalize_to/(np.amax(np.abs(y_deflection_curve))+1e-5))*y_deflection_curve

    # Plot the ting
    ax.plot(x_deflection_curve, y_deflection_curve, beam[0], color='c')

    # Create bounding box
    max_range = beam[0][-1]
    Xb = 0.25*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() # x coordinates of bounding box
    Yb = Xb # y coordinates of bounding box
    Zb = 0.25*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(max_range) # z coordinates of bounding box
    
    # plot bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    # Set Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


if __name__ == '__main__':
    beam = generate_random_beam(10, 1200, 101, 0.05)
    plot_deformed_beam(beam, normalize=True)
