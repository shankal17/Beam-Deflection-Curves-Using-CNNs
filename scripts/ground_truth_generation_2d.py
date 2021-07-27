import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

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
    forces = beam[1]
    deflection_curve = np.zeros_like(x)

    # Apply principle of superposition
    for c, coordinate in enumerate(x):
        location_force = forces[c]
        deflection_curve += deflection(location_force, x, coordinate, flexural_rigidity)
    
    return np.stack((x, forces, deflection_curve))

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
    x = np.linspace(0, length, num=num_points)

    # Generate random loads corresponding to each height
    is_loaded = np.array(np.random.rand(x.shape[0]) < load_probability, dtype=np.int8)
    forces = is_loaded*25*(np.random.random(x.shape[0])*6 - 3)*np.flip(np.linspace(0.1, 1, x.shape[0])**4)
    # forces = is_loaded*10*(np.random.random(x.shape[0])*6 - 3)/np.sqrt(np.linspace(0.01, 2, x.shape[0]))
    print(forces)

    beam = np.stack((x, forces))
    deflected_beam = generate_deflected_beam(beam, flexural_rigidity)

    return deflected_beam

def generate_deformed_beam_single_force(length, flexural_rigidity, num_points, max_force):
    """returns final deformed beam from single force

    Parameters
    ----------
    length : float
        Length of the generated beam
    flexural_rigidity : float
        Flexural rigidity of the beam
    num_points: int
        Number of points in the generated beam
    max_force : float
        Maximum magnitude of the transverse force applied

    Returns
    -------
    ndarray
        Array containing beam deflection data
    """

    x = np.linspace(0, length, num=num_points)
    forces = np.zeros_like(x)
    force_index = np.random.randint(0, x.shape[0])
    forces[force_index] = np.random.random()*max_force*2 - max_force
    # print(forces)
    beam = np.stack((x, forces))
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
    ax = fig.add_subplot()
    deflection_curve = beam[2]

    # Normalize if desired
    if normalize:
        deflection_curve = (normalize_to/(np.amax(np.abs(deflection_curve))+1e-20))*deflection_curve

    # Plot the ting
    ax.plot(beam[0], deflection_curve, color='c')

    # Set limits
    ax.set_ylim([-0.5, 0.5])

    # Set Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')

    plt.show()

def generate_dataset(beam_length, flexural_rigidity, num_points, max_force, num_samples, base_save_name):
    """Generates a dataset of beams deflected by a single force

    Parameters
    ----------
    beam_length : Float
        Length of the beams
    flexural_rigidity : float
        Flexural rigidity of the generated beams
    num_pionts: int
        Number of points in each generated beam
    max_force : float
        Maximum possible magnitude of the transverse force applied
    num_samples : int
        Number of beams to generate
    base_save_name : string
        Name to save the beams to. "_i.npy" will be appended

    Returns
    -------
    ndarray
        Array containing beam deflection data
    """

    for i in tqdm(range(num_samples)):
        save_name = base_save_name + str(i) +".npy"
        beam = generate_deformed_beam_single_force(beam_length, flexural_rigidity, num_points, max_force)
        np.save(save_name, beam)

if __name__ == '__main__':
    beam_length = 0.5 # [m]
    side_length = 0.02 # [m]
    num_points = 51
    max_force = 5000 #[N]
    E = 69e9 # Modulus of elasticity for alloy 1100 [Pa]
    I = (side_length**4)/12 # Moment of inertia

    # beam = generate_deformed_beam_single_force(beam_length, E*I, num_points, max_force)
    # plot_deformed_beam(beam)
    generate_dataset(beam_length, E*I, num_points, max_force, 1000, 'data/train/beam_')
    generate_dataset(beam_length, E*I, num_points, max_force, 100, 'data/test/beam_')

