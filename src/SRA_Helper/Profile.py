import numpy as np

def stair_step_Vs_profile(Thickness: np.ndarray, 
                          Vs: np.ndarray):
    
    """
    Converts a layered soil profile into a stair-step profile for plotting.

    Each layer is represented as a horizontal step, with the same Vs value
    repeated at the top and bottom of that layer.

    Parameters
    ----------
    Thickness : np.ndarray
        Thickness of each soil layer (m). Shape: (n,)
    Vs : np.ndarray
        Shear wave velocity of each soil layer (m/s). Shape: (n,)

    Returns
    -------
    Depth : np.ndarray
        Depth values for the stair-step profile (m). Shape: (2n,)
    New_Vs : np.ndarray
        Vs values paired with each depth point (m/s). Shape: (2n,)
    """
    
    if len(Thickness) != len(Vs):
        raise ValueError("Thickness and Vs arrays must have the same length.")

    cumulative = np.cumsum(Thickness)
    tops = np.concatenate([[0], cumulative[:-1]])
    Depth = np.stack([tops, cumulative], axis=1).ravel()
    New_Vs = np.repeat(Vs, 2)

    return Depth, New_Vs

