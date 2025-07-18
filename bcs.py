import numpy as np

# --- Input/output/target functions ---


def input_val(y0_in, sigma_in, offset_in, coords):
    """
    Return both line (1D) and spatial (2D) input functions centered at y0_in.

    Parameters:
    - y0_in: float, center location in y
    - sigma_in: float, Gaussian width
    - offset_in: float, additive offset

    Returns:
    - list of functions: [1D function of y, 2D function of x]
    """
    def input_val_1d(y: np.ndarray) -> float:
        """Return 1d value input function."""
        return np.exp(-(y - y0_in)**2 / sigma_in) + offset_in

    def input_val_2d(y: np.ndarray) -> float:
        """Return 2d value input function."""
        return np.exp(-(y[1] - y0_in)**2 / sigma_in) + offset_in

    input_val_array = np.array([input_val_1d(y) for y in coords[:, 1]])

    return input_val_1d, input_val_2d, input_val_array


def output_val(coords):
    """
    Return line (1D) output function as zeros.

    Parameters:
    None

    Returns:
    - list of functions: [1D function of y, 2D function of x]
    """
    def output_val_1d(y: np.ndarray) -> float:
        """Return 1d value output function."""
        return 0.0

    def output_val_2d(y: np.ndarray) -> float:
        """Return 2d value input function."""
        return np.zeros(y.shape[1])

    output_val_array = np.array([output_val_1d(y) for y in coords[:, 1]])

    return [output_val_1d, output_val_2d, output_val_array]


def target_val(y0_target, sigma_target, offset_target, coords):

    def target_val_1d(y: np.ndarray) -> float:
        """Return 1d value target function."""
        return np.exp(-(y - y0_target)**2 / sigma_target) + offset_target

    target_val_array = np.array([target_val_1d(y) for y in coords[:, 1]])

    return target_val_1d, target_val_array