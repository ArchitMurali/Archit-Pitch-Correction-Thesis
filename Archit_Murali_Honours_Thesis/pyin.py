import numpy as np
from scipy.signal import medfilt

# # Synthesis of a 1 Hz Signal
# def f(x):
#     """
#     This function synthesizes a 1 Hz sine wave with an exponentially decaying amplitude envelope.

#     Args:
#         x: A numpy array representing time points.

#     Returns:
#         A numpy array representing the synthesized signal values at each time point.
#     """
#     f_0 = 1
#     envelope = lambda x: np.exp(-x)  # Sine wave with exponentially diminishing amplitude
#     return np.sin(x * np.pi * 2 * f_0) * envelope(x)

# Synthesis of a Signal with More Amplitude Fluctuations and Noise
def f(x):
    """
    This function synthesizes a signal with more amplitude fluctuations and noise.

    Args:
        x: A numpy array representing time points.

    Returns:
        A numpy array representing the synthesized signal values at each time point.
    """
    f_0 = 1
    envelope = lambda x: np.exp(-x)  # Exponentially decaying envelope
    noise = np.random.normal(0, 0.1, len(x))  # Gaussian noise
    amplitude_modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * x)  # Amplitude modulation
    return amplitude_modulation * np.sin(x * np.pi * 2 * f_0) * envelope(x) + noise


# Autocorrelation Function (ACF)
def ACF(f, W, t, lag):
    """
    This function calculates the Autocorrelation Function (ACF) of a signal segment.

    Args:
        f: A numpy array representing the signal.
        W: The window size (number of samples) for the ACF calculation.
        t: The starting time index of the signal segment for ACF.
        lag: The time lag (number of samples) to shift the copy within the window.

    Returns:
        The ACF value for the given signal segment, window size, and lag.
    """
    return np.sum(f[t: t + W] * f[lag + t: lag + t + W])

# Difference Function (DF)
def DF(f, W, t, lag):
    """
    This function calculates the Difference Function (DF) based on the ACF.

    Args:
        f: A numpy array representing the signal.
        W: The window size (number of samples) used for ACF calculation.
        t: The starting time index of the signal segment for ACF.
        lag: The time lag (number of samples) to shift the copy within the window.

    Returns:
        The DF value for the given signal segment, window size, and lag.

    Note:
        This implementation calculates the DF based on the ACF as shown in the video
        for simplicity. A more efficient approach might exist.
    """
    return ACF(f, W, t, 0) + ACF(f, W, t + lag, 0) - (2 * ACF(f, W, t, lag))

# Cumulative Mean Normalized Difference Function (CMNDF)
def CMNDF(f, W, t, lag):
    """
    This function calculates the Cumulative Mean Normalized Difference Function (CMNDF).

    Args:
        f: A numpy array representing the signal.
        W: The window size (number of samples) used for ACF calculation.
        t: The starting time index of the signal segment for ACF.
        lag: The time lag (number of samples) to shift the copy within the window.

    Returns:
        The CMNDF value for the given signal segment, window size, and lag.

    Note:
        This implementation uses a simplified approach for calculating the local mean
        compared to the mathematical explanation. It directly sums past DF values
        instead of maintaining a separate window. 
    """
    if lag == 0:
        return 1
    return DF(f, W, t, lag) / np.sum([DF(f, W, t, j + 1) for j in range(lag)]) * lag

# Post-processing: Median Filtering
def median_filtering(pitch_candidates, kernel_size):
    """
    Apply median filtering to smooth pitch candidates.

    Args:
        pitch_candidates: A list of pitch candidates.
        kernel_size: Size of the median filter kernel.

    Returns:
        Smoothed pitch candidates.
    """
    if len(pitch_candidates) == 0:
        return []
    elif len(pitch_candidates) == 1:
        return pitch_candidates
    else:
        return medfilt(pitch_candidates, kernel_size)

# Pitch Detection Function with Enhanced Accuracy
def detect_pitch_enhanced(f, W, t, sample_rate, bounds, kernel_size=3):
    """
    This function detects the pitch of a signal with enhanced accuracy.

    Args:
        f: A function representing the signal to analyze.
        W: The window size (number of samples) for ACF calculation.
        t: The starting time index of the signal segment for analysis (usually 1).
        sample_rate: The sampling rate of the signal (samples per second).
        bounds: A tuple representing the minimum and maximum lag values to search.
        kernel_size: Size of the median filter kernel for post-processing (default is 3).

    Returns:
        The estimated pitch of the signal in Hertz (Hz).
    """
    # Step 1: Probabilistic YIN pitch estimation
    CMNDF_Vals = [CMNDF(f, W, t, i) for i in range(*bounds)]
    pitch_candidates = [sample_rate / (np.argmin(CMNDF_Vals) + bounds[0])]  # Initial pitch candidates
    
    # Step 2: Post-processing with median filtering
    smoothed_pitch_candidates = median_filtering(pitch_candidates, kernel_size)
    
    # Step 3: Select the median pitch value
    if len(smoothed_pitch_candidates) > 0:
        estimated_pitch = np.median(smoothed_pitch_candidates)
    else:
        estimated_pitch = 0.0
    
    return estimated_pitch

# Example Usage
sample_rate = 500
start = 0
end = 5
num_samples = int(sample_rate * (end - start) + 1)
window_size = 200
bounds = [20, num_samples // 2]

x = np.linspace(start, end, num_samples)
print(detect_pitch_enhanced(f(x), window_size, 1, sample_rate, bounds))

# Output varies due to random noise but is close to 1 Hz