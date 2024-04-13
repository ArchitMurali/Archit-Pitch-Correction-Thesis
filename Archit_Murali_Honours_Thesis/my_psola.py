import numpy as np
import matplotlib.pyplot as plt

def synthesise_signal(duration, sampling_rate, frequency=100):
    # Synthesize a test signal.
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
    envelope = np.exp(-t)  # Exponentially decaying amplitude envelope
    return np.sin(2 * np.pi * frequency * t) * envelope

def psola(pitch, time_scale_factor, duration, sampling_rate):
    # Apply PSOLA algorithm for time-scale modification.
    input_signal = synthesise_signal(duration, sampling_rate)
    input_length = len(input_signal)
    output_length = int(input_length * time_scale_factor)
    output_signal = np.zeros(output_length)
    
    # Determine frame size and overlap
    frame_size = int(pitch)  # Simple approach for frame size based on pitch
    overlap = int(frame_size * 0.5)  # 50% overlap
    
    # Iterate over frames
    for i in range(0, input_length - frame_size, frame_size - overlap):
        # Adjust frame size for the last frame
        frame_end = min(i + frame_size, input_length)
        frame = input_signal[i:frame_end]
        
        # Stretch or compress the frame
        if time_scale_factor > 1:
            output_frame = np.interp(np.arange(0, len(frame), 1 / time_scale_factor), np.arange(len(frame)), frame)
        else:
            output_frame = frame[::int(1 / time_scale_factor)]
        
        # Overlap and add
        output_start = int(i * time_scale_factor)
        output_end = min(output_start + len(output_frame), output_length)
        output_signal[output_start:output_end] += output_frame[:output_end - output_start]
    
    # Normalize the output signal
    output_signal /= np.max(np.abs(output_signal))
    
    return output_signal

# Define parameters
duration = 1  # Duration of the signal in seconds
sampling_rate = 44100  # Sampling rate in Hz
pitch = 100  # Example pitch value (in samples)
time_scale_factor = 1.5  # Example time scale factor

# Synthesize original signal
original_signal = synthesise_signal(duration, sampling_rate)

# Apply PSOLA algorithm to modify the signal
modified_signal = psola(pitch, time_scale_factor, duration, sampling_rate)

# Generate time axis for the modified signal
t_modified = np.linspace(0, duration * time_scale_factor, len(modified_signal), endpoint=False)

# Plot both original and modified signals
plt.plot(np.linspace(0, duration, len(original_signal), endpoint=False), original_signal, label='Original Signal')
plt.plot(t_modified, modified_signal, label='Modified Signal')
plt.title('Original and Modified Signals')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
