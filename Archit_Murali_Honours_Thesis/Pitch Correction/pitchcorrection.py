# Import Libraries
import pyaudio              # Audio input and output
import wave                 # Reading and writing WAV files
import librosa              # Music and audio analysis
from pathlib import Path    # Handling file paths
import soundfile as sf      # Reading and writing sound files
import psola                # Pitch shifting
import numpy as np          # Numerical operations
import scipy.signal as sig  # Signal processing
import simpleaudio as sa    # Playing audio files

######################################################################################################################

# Audio Input

# Set Up Parameters
FORMAT = pyaudio.paInt16  # Format of audio samples (16-bit signed integers)
CHANNELS = 1              # Number of audio channels (1 for mono, 2 for stereo)
RATE = 44100              # Sample rate (samples per second)
CHUNK = 1024              # Number of frames per buffer
RECORD_SECONDS = 20       # Duration of recording in seconds

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open an Audio Stream
stream = p.open(format=FORMAT,
                 channels=CHANNELS,
                 rate=RATE,
                 input=True,
                 frames_per_buffer=CHUNK)

# Record Audio
print("Recording STARTED.")
print("Recording for ",RECORD_SECONDS," seconds...")

frames = []

for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Recording Complete.")

# Close the Audio Stream
stream.stop_stream()
stream.close()

# Save the Recording
WAVE_OUTPUT_FILENAME = "recorded_audio.wav"

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print("Recording saved as", WAVE_OUTPUT_FILENAME)

# Clean Up
p.terminate()

######################################################################################################################

# Pitch Correction

print("Correcting Pitch...")

# Correct the pitch to the nearest note in the C Major scale
def correct(f0): 
    if np.isnan(f0):
        return np.nan
    c_major_degrees = librosa.key_to_degrees('c:maj') # Setting scale to C Major
    c_major_degrees = np.concatenate((c_major_degrees,[c_major_degrees[0] + 12]))

    midi_note = librosa.hz_to_midi(f0)
    degree = midi_note % 12
    closest_degree_id = np.argmin(np.abs(c_major_degrees - degree))

    degree_difference = degree - c_major_degrees[closest_degree_id]

    midi_note -= degree_difference # To ensure correct octave

    return librosa.midi_to_hz(midi_note)

# Correct the pitch of an array of fundamental frequencies
def correct_pitch(f0): 
    corrected_f0 = np.zeros_like(f0)
    for i in range(f0.shape[0]):
        corrected_f0[i] = correct(f0[i])
    smoothed_corrected_f0 = sig.medfilt(corrected_f0, kernel_size=11)

    smoothed_corrected_f0[np.isnan(smoothed_corrected_f0)] = corrected_f0[np.isnan(smoothed_corrected_f0)]

    return smoothed_corrected_f0

# Correct the pitch of an audio signal
def autotune(y, sr):
    # Track Pitch
    frame_length = 2048
    hop_length = frame_length // 4
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')
    f0, _, _ = librosa.pyin(y, frame_length = frame_length,
                             hop_length = hop_length,
                             sr = sr,
                             fmin = fmin,
                             fmax = fmax)
    
    # Calculte Desired Pitch
    corrected_f0 = correct_pitch(f0)

    # Pitch Shifting    
    return psola.vocode(y, sample_rate = int(sr),
                         target_pitch=corrected_f0,
                         fmin=fmin, fmax=fmax)


y, sr = librosa.load("recorded_audio.wav", sr=None, mono=False) # Loading the file

if y.ndim > 1: # Processing only one channel if there is more than one
    y = y[0, :]

pitch_corrected_y = autotune(y, sr) # Calling autotune

filepath = Path("recorded_audio.wav")

output_filepath = filepath.parent / (filepath.stem + "_pitch_corrected" + filepath.suffix) # Defining Output Filepath
sf.write(str(output_filepath), pitch_corrected_y, sr) # Writing to output file

######################################################################################################################

# Playback Pitch-Corrected File

print("Playing Pitch Corrected Audio...")

filename = 'recorded_audio_pitch_corrected.wav'
wave_obj = sa.WaveObject.from_wave_file(filename)
play_obj = wave_obj.play()
play_obj.wait_done()  # Wait until sound has finished playing

print("Done :)")