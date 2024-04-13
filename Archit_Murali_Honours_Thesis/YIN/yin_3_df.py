import numpy as np

# Synthesis of a 1 Hz Signal
def f(x):
    f_0 = 1

    return np.sin(x * np.pi * 2 * f_0)

# Auto Correlation Function. Takes Signal (f), Window Size (W), 
# Timestep (t), No. of samples to shift the copy by (lag)
def ACF(f,W,t,lag):
    return np.sum(
        f[t : t + W] *
        f[lag + t : lag + t + W]
    )


def DF(f,W,t,lag):
    return ACF(f, W, t, 0) + ACF(f, W, t + lag, 0) - (2 * ACF(f, W, t, lag))

def detect_pitch(f,W,t,sample_rate,bounds):
    DF_Vals = [DF(f,W,t,i) for i in range(*bounds)]
    sample = np.argmin(DF_Vals) + bounds[0]
    return sample_rate / sample

sample_rate = 500
start = 0
end = 5
num_samples = int(sample_rate * (end - start) + 1)
window_size = 200
bounds = [20, num_samples // 2]

x = np.linspace(start, end, num_samples)
print(detect_pitch(f(x), window_size, 1, sample_rate, bounds))