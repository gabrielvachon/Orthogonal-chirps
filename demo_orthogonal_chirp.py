from matplotlib import pyplot as plt
from scipy import signal
import numpy as np
import orthogonal_chirp

M = 8
T = 1
f0 = 5500
f1 = 9500
fs = 48000
chirp = orthogonal_chirp.orthogonal_chirp_base(fs=fs, f0=f0, f1=f1, M=M, T=T, type='hybrid', method='hyper', window='kaiser', win_b=2, optimal=True, max_candidate=100)

chirp.write(path="./")

plt.figure()
Sxx_tot = 0
for m in chirp.matrix:
   f, s, Sxx = signal.stft(x=m, fs=fs, nperseg=1024, noverlap=512)
   Sxx_tot = Sxx_tot + Sxx
plt.ylim([f0, f1])
plt.pcolormesh(s, f, np.abs(Sxx_tot))
plt.figure()

Sxx_tot = 0
for m in chirp.bb_up:
   f, s, Sxx = signal.stft(x=m, fs=fs, nperseg=1024, noverlap=512)
   Sxx_tot = Sxx_tot + Sxx
plt.ylim([f0, f1])
plt.pcolormesh(s, f, np.abs(Sxx_tot))

fig, ax = plt.subplots(M) 
for m in range(M):
   f, s, Sxx = signal.stft(x=chirp[m], fs=fs, nperseg=1024, noverlap=512)
   ax[m].set_title(f"Chirp {m}")
   ax[m].set_ylim([f0, f1])
   ax[m].pcolormesh(s, f, np.abs(Sxx))
      
plt.show()

time = np.arange(start=-len(chirp[0])/fs, stop=(len(chirp[0])-1)/fs, step=1/fs)
m_idx = 0
n_idx = 0
for m in chirp:
    n_idx = 0
    fig, ax = plt.subplots(M)
    for n in chirp:
        corr = signal.correlate(in1=m, in2=n, method='fft', mode='full') / (np.sqrt(np.sum(m ** 2)) * np.sqrt(np.sum(n ** 2)))
        
        if m_idx == n_idx:
            ax[n_idx].set_title(f"Autocorrelation for chirp {n_idx}")
            ax[n_idx].plot(time, corr, "C3")
        else:
            ax[n_idx].set_title(f"Cross-correlation with chirp {n_idx}")
            ax[n_idx].plot(time, corr, "C0")
        ax[n_idx].set_ylim(-1.1, 1.1)
        n_idx += 1
    plt.show()
    m_idx += 1

print(chirp)