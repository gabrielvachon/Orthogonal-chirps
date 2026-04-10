from matplotlib import pyplot as plt
from scipy import signal
import numpy as np
import orthogonal_chirp

M = 4
T = 2
f0 = 4000
f1 = 10000
fs = 48000
chirp = orthogonal_chirp.orthogonal_chirp_base(fs, f0, f1, M, T, 'hybrid')

plt.figure()
Sxx_tot = 0
for m in chirp.matrix:
   f, s, Sxx = signal.stft(x=m, fs=fs, nperseg=1024, noverlap=512)
   Sxx_tot = Sxx_tot + Sxx
plt.ylim([f0, f1])
plt.pcolormesh(s, f, np.abs(Sxx_tot))


for m in range(M):
   fig, ax = plt.subplots(M)
   fig.suptitle(f"Unity height pulse {m}")
   for n in range(M):
      ax[M-1-n].plot(chirp.uhb[m][n])
      ax[M-1-n].set_ylabel(f"Band {n}")


fig, ax = plt.subplots(M) 
for m in range(M):
   f, s, Sxx = signal.stft(x=chirp[m], fs=fs, nperseg=1024, noverlap=512)
   ax[m].set_title(f"Chirp {m}")
   ax[m].set_ylim([f0, f1])
   ax[m].pcolormesh(s, f, np.abs(Sxx))
      
plt.show()

print(chirp)