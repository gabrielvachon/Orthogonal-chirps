import numpy as np

class orthogonal_chirp_base:
    def __init__(self, fs, f0, f1, M, T, type):
        self.fs     = fs
        self.f0     = f0
        self.f1     = f1
        self.M      = M
        self.T      = T
        if type in ['up', 'down', 'hybrid']:
            self.type   = type
        else:
            raise(ValueError(f"Wrong input type : {type}. Available inputs : ['up', 'down', 'hybrid']"))
                
        self.bb_up, self.bb_down = self.generate_chirp_base_block(f0=self.f0, f1=self.f1, fs=self.fs, T=self.T, M=self.M)
        self.matrix = self.generate_chirp_matrix(chirps_up=self.bb_up, chirps_down=self.bb_down, fs=self.fs, T=self.T, M=self.M, type=self.type)
        self.uhb = self.unity_height_base(fs=self.fs, T=self.T, M=self.M)
        self.bases = self.generate_chirps()

    def generate_chirp_base_block(self, f0, f1, fs, T, M):
        N_CHIRP = int(fs * T / M)   # Number of sample of one chirp
        t = np.arange(N_CHIRP) / fs # Time array
        T_b = T / M                 # Time duration of one chirp
        chirps_up = np.zeros((M, N_CHIRP))      # Up sub-chirp array
        chirps_down = np.zeros((M, N_CHIRP))    # Down sub-chirp array
        for m in range(M):                      # For a set number of transmitters M
            f_s_m = f0 + m * (f1 - f0) / M   # Start frequency of sub-chirp m
            f_e_m = f_s_m + (f1 - f0) / M     # End frequency of sub-chirp m
            T_b = T / M                         # Sub-chirp period
            chirps_up[m] = np.sin( \
                    2 * np.pi * (f_s_m * (t - m * T_b) + \
                    ((f_e_m - f_s_m) * t ** 2) / (2 * T_b)))    # Calculate up sub-chirp m
            chirps_down[m] = np.sin( \
                    2 * np.pi * (f_e_m * (t - m * T_b) - \
                    ((f_e_m - f_s_m) * t ** 2) / (2 * T_b)))    # Calculate down sub-chirp m
        return chirps_up, chirps_down
    
    def generate_chirp_matrix(self, chirps_up, chirps_down, fs, T, M, type='hybrid'):
        N = int(fs * T)                 # Total number of samples
        N_CHIRP = int(fs * T / M)       # Number of sample of one chirp
        chirp_matrix = np.zeros((M, N)) # Chirp matrix of M channels by N samples

        if type == 'up':            # For "up" type sub-chirp base
            for m in range(M):
                for n in range(M):
                    t_c = int(n * N_CHIRP)      # Starting sample of sub-chirp on channel m
                    t_b = int(t_c + N_CHIRP)    # Ending sample of sub-chirp on channel m
                    chirp_matrix[m][t_c:t_b] = chirps_up[m] # Append up sub-chirps samples to channel m at m intervals
            return chirp_matrix

        elif type == 'down':        # For "down" type sub-chirp base
            for m in range(M):
                for n in range(M):
                    t_c = int(n * N_CHIRP)      # Starting sample of sub-chirp on channel m
                    t_b = int(t_c + N_CHIRP)    # Ending sample of sub-chirp on channel m
                    chirp_matrix[m][t_c:t_b] = chirps_down[m] # Append down sub-chirps samples to channel m at m intervals
            return chirp_matrix
        
        elif type == 'hybrid':      # For "hybrid" type sub-chirp base
            for m in range(M):
                for n in range(M):
                    t_c = int(n * N_CHIRP)      # Starting sample of sub-chirp on channel m
                    t_b = int(t_c + N_CHIRP)    # Ending sample of sub-chirp on channel m
                    if m % 2 == 0:  # For even rows
                        if n % 2 == 0:  # For even rows
                            chirp_matrix[m][t_c:t_b] = chirps_up[m]     # Append up sub-chirp samples to channel m at position n of m
                        else:           # For odd columns
                            chirp_matrix[m][t_c:t_b] = chirps_down[m]   # Append down sub-chirp samples to channel m at position n of m
                    else:           # For odd rows
                        if n % 2 == 0:  # For even columns
                            chirp_matrix[m][t_c:t_b] = chirps_down[m]   # Append down sub-chirp samples to channel m at position n of m
                        else:           # For odd columns
                            chirp_matrix[m][t_c:t_b] = chirps_up[m]     # Append up sub-chirp samples to channel m at position n of m
            return chirp_matrix
        
        raise(ValueError(f"Wrong input type : {type}. Available inputs : ['up', 'down', 'hybrid']"))

    def unity_height_base(self, fs, T, M):
        N = int(fs * T)                 # Total number of samples
        N_CHIRPS = int(fs * T / M)
        R = np.random.permutation(M)    # One random permutation of M samples from 0 -> (M - 1)
        R_i = np.zeros((M, M))          # Array of all circular shifted sets of R

        for i in range(len(R)):
            R_i[i] = np.roll(R, i)      # Circular shift of R

        psi = np.zeros((M, M, N))       # Array of unity-height pulses
        for i in range(M):
            for m in range(M):
                t_c = int(R_i[i][m] * N_CHIRPS)   # Starting sample of sub-chirp on channel m
                t_b = int(t_c + N_CHIRPS)         # Ending sample of sub-chirp on channel m
                psi[i][m][t_c:t_b] = 1              # Adding ones for range t_c to t_b

        return psi

    def generate_chirps(self):
        N = int(self.fs * self.T)                               # Total number of samples
        chirps = np.zeros((self.M, N))                          # Array for M constructed chirps
        for m in range(self.M):
            chirp = np.zeros(N)                                 # Initialize a single chirp
            for n in range(self.M):
                chirp = chirp + self.matrix[n] * self.uhb[m][n] # Muliply each channel by its unity height pulse and sum each one to make a chirp
            chirps[m] = chirp                                   # Assign chirp to chirp array
            
        return chirps
        
    def __str__(self):
        return (f"Orthogonal chirp\n\r"
                f"    Type: {self.type}\n\r"
                f"    Chirp duration (sec): {self.T}\n\r"
                f"    Transmitters (M): {self.M}\n\r"
                f"    Starting frequency: {self.f0}\n\r"
                f"    Ending frequency: {self.f1}")
    
    def __getitem__(self, key):
        return self.bases[key]