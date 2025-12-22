import numpy as np

class EmotionalAgent:
    def __init__(self, profile_name, b, alpha, T=1.0, beta=0.1, eta=0.05):
        """
        Args:
            profile_name (str): Name of the psychological profile.
            b (np.array): Base bias vector (Eq 1).
            alpha (np.array): Sensitivity matrix (Eq 1).
            T (float): Temperature for Softmax (Eq 5).
            beta (float): Hysteresis/Dissipation factor (Eq 18).
            eta (float): Sensitivity to stochastic modulation.
        """
        self.name = profile_name
        self.b = np.array(b)           # Baseline biases
        self.alpha = np.array(alpha)   # Influence weights
        self.T = T
        self.beta = beta
        self.eta = eta
        
        # State variables
        self.n_components = len(b)
        self.E_h = 0.0                 # Historical state (Mood)
        self.E = 0.0                   # Instantaneous Valence
        self.E_s = 0.0                 # Intensity (Global Gain)
    
    def softmax(self, x):
        """Eq 5: Softmax with Temperature T."""
        e_x = np.exp((x - np.max(x)) / self.T) # Stable softmax
        return e_x / e_x.sum()

    def step(self, inputs, zeta_t):
        """
        Performs one simulation step.
        
        Args:
            inputs (np.array): Vector of cognitive function values F(t) [0, 1].
            zeta_t (float): Current value of 1/f noise (Stochastic Modulation).
            
        Returns:
            dict: Current state (E, E_out, E_s, etc.)
        """
        # 1. Compute Omega (Eq 1): Integration of weights
        # Note: We assume inputs are mapped to weights via alpha matrix
        # omega_i = b_i + sum(alpha * input)
        omega = self.b + np.dot(self.alpha, inputs)
        
        # 2. Apply Softmax Competition (Eq 5)
        # We inject micro-variability here or at E level depending on Section 5 implementation.
        # Here we follow the global modulation approach.
        lambdas = self.softmax(omega)
        
        # 3. Compute Valence E(t) (Eq 7)
        self.E = np.sum(lambdas * omega)
        
        # 4. Compute Intensity E_s(t) (Eq 13)
        self.E_s = np.log(1 + np.abs(self.E))
        
        # 5. Compute Hysteresis/Mood E_h (Eq 18)
        self.E_h = self.beta * self.E + (1 - self.beta) * self.E_h
        
        # 6. Compute Global Output E_out (Eq 17) with Stochastic Modulation
        # E_out = tanh(E(t) + zeta(t))
        E_out = np.tanh(self.E + (self.eta * zeta_t))
        
        return {
            "E": self.E,
            "E_s": self.E_s,
            "E_h": self.E_h,
            "E_out": E_out,
            "lambdas": lambdas
        }