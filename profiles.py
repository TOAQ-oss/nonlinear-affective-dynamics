import numpy as np

# Mapping des indices pour la lisibilité
# 0: Base, 1: Cog, 2: Soc, 3: Mot, 4: Mem, 5: Sens
IDX_BASE = 0
IDX_COG = 1
IDX_SOC = 2
IDX_MOT = 3
IDX_MEM = 4
IDX_SENS = 5

def get_profile(name):
    # Default Architecture (6 components)
    # Inputs vector size depends on how we map F(t) to components.
    # Simplified for demo: 1-to-1 mapping for sensitivity.
    
    # Base configuration (Human-Like)
    b = np.array([0.5, 0.3, 0.4, 0.3, 0.2, 0.4]) 
    
    # Alpha matrix (Sensitivity) - Simplified diagonal for clarity
    # Diagonal means Input[i] affects Component[i]
    alpha = np.eye(6) * 0.5 
    
    if name == "Extremely Sociable":
        # Section 4.0.1: High social base (b3) and high sensitivity
        b[IDX_SOC] = 0.8  # Was 0.4
        alpha[IDX_SOC, IDX_SOC] = 0.9
        
    elif name == "Asocial":
        # Section 4.0.2: Low social base and sensitivity
        b[IDX_SOC] = 0.1
        alpha[IDX_SOC, IDX_SOC] = 0.05
        
    elif name == "Very Stressed":
        # Section 4.0.4: Low cognitive base, negative sensitivity to load
        b[IDX_COG] = 0.1
        alpha[IDX_COG, IDX_COG] = -0.8 # Collapse under load
        
    elif name == "Very Calm":
        # Section 4.0.3: Moderate base, very low sensitivity (inertia)
        alpha = np.eye(6) * 0.1
        
    return b, alpha