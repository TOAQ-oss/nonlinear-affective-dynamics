import numpy as np
import matplotlib.pyplot as plt
from model import EmotionalAgent
from profiles import get_profile
from pink_noise import generate_pink_noise

# --- Simulation Parameters ---
DURATION = 400
T_SOCIAL_START = 120
T_STRESS_START = 220
PROFILES = ["Extremely Sociable", "Asocial", "Very Calm", "Very Stressed", "Human-Like"]

# --- Generate Scenarios (Inputs F(t)) ---
# 6 inputs corresponding to the 6 components
inputs = np.zeros((DURATION, 6))

# Phase 1: Calm (Noise only)
# Phase 2: Social Stimulus (Input index 2)
inputs[T_SOCIAL_START:T_STRESS_START, 2] = 0.9 

# Phase 3: Cognitive Load / Stress (Input index 1)
inputs[T_STRESS_START:, 1] = 0.9 
# Note: For Stressed profile, this input will trigger the collapse via negative alpha

# --- Run Simulation ---
plt.figure(figsize=(12, 6))

for profile_name in PROFILES:
    # 1. Init Agent
    b, alpha = get_profile(profile_name)
    agent = EmotionalAgent(profile_name, b, alpha)
    
    # 2. Generate Noise
    zeta = generate_pink_noise(DURATION, sigma=0.15)
    
    # 3. Loop
    history_E = []
    
    for t in range(DURATION):
        # Current inputs + slight random variations
        current_inputs = inputs[t] + np.random.normal(0, 0.05, 6)
        current_inputs = np.clip(current_inputs, 0, 1)
        
        state = agent.step(current_inputs, zeta[t])
        history_E.append(state["E"])
        
    # 4. Plot
    plt.plot(history_E, label=profile_name, linewidth=1.5)

# --- Formatting match paper Fig 1 ---
plt.title("Comparison of emotional state E(t) across profiles")
plt.xlabel("t (steps)")
plt.ylabel("E(t)")
plt.axvline(x=T_SOCIAL_START, color='k', linestyle='--', alpha=0.3, label="Social Stimulus")
plt.axvline(x=T_STRESS_START, color='r', linestyle='--', alpha=0.3, label="Stress Event")
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()

print("Simulation complete. Saving figure...")
plt.savefig("simulation_results.png")
plt.show()