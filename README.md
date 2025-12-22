# Non-Linear Affective Dynamics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Organization](https://img.shields.io/badge/Organization-TOAQ-black)](https://toaq.fr)

> **Official Python implementation of the paper:** > [*Non-Linear Computational Modeling of Emotional-Like States in Artificial Agents*](https://research.toaq.fr/index/Non-Linear-Computational-Modeling-of-Emotional-Like-States-in-Artificial-Agents)  
> **Author:** Côme Bruneteau (TOAQ)

## 📌 Overview

This repository provides a unified computational framework for generating **non-rule-based emotional dynamics** in artificial agents. Moving beyond static symbolic approaches, this model leverages **Control Theory** and **Non-Linear Dynamics** to simulate stable, adaptive, and biologically plausible affective states.

### Key Features
* **Dynamic Stability:** Uses Lipschitz-continuous functions (Softmax, Tanh) to ensure system robustness.
* **Hysteresis & Inertia:** Implements emotional memory ($\beta$ parameter) to simulate resilience and mood persistence.
* **Stochastic Modulation:** Integrates endogenous $1/f$ noise (Pink Noise) to ensure behavioral uniqueness and prevent deterministic redundancy.
* **Resource Allocation:** Models "feeling" as a global gain control signal ($E_s$) that saturates cognitive processing under high intensity.

## 📂 Structure

The codebase is modular and designed for easy integration:

* `model.py`: Core logic of the `EmotionalAgent` (Equations 1-18 from the paper).
* `pink_noise.py`: Generator for endogenous stochastic modulation ($1/f$ noise).
* `profiles.py`: Configuration of psychological profiles (e.g., *Extremely Sociable*, *Very Stressed*).
* `simulation.py`: Main script to reproduce the paper's figures and run scenarios.

## 🚀 Quick Start

### Installation

```bash
git clone [https://github.com/TOAQ/nonlinear-affective-dynamics.git](https://github.com/TOAQ/nonlinear-affective-dynamics.git)
cd nonlinear-affective-dynamics
pip install -r requirements.txt
```

### Reproduce Paper Results
Run the simulation script to generate the emotional trajectories and phase space analysis:

```bash
python simulation.py
```

This will generate a simulation_results.png file comparing the 5 psychological profiles.

## 💻 Usage
You can easily integrate the EmotionalAgent into your own cognitive architecture:


```python
import numpy as np
from model import EmotionalAgent
from pink_noise import generate_pink_noise

# 1. Define agent configuration (e.g., a 'Stressed' profile)
b = np.array([0.5, 0.1, 0.4, 0.3, 0.2, 0.4]) # Base biases
alpha = np.eye(6) * 0.5                      # Sensitivity matrix
alpha[1, 1] = -0.8                           # Negative sensitivity to Cognitive Load

# 2. Initialize Agent
agent = EmotionalAgent("Custom Agent", b, alpha, T=1.0, beta=0.2)

# 3. Simulation Loop
zeta = generate_pink_noise(num_steps=100) # Generate internal noise

for t in range(100):
    # Simulated inputs from environment (normalized 0-1)
    inputs = np.random.rand(6) 
    
    # Update state
    state = agent.step(inputs, zeta_t=zeta[t])
    
    print(f"Time {t}: Valence E={state['E']:.2f}, Intensity Es={state['E_s']:.2f}")
```

## 📜 Citation
If you use this code or the theoretical framework in your research, please cite the paper:

Extrait de code

@article{bruneteau2025nonlinear,
  title={Non-Linear Computational Modeling of Emotional-Like States in Artificial Agents},
  author={Bruneteau, Côme},
  journal={TOAQ Research},
  year={2025},
  url={[https://research.toaq.fr/index/Non-Linear-Computational-Modeling-of-Emotional-Like-States-in-Artificial-Agents](https://research.toaq.fr/index/Non-Linear-Computational-Modeling-of-Emotional-Like-States-in-Artificial-Agents)}
}

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

Research conducted at TOAQ.