import numpy as np

def generate_pink_noise(num_steps, sigma=0.1):
    """
    Generates 1/f noise (Pink Noise) using the Voss-McCartney algorithm.
    Represents the Endogenous Stochastic Modulation term zeta(t).
    
    Args:
        num_steps (int): Duration of the simulation.
        sigma (float): Standard deviation (amplitude) of the noise.
    """
    num_rows = 16
    array = np.empty((num_rows, num_steps))
    array.fill(np.nan)
    array[0, :] = np.random.randn(num_steps)
    array[:, 0] = np.random.randn(num_rows)
    
    # Voss algorithm
    for i in range(1, num_steps):
        index = 0
        n = i
        while n % 2 == 0:
            n //= 2
            index += 1
        if index < num_rows:
            array[index, i] = np.random.randn()
    
    # Fill nans with previous values
    mask = np.isnan(array)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    filled = array[np.arange(mask.shape[0])[:,None], idx]
    
    # Sum rows to get pink noise
    pink_noise = np.sum(filled, axis=0)
    
    # Normalize to match desired sigma
    pink_noise = pink_noise - np.mean(pink_noise)
    pink_noise = pink_noise * (sigma / np.std(pink_noise))
    
    return pink_noise