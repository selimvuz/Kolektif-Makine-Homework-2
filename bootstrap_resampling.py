import numpy as np

# Original dataset
original_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Number of bootstrap samples
num_bootstrap_samples = 5

# Generate bootstrap samples
bootstrap_samples = [np.random.choice(original_data, size=len(
    original_data), replace=True) for _ in range(num_bootstrap_samples)]

# Print the bootstrap samples
for i, sample in enumerate(bootstrap_samples):
    print(f"Bootstrap Sample {i + 1}: {sample}")
