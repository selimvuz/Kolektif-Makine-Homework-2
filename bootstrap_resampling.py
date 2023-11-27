import numpy as np

# Bu kod, bootstrap resampling yönteminin nasıl uygulandığını
# basit bir şekilde göstermektedir.

# Orijinal veri kümesi
original_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Bootstrap örneklemlerinin sayısı
num_bootstrap_samples = 5

# Bootstrap örneklemlerini oluştur
bootstrap_samples = [np.random.choice(original_data, size=len(
    original_data), replace=True) for _ in range(num_bootstrap_samples)]

# Bootstrap örneklemlerini yazdır
for i, sample in enumerate(bootstrap_samples):
    print(f"Bootstrap Sample {i + 1}: {sample}")
