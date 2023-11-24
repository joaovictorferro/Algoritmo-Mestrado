from PyEMD import CEEMDAN
import numpy as np
import matplotlib.pyplot as plt

# Gere uma série temporal de exemplo
signal = np.sin(2 * np.pi * 0.1 * np.arange(1000)) + 0.5 * np.random.randn(1000)

# Aplique a CEEMDAN
ceemdan = CEEMDAN()
result = ceemdan(signal)

# A variável result contém dois elementos: result[0] são as IMFs e result[1] é o resíduo (ruído)

# Exemplo de visualização das IMFs e do resíduo
plt.figure(figsize=(12, 8))

plt.subplot(len(result[0]) + 1, 1, 1)
plt.plot(signal, label='Original Signal')
plt.legend()

for i in range(len(result[0])):
    plt.subplot(len(result[0]) + 1, 1, i + 2)
    plt.plot(result[0][i], label=f'IMF {i + 1}')
    plt.legend()

plt.subplot(len(result[0]) + 1, 1, len(result[0]) + 1)
plt.plot(result[1], label='Residue (Noise)')
plt.legend()

plt.tight_layout()
plt.show()
