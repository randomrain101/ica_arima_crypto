#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA

#%%
from scipy import signal

np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

df = pd.DataFrame([s1, s2, s3]).T

df.plot(figsize=(15, 5))

#%%
for std in [0, 0.1, 0.2, 0.5]:
    S = np.c_[s1, s2, s3]
    S += std * np.random.normal(size=S.shape)  # Add noise

    #print(np.corrcoef(S).mean())

    S /= S.std(axis=0)  # Standardize data
    # Mix data
    A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
    X = np.dot(S, A.T)  # Generate observations
    S += std * np.random.normal(size=S.shape)  # Add noise

    ica = FastICA().fit(X)
    df_unmixed = pd.DataFrame(ica.transform(X))
    pd.DataFrame(S).plot(figsize=(15, 5))
    plt.show()
    df_unmixed.plot(figsize=(15, 5))
    plt.show()

# %%


# %%
