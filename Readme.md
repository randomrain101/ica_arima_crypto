## Original paper
 Oja, Erkki, Kimmo Kiviluoto, and Simona Malaroiu. "Independent component analysis for financial time series." Proceedings of the IEEE 2000 Adaptive Systems for Signal Processing, Communications, and Control Symposium (Cat. No. 00EX373). IEEE, 2000.
 [doi.org/10.1109/ASSPCC.2000.882456](https://doi.org/10.1109/ASSPCC.2000.882456)

  ## Empirical Results
  ### R (Pearson Correlation)
 - **Pearson Correlation** of ica_arima **better** than just arima


```python
pearsonr(df_ret, df_ica_pred, alternative="greater"), \
    pearsonr(df_ret, df_arma_pred, alternative="greater")
```




    (PearsonRResult(statistic=0.013404161902935948, pvalue=0.0630091125365575),
     PearsonRResult(statistic=0.005678315103859606, pvalue=0.2584545828881826))



 ### Directional accuracy
 - Diebold Mariano test shows **Directional Accuarcy** (correct sign of returns predicted) for ica_arima significantly **better** than arima


```python
print("Diebold Mariano p-value:",
DM(
    np.sign(df_ret).values.reshape(-1, 2),
    np.sign(df_arma_pred).values.reshape(-1, 2),
    np.sign(df_ica_pred).values.reshape(-1, 2),
    norm=1,
    version="multivariate"))
```

    Diebold Mariano p-value: 0.11904912161485659


 ### Mean Absolute Error
 - Diebold Mariano test shows **Mean Absolute Error** of ica_arima predicitons significantly **worse** than arima


```python
print("Diebold Mariano p-value:",
DM(
    df_ret.values.reshape(-1, 2),
    df_arma_pred.values.reshape(-1, 2),
    df_ica_pred.values.reshape(-1, 2),
    norm=1,
    version="multivariate"))
```

    Diebold Mariano p-value: 1.0


 ### Mean Squared Error
 - Diebold Mariano test shows **Mean Squared Error** of ica_arima predicitons significantly **worse** than arima


```python
print("Diebold Mariano p-value:",
DM(
    df_ret.values.reshape(-1, 2),
    df_arma_pred.values.reshape(-1, 2),
    df_ica_pred.values.reshape(-1, 2),
    norm=2,
    version="multivariate"))
```

    Diebold Mariano p-value: 1.0


 ## Comaparison of ARIMA orders
 - for just ARIMA the AutoARIMA algorithm determined Brownian Noise with order (0, 1, 0) to be the best model of the process
 way more often than for ica + ARIMA, indicating application of ICA improved the Signal to Noise Ratio
    
![png](results_files/results_9_1.png)
    

# License

All source code is made available under a GNU AFFERO GENERAL PUBLIC LICENSE Version 3 license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.