import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from boruta import BorutaPy


class SafeSNVTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.array(X)
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        std[std == 0] = 1e-98
        return (X - mean) / std
    
def spectral_binning(X, bin_size=10):
    # X shape (samples, 230)
    n_bins = X.shape[1] // bin_size
    X_binned = np.array([X[:, i*bin_size:(i+1)*bin_size].mean(axis=1) for i in range(n_bins)]).T
    return X_binned

def to_absorbance(X):
    return np.log10(1.0/ (X+1e-6))

class PLSFeatureExtractor(BaseEstimator, TransformerMixin):
    """Wyciąga składowe PLS jako cechy dla kolejnego estymatora."""
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=self.n_components)

    def fit(self, X, y):
        # PLS potrzebuje Y do znalezienia kierunków kowariancji
        self.pls.fit(X, y)
        return self

    def transform(self, X):
        return self.pls.transform(X)

class BandFilterSNR(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.keep_idx = None

    def fit(self, X, y=None):
        snr = np.abs(np.mean(X, axis=0)) / np.std(X, axis=0)
        self.keep_idx = np.where(snr > self.threshold)[0]
        return self

    def transform(self, X):
        return X[:, self.keep_idx]
    

class SpectralBinning(BaseEstimator, TransformerMixin):
    def __init__(self, factor = 5):
        self.factor = factor
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # oczekuje X (n_pixels, n_bands)
        p = X[0].shape[1]
        return [ x.reshape(-1, p // self.factor, self.factor ).mean(axis=2)
                for x in X ]
    

# =============================
    
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Memory
from scipy.signal import savgol_filter, welch
from scipy.stats import kurtosis, skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import OAS, LedoitWolf
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from tqdm import tqdm

class VariableSizeCovariance(BaseEstimator, TransformerMixin):
    def __init__(self, estimator='lwf'):
        self.estimator = estimator
        self.n_bands = None
        
    def fit(self, X, y=None):
        self.n_bands = X[0].shape[1]
        print(f'[VariableSizeCovariance] n_bands: {self.n_bands}')
        assert np.all([x.shape[1] == self.n_bands for x in X])
        return self

    def transform(self, X):
        """
        X: Lista (list) macierzy numpy. Każda ma kształt (n_channels, n_samples_variable)
        Zwraca: np.array (n_observations, n_channels, n_channels)
        """
        covs = []
        # Używamy OAS lub LedoitWolf, bo są odporne na n_features > n_samples (150 > 73)
        if self.estimator == 'lwf':
            cov_est = LedoitWolf(assume_centered=True)
        else:
            cov_est = OAS(assume_centered=True) # Często lepszy niż LWF

        
        n_vs_p_checker = np.mean([ x.shape[0] > self.n_bands**2 for x in X ])
        print(f'N > p in {n_vs_p_checker*100:0.1f}% samples')
        

        for x in tqdm(X, desc="Computing Covariances"):
            # Sklearn oczekuje (n_samples, n_features), czyli (n_pixels, 150)
            assert x.shape[1] == self.n_bands
            x_centered = x - x.mean(axis=0, keepdims=True)
            c = cov_est.fit(x_centered).covariance_
            covs.append(c)
            
        return np.stack(covs)
    
class CovarianceVectorizer(BaseEstimator, TransformerMixin):
    """
    Przyjmuje: Tensor (n_samples, n_bands, n_bands)
    Zwraca: Macierz (n_samples, n_features) gdzie n_features to unikalne elementy kowariancji
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        n_samples, n_bands, _ = X.shape
        # Indeksy górnego trójkąta (włącznie z przekątną)
        iu = np.triu_indices(n_bands)
        
        # Spłaszczanie każdej macierzy
        vectors = [cov_mat[iu] for cov_mat in X]
        return np.stack(vectors)
    

class ListSpectralBinner(BaseEstimator, TransformerMixin):
    """Binning spektralny operujący na liście macierzy (n_pixels, n_bands)."""
    def __init__(self, bin_factor=1):
        self.bin_factor = bin_factor
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.bin_factor <= 1:
            return X
        new_X = []
        for x in X:
            n_pix, n_bands = x.shape
            new_B = n_bands // self.bin_factor
            # Średnia po sąsiednich pasmach
            x_binned = x[:, :new_B*self.bin_factor].reshape(n_pix, new_B, self.bin_factor).mean(axis=2)
            new_X.append(x_binned)
        return new_X


class ListSNV(BaseEstimator, TransformerMixin):
    """SNV aplikowane do każdej macierzy w liście (per-spectrum normalization)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # x ma kształt (n_pixels, n_bands)
        # Odejmujemy średnią i dzielimy przez std dla każdego widma (wiersza)
        return [(x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True) for x in X]

class ListSavgol(BaseEstimator, TransformerMixin):
    """Filtr Savitzky-Golay aplikowany do każdej macierzy w liście."""
    def __init__(self, window_length=11, polyorder=2, deriv=0):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [savgol_filter(x, window_length=self.window_length, 
                              polyorder=self.polyorder, deriv=self.deriv, axis=1) for x in X]

class MeanSpectralAggregator(BaseEstimator, TransformerMixin):
    """List[Matrix] -> Matrix(n_samples, n_bands) przez uśrednianie pikseli."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        return np.vstack([np.mean(x, axis=0) for x in X])
    

class DummyTrainMeanStorer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean_y_ = None

    def fit(self, X, y=None):
        if y is not None:
            self.mean_y_ = np.mean(y)
        return self

    def transform(self, X):
        # Nie zmienia danych, po prostu je przepuszcza
        return X
    
class PLSFeatureExtractor(BaseEstimator, TransformerMixin):
    """Wyciąga składowe PLS jako cechy dla kolejnego estymatora."""
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=self.n_components)

    def fit(self, X, y):
        # PLS potrzebuje Y do znalezienia kierunków kowariancji
        self.pls.fit(X, y)
        return self

    def transform(self, X):
        return self.pls.transform(X)


class ListPSDTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_per_seg=None):
        self.n_per_seg = n_per_seg

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        psd_features = []
        for x in X:
            # 1. Agregacja do średniego widma: (n_pixels, n_bands) -> (n_bands,)
            mean_spectrum = np.mean(x, axis=0)
            
            # 2. Obliczenie PSD (Power Spectral Density)
            # n_per_seg musi być mniejsze niż liczba pasm (n_bands)
            f, pxx = welch(mean_spectrum, nperseg=self.n_per_seg)
            
            psd_features.append(pxx)
            
        return np.vstack(psd_features)


class ListStatsTransformer(BaseEstimator, TransformerMixin):
    """Oblicza statystyki opisowe dla każdego widma w liście."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        stats_list = []
        for x in X:
            # Agregacja do średniego widma (n_bands,)
            mean_spec = np.mean(x, axis=0)
            
            # Statystyki po osi pasm
            sk = skew(mean_spec)
            ku = kurtosis(mean_spec)
            
            stats_list.append([sk, ku])
            
        return np.array(stats_list)


def mse_ratio_to_dummy_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    mse_model = mean_squared_error(y, y_pred)
    
    train_mean = estimator.regressor_.named_steps['storer'].mean_y_
    train_mean_original = estimator.transformer_.inverse_transform([[train_mean]])[0][0]
    mse_dummy = np.mean((y - train_mean_original)**2)
    
    if mse_dummy == 0: return 0.0
    
    return -(mse_model / mse_dummy)


class BorutaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, estimator=None, n_estimators=100, random_state=42, max_iter=50):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_iter = max_iter
        self.boruta_ = None

    def fit(self, X, y=None):
        # Boruta wymaga np.array, nie DataFrame
        X_arr = np.array(X)
        y_arr = np.array(y).ravel()
        
        # Domyślny estimator dla Boruty (musi być RF)
        rf = RandomForestRegressor(
            n_jobs=-1, 
            max_depth=5, 
            n_estimators=self.n_estimators, 
            random_state=self.random_state
        )
        
        self.boruta_ = BorutaPy(rf, n_estimators='auto', random_state=self.random_state, max_iter=self.max_iter, verbose=0)
        self.boruta_.fit(X_arr, y_arr)
        return self

    def transform(self, X):
        X_arr = np.array(X)
        return self.boruta_.transform(X_arr)