import os
import numpy as np
import joblib
import json

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn import svm

from imblearn.over_sampling import SMOTE
from abc import ABC, abstractmethod
from scipy.stats import chi2
from xgboost import XGBClassifier

def convert(o):
    if isinstance(o, (np.integer, )):
        return int(o)
    if isinstance(o, (np.floating, )):
        return float(o)
    if isinstance(o, (np.ndarray, )):
        return o.tolist()
    raise TypeError

def make_folds_idx(n, N, seed=42):
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    return np.array_split(rng.permutation(idx), N)

class BaseTrainer(ABC):
    """
    Abstract base class for training pipelines with cross-validation and logging.

    Automatically detects numeric and categorical columns unless provided.
    Works for both supervised and unsupervised setups.

    Subclass must implement:
    - build_estimator
    - split_data_for_cv
    - make_features
    - fit_test
    - final_refit
    """

    def __init__(self, X_df, results_path, cv_number=5, y=None,
                 numeric_cols=None, categorical_cols=None):
        self.X = X_df.reset_index(drop=True)
        self.y = None if y is None else (y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y))
        self.results_path = results_path
        self.cv_number = cv_number
        self.folds = make_folds_idx(len(self.X), cv_number, seed=42)

        if numeric_cols is None:
            numeric_cols = [c for c in self.X.columns if np.issubdtype(self.X[c].dtype, np.number)]
        if categorical_cols is None:
            categorical_cols = [c for c in self.X.columns if c not in numeric_cols]
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

    @abstractmethod
    def build_estimator(self, params):
        ...

    @abstractmethod
    def split_data_for_cv(self, i):
        """Returns: X_tr, X_va, y_tr, y_va (y_* can be None in unsupervised)."""
        ...

    @abstractmethod
    def make_features(self, X_tr, X_va, params):
        """
        Returns: Z_tr, Z_va, preproc  (preproc equals pca).
        """
        ...

    @abstractmethod
    def fit_test(self, clf, Z_tr, y_tr, Z_va, y_va, params):
        """Returns scalar: the bigger the better."""
        ...

    @abstractmethod
    def final_refit(self, best_params):
        """Final fit for entire dataset; returns (best_model, best_preproc)."""
        ...

    def _ensure_file(self):
        if not os.path.exists(self.results_path):
            open(self.results_path, 'w').close()

    def custom_grid_search(self, param_grid):
        self._ensure_file()
        best_score = -np.inf
        best_params = None

        with open(self.results_path, 'a') as f:
            for params in ParameterGrid(param_grid):
                fold_scores = []
                for i in range(self.cv_number):
                    X_tr, X_va, y_tr, y_va = self.split_data_for_cv(i)
                    Z_tr, Z_va, preproc = self.make_features(X_tr, X_va, params)

                    clf = self.build_estimator(params)
                    sc = self.fit_test(clf, Z_tr, y_tr, Z_va, y_va, params)
                    fold_scores.append(sc)

                mean_score = float(np.mean(fold_scores))
                out = dict(params); out["score"] = mean_score
                json.dump(out, f, indent=2, default=convert)
                print(out, "mean score:", mean_score)

                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params

        best_model, best_preproc = self.final_refit(best_params)
        return best_score, best_params, best_model, best_preproc


class UnsupervisedTrainer(BaseTrainer):
    """
    Base class for unsupervised training.

    Splits data into folds with only 'normal' samples in training.
    Validation labels (y_va) are all zeros.
    """

    def split_data_for_cv(self, i):
        val_idx = self.folds[i]
        mask = np.ones(len(self.X), dtype=bool); mask[val_idx] = False
        X_tr = self.X.iloc[mask].copy()
        X_va = self.X.iloc[val_idx].copy()
        y_tr = None
        y_va = np.zeros(len(X_va), dtype=int)
        return X_tr, X_va, y_tr, y_va


class SupervisedTrainer(BaseTrainer):
    """
    Base class for supervised training with stratified cross-validation.
    """
    def __init__(self, X_df, y, results_path, cv_number=5, numeric_cols=None, categorical_cols=None):
        super().__init__(X_df, results_path, cv_number=cv_number, y=y,
                         numeric_cols=numeric_cols, categorical_cols=categorical_cols)
        self._skf = StratifiedKFold(n_splits=cv_number, shuffle=True, random_state=42)
        self.folds = list(self._skf.split(self.X, self.y))

    def split_data_for_cv(self, i):
        tr_idx, va_idx = self.folds[i]
        X_tr = self.X.iloc[tr_idx].copy()
        X_va = self.X.iloc[va_idx].copy()
        y_tr = self.y[tr_idx]
        y_va = self.y[va_idx]
        return X_tr, X_va, y_tr, y_va

    
class Numerical_trainer(ABC):
    """
    Mixin providing numeric-only preprocessing with optional PCA.
    """    
    def make_pca(self, n_components, df, pca=None, *, whiten=False):
        X_num = df[self.numeric_cols].to_numpy()
        if n_components is None:
            return X_num, None
        if pca is None:
            pca = PCA(n_components=n_components, whiten=whiten, random_state=42)
            Z = pca.fit_transform(X_num)
        else:
            Z = pca.transform(X_num)
        return Z, pca
    
    def make_features(self, X_tr, X_va, params):
        Z_tr, pca = self.make_pca(params.get("n_components", None), X_tr,
                                    pca=None, whiten=False)
        Z_va, _   = self.make_pca(params.get("n_components", None), X_va,
                                    pca=pca,  whiten=False)
        return Z_tr, Z_va, pca

class Universal_trainer(ABC):
    """
    Mixin providing preprocessing for both numeric and categorical data.

    Features:
    - Numeric columns with optional PCA
    - Categorical columns either dropped or appended as one-hot
    """    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self._scaler = None

    def make_pca(self, n_components, df, pca=None, *, whiten=False, cat_mode="drop"):
        X_num = df[self.numeric_cols].to_numpy()
        if n_components is None:
            Z_num, pca_out = X_num, None
        else:
            if pca is None:
                pca_out = PCA(n_components=n_components, whiten=whiten, random_state=42)
                Z_num = pca_out.fit_transform(X_num)
            else:
                pca_out = pca
                Z_num = pca_out.transform(X_num)

        if len(self.categorical_cols) == 0 or cat_mode == "drop":
            Z = Z_num
        elif cat_mode == "onehot":
            Z_cat = df[self.categorical_cols]
            Z = np.concatenate([Z_num, Z_cat], axis=1)
        else:
            raise ValueError("cat_mode bust be 'drop' albo 'onehot'.")

        return Z, pca_out
    
    def make_features(self, X_tr, X_va, params):
        Z_tr, preproc = self.make_pca(params.get("n_components", None), X_tr,
                                        pca=None, whiten=False, cat_mode="onehot")
        Z_va, _       = self.make_pca(params.get("n_components", None), X_va,
                                        pca=preproc, whiten=False, cat_mode="onehot")
        return Z_tr, Z_va, preproc
    
class KMeansTrainer(Numerical_trainer, UnsupervisedTrainer):
    """
    Specific trainer implementations for KMeans / IsolationForest / XGBoost / SVM.

    Each implements:
    - build_estimator
    - fit_test
    - final_refit
    with logic tailored to the algorithm.
    """
    def build_estimator(self, params):
        return KMeans(n_clusters=params["n_clusters"], n_init=50, random_state=42)

    def fit_test(self, clf, Z_tr, y_tr, Z_va, y_va, params):
        clf.fit(Z_tr)      
        alpha = params.get("alpha", 0.1)
        d = Z_tr.shape[1]
        tau = chi2.ppf(1 - alpha, df=d)
        D_va = clf.transform(Z_va)
        r2_va = np.min(D_va**2, axis=1)
        fpr = (r2_va > tau).mean()
        penalty_over = max(0.0, fpr - alpha)
        return -(abs(fpr - alpha) + 0.5 * penalty_over)

    def final_refit(self, best_params):
        Z_all, pca = self.make_pca(best_params.get("n_components", None), self.X,
                                   pca=None, whiten=best_params.get("whiten", True))
        model = self.build_estimator(best_params)
        model.fit(Z_all)
        return model, pca
    
    
class IFTrainer(Universal_trainer, UnsupervisedTrainer):
    """
    Specific trainer implementations for KMeans / IsolationForest / XGBoost / SVM.

    Each implements:
    - build_estimator
    - fit_test
    - final_refit
    with logic tailored to the algorithm.
    """    
    def build_estimator(self, params):
        keep = {k: v for k, v in params.items()
                if k not in ("n_components", "alpha", "whiten", "cat_mode")}
        return IsolationForest(random_state=42, contamination='auto', **keep)

    def fit_test(self, clf, Z_tr, y_tr, Z_va, y_va, params):
        clf.fit(Z_tr)
        alpha = params.get("alpha", 0.1)
        s_tr = clf.score_samples(Z_tr)
        t = np.quantile(s_tr, alpha)
        s_va = clf.score_samples(Z_va)
        fpr = (s_va < t).mean()
        penalty_over = max(0.0, fpr - alpha)
        return -(abs(fpr - alpha) + 0.5 * penalty_over)

    def final_refit(self, best_params):
        Z_all, preproc = self.make_pca(best_params.get("n_components", None), self.X,
                                       pca=None, whiten=False, cat_mode="onehot")
        model = self.build_estimator(best_params)
        model.fit(Z_all)
        return model, preproc        
    
class XGBoostTrainer(Universal_trainer, SupervisedTrainer):
    """
    Specific trainer implementations for KMeans / IsolationForest / XGBoost / SVM.

    Each implements:
    - build_estimator
    - fit_test
    - final_refit
    with logic tailored to the algorithm.
    """    
    def build_estimator(self, params):
        keep = {k: v for k, v in params.items()
                if k not in ("n_components", "alpha", "whiten", "cat_mode", "balancing")}
        clf = XGBClassifier(random_state=42, eval_metric='logloss', **keep)
        if params.get("balancing", None) == "proportional":
            neg = (self.y == 0).sum()
            pos = (self.y == 1).sum()
            if pos > 0:
                clf.set_params(scale_pos_weight=neg / pos)
        return clf
    
    def fit_test(self, clf, Z_tr, y_tr, Z_va, y_va, params):
        clf2 = clone(clf)
        if params.get("balancing", None) == "SMOTE":
            sm = SMOTE(random_state=42)
            Z_tr_sm, y_tr_sm = sm.fit_resample(Z_tr, y_tr)
            clf2.fit(Z_tr_sm, y_tr_sm)
            s = clf2.predict_proba(Z_va)[:, 1]
        else:
            clf2.fit(Z_tr, y_tr)
            s = clf2.predict_proba(Z_va)[:, 1]
        return roc_auc_score(y_va.astype(float), s)

    def final_refit(self, best_params):
        Z_all, preproc = self.make_pca(best_params.get("n_components", None), self.X,
                                       pca=None, whiten=False, cat_mode="onehot")
        clf = self.build_estimator(best_params)

        if best_params.get("balancing", None) == "SMOTE":
            sm = SMOTE(random_state=42)
            Z_all, y_all = sm.fit_resample(Z_all, self.y)
            Z_all, y_all = shuffle(Z_all, y_all, random_state=42)
            clf.fit(Z_all, y_all)
        else:
            clf.fit(Z_all, self.y)
        return clf, preproc
    
class SVMTrainer(Numerical_trainer, SupervisedTrainer):
    """
    Specific trainer implementations for KMeans / IsolationForest / XGBoost / SVM.

    Each implements:
    - build_estimator
    - fit_test
    - final_refit
    with logic tailored to the algorithm.
    """    
    def build_estimator(self, params):
        keep = {k: v for k, v in params.items()
                if k not in ("n_components", "alpha", "whiten", "cat_mode", "balancing")}
        # class_weight for proportional
        if params.get("balancing", None) == "proportional":
            keep["class_weight"] = "balanced"
        clf = svm.SVC(probability=False, random_state=42, **keep)
        return clf
    
    def fit_test(self, clf, Z_tr, y_tr, Z_va, y_va, params):
        clf2 = clone(clf)
        if params.get("balancing", None) == "SMOTE":
            sm = SMOTE(random_state=42)
            Z_tr_sm, y_tr_sm = sm.fit_resample(Z_tr, y_tr)
            clf2.fit(Z_tr_sm, y_tr_sm)
            s = clf2.decision_function(Z_va).ravel()
        else:
            clf2.fit(Z_tr, y_tr)
            s = clf2.decision_function(Z_va).ravel()
        return roc_auc_score(y_va.astype(float), s)

    def final_refit(self, best_params):
        Z_all, pca = self.make_pca(best_params.get("n_components", None), self.X,
                                   pca=None, whiten=False)
        clf = self.build_estimator(best_params)

        if best_params.get("balancing", None) == "SMOTE":
            sm = SMOTE(random_state=42)
            Z_all, y_all = sm.fit_resample(Z_all, self.y)
            Z_all, y_all = shuffle(Z_all, y_all, random_state=42)            
            clf.fit(Z_all, y_all)
        else:
            clf.fit(Z_all, self.y)
        return clf, pca
    
def save_results(algorithm_name, best_params, 
                 best_model, best_preproc, 
                 results_path, model_path, 
                 preproc_path):
    
    print(f"{algorithm_name} best:", best_params)
    joblib.dump(best_model, model_path)
    joblib.dump(best_preproc, preproc_path)
    with open(results_path, 'w') as f:
        json.dump(best_params, f, indent=4, default=convert) 