import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from tqdm import tqdm

def load_model_and_data():
    df = pd.read_excel("fitur_dengan_temporal_normalisasi.xlsx")
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    y = df['total_pembayaran'].values
    X_all = df.drop(columns=['Tanggal', 'total_pembayaran'])

    selector = SelectKBest(score_func=f_regression, k=20)
    X_selected = selector.fit_transform(X_all, y)
    selected_columns = X_all.columns[selector.get_support()]

    model = ManualXGBoost()
    model.fit(X_selected, y)
    return model, df, selected_columns

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class SimpleTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2, reg_lambda=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.root = None

    def fit(self, X, grad, hess):
        self.root = self._build_tree(X, grad, hess, 0)

    def _build_tree(self, X, grad, hess, depth):
        if depth >= self.max_depth or len(X) < self.min_samples_split:
            value = -np.sum(grad) / (np.sum(hess) + self.reg_lambda)
            return Node(value=value)
        best_gain = -np.inf
        for i in range(X.shape[1]):
            for threshold in np.unique(X[:, i]):
                left = X[:, i] <= threshold
                right = ~left
                if not left.any() or not right.any(): continue
                gain = self._calc_gain(grad[left], hess[left], grad[right], hess[right])
                if gain > best_gain:
                    best_gain, split = gain, (i, threshold, left, right)
        if best_gain == -np.inf:
            return Node(value=-np.sum(grad) / (np.sum(hess) + self.reg_lambda))
        i, t, left, right = split
        return Node(i, t,
            self._build_tree(X[left], grad[left], hess[left], depth+1),
            self._build_tree(X[right], grad[right], hess[right], depth+1)
        )

    def _calc_gain(self, g_left, h_left, g_right, h_right):
        g_total, h_total = np.sum(g_left)+np.sum(g_right), np.sum(h_left)+np.sum(h_right)
        return 0.5 * ((np.sum(g_left)**2)/(np.sum(h_left)+self.reg_lambda) + 
                      (np.sum(g_right)**2)/(np.sum(h_right)+self.reg_lambda) - 
                      (g_total**2)/(h_total+self.reg_lambda))

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.value is not None: return node.value
        return self._traverse(x, node.left) if x[node.feature_index] <= node.threshold else self._traverse(x, node.right)

class ManualXGBoost:
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=5, reg_lambda=1, min_samples_split=2):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.min_samples_split = min_samples_split
        self.trees = []
        self.init_value = None

    def fit(self, X, y):
        y = np.array(y)
        self.init_value = np.mean(y)
        y_pred = np.full_like(y, self.init_value, dtype=float)
        for _ in tqdm(range(self.n_estimators), desc="Training ManualXGBoost"):
            grad, hess = y_pred - y, np.ones_like(y)
            tree = SimpleTreeRegressor(self.max_depth, self.min_samples_split, self.reg_lambda)
            tree.fit(X, grad, hess)
            y_pred += self.lr * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.init_value)
        for tree in self.trees:
            y_pred += self.lr * tree.predict(X)
        return y_pred

def predict_aggregated(model, df, selected_columns, start_date, n_days):
    df_pred = df[df['Tanggal'].between(start_date, start_date + pd.Timedelta(days=n_days - 1))].copy()
    if df_pred.empty:
        return 0, 0
    X_input = df_pred[selected_columns].values
    total = model.predict(X_input).sum()
    jumlah = df_pred['jumlah_pembelian'].sum() if 'jumlah_pembelian' in df_pred.columns else len(df_pred)
    return total, jumlah
