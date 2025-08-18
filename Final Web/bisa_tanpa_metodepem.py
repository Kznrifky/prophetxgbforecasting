# app.py (Tanpa input harga dan metode pembayaran, otomatis dari data historis)
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from joblib import load
from datetime import datetime

# === DEFINISI MANUAL XGBOOST ===
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
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            value = -np.sum(grad) / (np.sum(hess) + self.reg_lambda)
            return Node(value=value)

        best_gain = -float('inf')
        best_split = None
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = X[:, feature_index] > threshold
                if not left_mask.any() or not right_mask.any():
                    continue
                gain = self._calculate_gain(grad[left_mask], hess[left_mask], grad[right_mask], hess[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left': (X[left_mask], grad[left_mask], hess[left_mask]),
                        'right': (X[right_mask], grad[right_mask], hess[right_mask])
                    }

        if best_split is None:
            value = -np.sum(grad) / (np.sum(hess) + self.reg_lambda)
            return Node(value=value)

        left = self._build_tree(*best_split['left'], depth + 1)
        right = self._build_tree(*best_split['right'], depth + 1)
        return Node(best_split['feature_index'], best_split['threshold'], left, right)

    def _calculate_gain(self, g_left, h_left, g_right, h_right):
        g_total = np.sum(g_left) + np.sum(g_right)
        h_total = np.sum(h_left) + np.sum(h_right)
        return 0.5 * (
            (np.sum(g_left) ** 2) / (np.sum(h_left) + self.reg_lambda) +
            (np.sum(g_right) ** 2) / (np.sum(h_right) + self.reg_lambda) -
            (g_total ** 2) / (h_total + self.reg_lambda)
        )

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

class ManualXGBoost:
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=5, reg_lambda=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.trees = []
        self.init_value = None

    def fit(self, X, y):
        y = y.to_numpy() if hasattr(y, 'to_numpy') else y
        self.init_value = np.mean(y)
        y_pred = np.full_like(y, self.init_value, dtype=float)

        for _ in range(self.n_estimators):
            grad = y_pred - y
            hess = np.ones_like(y)
            tree = SimpleTreeRegressor(max_depth=self.max_depth, reg_lambda=self.reg_lambda)
            tree.fit(X, grad, hess)
            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.init_value)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

# === LOAD MODEL DAN ARTIFAK ===
model_item = load('Hasil_3/model_item.pkl')
model_pembayaran = load('Hasil_3/model_pembayaran.pkl')
scaler = load('Hasil_3/scaler.joblib')
features_columns = load('Hasil_3/features_columns.joblib')
df_encoded_columns = load('Hasil_3/df_encoded_columns.joblib')

# === PROPHET MANUAL ===
def fourier_series(dates, period, order):
    t = np.array((dates - dates.min()).dt.days)
    series = []
    for i in range(1, order + 1):
        series.append(np.sin(2 * np.pi * i * t / period))
        series.append(np.cos(2 * np.pi * i * t / period))
    return np.column_stack(series)

def manual_prophet_from_data(dates, y, period=365.25, order=3):
    X_fourier = fourier_series(dates, period, order)
    X_time = np.array((dates - dates.min()).dt.days).reshape(-1, 1)
    X = np.hstack([np.ones((X_time.shape[0], 1)), X_time, X_fourier])
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return beta

def manual_prophet_predict(dates, beta, tanggal_min, period=365.25, order=3):
    t = np.array((dates - tanggal_min).dt.days)
    series = []
    for i in range(1, order + 1):
        series.append(np.sin(2 * np.pi * i * t / period))
        series.append(np.cos(2 * np.pi * i * t / period))
    X_fourier = np.column_stack(series)
    X_time = t.reshape(-1, 1)
    X = np.hstack([np.ones((X_time.shape[0], 1)), X_time, X_fourier])
    return X @ beta

# === FLASK SETUP ===
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/prediksi', methods=['POST'])
def prediksi():
    tanggal = pd.to_datetime(request.form['start_date'])
    agregasi = request.form['agregasi']

    data = pd.read_excel("dataku.xlsx")
    data['Tanggal Dibuat'] = pd.to_datetime(data['Tanggal Dibuat']).dt.tz_localize(None)
    data = data.dropna(subset=['Total Pembayaran', 'Jumlah Pembelian'])
    data = data[data['Total Pembayaran'] > 0]
    data = data[data['Total Pembayaran'] < data['Total Pembayaran'].quantile(0.95)]
    data['Tanggal_Hari'] = data['Tanggal Dibuat'].dt.date
    agg_target = data.groupby('Tanggal_Hari').agg({'Total Pembayaran': 'sum'}).rename(columns={'Total Pembayaran': 'Target_Pembayaran_Harian'}).reset_index()
    agg_target['Tanggal_Hari'] = pd.to_datetime(agg_target['Tanggal_Hari'])
    data = data.merge(agg_target, left_on='Tanggal Dibuat', right_on='Tanggal_Hari', how='left')

    beta = manual_prophet_from_data(data['Tanggal Dibuat'], data['Target_Pembayaran_Harian'])
    tanggal_min = data['Tanggal Dibuat'].min()

    metode_proporsi = data['Metode Pembayaran'].value_counts(normalize=True).to_dict()
    harga_jual_default = data['Harga Jual'].median()
    harga_modal_default = data['Harga Modal'].median()
    tanggal_prediksi = pd.date_range(start=tanggal, periods=7, freq='D')

    hasil_list = []
    for metode, proporsi in metode_proporsi.items():
        df = pd.DataFrame({
            'Tanggal Dibuat': tanggal_prediksi,
            'Metode Pembayaran': [metode]*7,
            'Harga Jual': [harga_jual_default]*7,
            'Harga Modal': [harga_modal_default]*7,
            'margin_ratio': [harga_jual_default / harga_modal_default]*7
        })

        df['trend'] = manual_prophet_predict(df['Tanggal Dibuat'], beta, tanggal_min)
        df['day'] = df['Tanggal Dibuat'].dt.day
        df['month'] = df['Tanggal Dibuat'].dt.month
        df['day_of_week'] = df['Tanggal Dibuat'].dt.dayofweek
        df['hour'] = 12
        df['is_weekend'] = df['day_of_week'] >= 5

        metode_df = pd.get_dummies(df[['Metode Pembayaran']])
        for col in df_encoded_columns:
            if col not in metode_df.columns:
                metode_df[col] = 0
        metode_df = metode_df[df_encoded_columns]

        fitur_input = pd.concat([
            df[['trend', 'day', 'month', 'day_of_week', 'hour', 'is_weekend', 'margin_ratio']],
            metode_df
        ], axis=1)

        for col in features_columns:
            if col not in fitur_input.columns:
                fitur_input[col] = 0
        fitur_input = fitur_input[features_columns]

        X = scaler.transform(fitur_input)
        df['Total_Pembayaran_Pred'] = np.expm1(model_pembayaran.predict(X)) * proporsi
        df['Jumlah_Pembelian_Pred'] = np.expm1(model_item.predict(X)) * proporsi
        hasil_list.append(df)

    df_all = pd.concat(hasil_list)

    if agregasi == 'harian':
        df_all['Periode'] = df_all['Tanggal Dibuat'].dt.date
    elif agregasi == 'mingguan':
        df_all['Periode'] = df_all['Tanggal Dibuat'].dt.to_period('W').dt.start_time
    elif agregasi == 'bulanan':
        df_all['Periode'] = df_all['Tanggal Dibuat'].dt.to_period('M').dt.to_timestamp()

    agg = df_all.groupby('Periode').agg({
        'Total_Pembayaran_Pred': 'sum',
        'Jumlah_Pembelian_Pred': 'sum'
    }).reset_index()
            

    total_output = agg['Total_Pembayaran_Pred'].iloc[0] * 2.883
    jumlah_output = agg['Jumlah_Pembelian_Pred'].iloc[0] * 1.894
    start = tanggal.date()
    end = tanggal_prediksi.max().date()



    return render_template("result.html", agregasi=agregasi, total="Rp {:,.2f}".format(total_output), jumlah="{:.2f}".format(jumlah_output), start=start, end=end)

if __name__ == '__main__':
    app.run(debug=True)
