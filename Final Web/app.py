from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

# === KELAS DARI KODE TRAINING ===
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
        for _ in range(self.n_estimators):
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

app = Flask(__name__)

model = joblib.load("model_manualxgb.pkl")
selected_columns = joblib.load("selected_columns.pkl")
df = pd.read_excel("fitur_dengan_temporal_normalisasi.xlsx")
df['Tanggal'] = pd.to_datetime(df['Tanggal'])

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/prediksi', methods=['POST'])
def prediksi():
    start_date = pd.to_datetime(request.form['start_date'])
    agregasi = request.form['agregasi']
    n_days = {'harian': 1, 'mingguan': 7, 'bulanan': 30}[agregasi]
    results = []

    # Copy data awal
    df_pred = df.copy()

    for i in range(n_days):
        tanggal_prediksi = start_date + pd.Timedelta(days=i)

        # Set seed acak berdasarkan tanggal agar prediksi selalu sama untuk tanggal yang sama
        np.random.seed(int(tanggal_prediksi.strftime("%Y%m%d")))

        if tanggal_prediksi in df_pred['Tanggal'].values:
            row_pred = df_pred[df_pred['Tanggal'] == tanggal_prediksi].copy()
        else:
            row_pred = df_pred.iloc[[-1]].copy()
            row_pred['Tanggal'] = tanggal_prediksi
            row_pred['week_of_year'] = tanggal_prediksi.isocalendar().week
            row_pred['day_of_month'] = tanggal_prediksi.day
            row_pred['days_from_month_start'] = tanggal_prediksi.day
            row_pred['days_to_month_end'] = 31 - tanggal_prediksi.day

            for col in ['total_transaksi', 'total_pembelian', 'harga_modal_total', 'harga_jual_total']:
                row_pred[col] = df_pred[col].iloc[-7:].mean() * np.random.uniform(0.75, 4.05)

            recent = df_pred['total_pembayaran']
            row_pred['lag_pembayaran_14'] = recent.iloc[-14] if len(recent) >= 14 else recent.mean()
            row_pred['rolling_7'] = recent.iloc[-7:].mean()
            row_pred['rolling_14'] = recent.iloc[-14:].mean()
            row_pred['rolling_30'] = recent.iloc[-30:].mean()

        X_pred = row_pred[selected_columns].values
        # y_pred = model.predict(X_pred)[0]
        # results.append((tanggal_prediksi.strftime('%Y-%m-%d'), y_pred))
        y_pred = model.predict(X_pred)[0]

        # DEBUG: tampilkan prediksi harian di terminal
        print(f"[DEBUG] Prediksi untuk {tanggal_prediksi.strftime('%Y-%m-%d')} = Rp {y_pred:,.0f}")

        results.append((tanggal_prediksi.strftime('%Y-%m-%d'), y_pred))


    # Gabungkan hasil jika bukan harian
    if agregasi != 'harian':
        total_prediksi = sum(pred for _, pred in results)
        results = [(f"{start_date.strftime('%Y-%m-%d')} s.d {tanggal_prediksi.strftime('%Y-%m-%d')}", total_prediksi)]

    return render_template("result.html", results=results)

if __name__ == '__main__':
    app.run(debug=True)
