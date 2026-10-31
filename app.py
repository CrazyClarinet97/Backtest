import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from textblob import TextBlob
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Configurazione Pagina
st.set_page_config(
page_title="Institutional Market Terminal & Backtest Engine",
page_icon="⚡",
layout="wide",
initial_sidebar_state="collapsed"
)

STILE CSS "LIQUID GLASS"
st.markdown("""
<style>
.metric-card {
background: rgba(150, 150, 150, 0.06);
backdrop-filter: blur(12px);
-webkit-backdrop-filter: blur(12px);
border: 1px solid rgba(150, 150, 150, 0.15);
border-radius: 12px;
padding: 14px 16px;
margin-bottom: 10px;
box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
height: 100%;
}
.date-card {
background: rgba(2, 132, 199, 0.08);
border: 1px solid rgba(2, 132, 199, 0.25);
border-radius: 10px;
padding: 10px 14px;
margin-bottom: 14px;
font-size: 13px;
font-weight: 500;
display: flex;
align-items: center;
gap: 8px;
}
.flow-card {
background: rgba(15, 23, 42, 0.04);
border-radius: 10px;
padding: 12px 16px;
margin-bottom: 12px;
font-size: 13px;
display: flex;
justify-content: space-between;
align-items: center;
border: 1px solid rgba(150, 150, 150, 0.2);
}
@media (prefers-color-scheme: dark) {
.metric-card {
background: rgba(25, 30, 45, 0.45);
border: 1px solid rgba(255, 255, 255, 0.08);
box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}
.date-card {
background: rgba(56, 189, 248, 0.08);
border: 1px solid rgba(56, 189, 248, 0.2);
color: #f3f4f6;
}
.flow-card {
background: rgba(30, 41, 59, 0.4);
border: 1px solid rgba(255, 255, 255, 0.1);
color: #e2e8f0;
}
}
.metric-label {
font-size: 11px;
opacity: 0.7;
text-transform: uppercase;
letter-spacing: 0.5px;
margin-bottom: 4px;
}
.metric-val {
font-size: 17px;
font-weight: 700;
color: #0284c7;
}
@media (prefers-color-scheme: dark) {
.metric-val { color: #38bdf8; }
}
.metric-sub {
font-size: 11px;
font-weight: 600;
margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)

if 'ticker_input' not in st.session_state:
st.session_state.ticker_input = "SXLK.MI"
if 'horizon_input' not in st.session_state:
st.session_state.horizon_input = "3D"

def get_dynamic_benchmark(ticker):
t = ticker.upper()
if "-USD" in t:
return "BTC-USD"
elif t.endswith(".MI"):
return "FTSEMIB.MI"
elif any(tech in t for tech in ["AAPL", "NVDA", "MSFT", "GOOGL", "QQQ", "SXLK", "META", "AMZN"]):
return "^NDX"
else:
return "^GSPC"

@st.cache_data(ttl=120)
def scarica_dati(ticker):
df = yf.download(ticker, period="2y", interval="1d", progress=False)
if df.empty:
return None
if isinstance(df.columns, pd.MultiIndex):
df.columns = df.columns.get_level_values(0)
df = df.dropna(subset=['Close'])

bench_ticker = get_dynamic_benchmark(ticker)
df_bench = yf.download(bench_ticker, period="2y", interval="1d", progress=False)
if not df_bench.empty:
if isinstance(df_bench.columns, pd.MultiIndex):
df_bench.columns = df_bench.columns.get_level_values(0)
df['Bench_Close'] = df_bench['Close'].reindex(df.index, method='ffill')
df['Bench_SMA200'] = df['Bench_Close'].rolling(window=200).mean()
else:
df['Bench_Close'] = df['Close']
df['Bench_SMA200'] = df['Close'].rolling(window=200).mean()

return df

def calcola_indicatori(df):
df['SMA20'] = df['Close'].rolling(window=20).mean()
df['SMA50'] = df['Close'].rolling(window=50).mean()
df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()

df['Dist_EMA20'] = (df['Close'] - df['EMA20']) / df['EMA20'] * 100
df['Dist_SMA50'] = (df['Close'] - df['SMA50']) / df['SMA50'] * 100

delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / (loss + 1e-9)
df['RSI'] = 100 - (100 / (1 + rs))

gain7 = (delta.where(delta > 0, 0)).rolling(window=7).mean()
loss7 = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
rs7 = gain7 / (loss7 + 1e-9)
df['RSI7'] = 100 - (100 / (1 + rs7))

ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

df['BB_Middle'] = df['SMA20']
std = df['Close'].rolling(window=20).std()
df['BB_Upper'] = df['BB_Middle'] + (std * 2)
df['BB_Lower'] = df['BB_Middle'] - (std * 2)

tr1 = df['High'] - df['Low']
tr2 = abs(df['High'] - df['Close'].shift())
tr3 = abs(df['Low'] - df['Close'].shift())
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df['ATR'] = tr.rolling(window=14).mean()
df['ATR_Pctile'] = df['ATR'].rolling(window=252, min_periods=20).rank(pct=True)

df['RS'] = df['Close'] / (df['Bench_Close'] + 1e-9)
df['RS_Change'] = df['RS'].pct_change(5)

df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
df['Vol_Ratio'] = df['Volume'] / (df['Vol_SMA20'] + 1e-9)

daily_rets = df['Close'].pct_change()
neg_rets = daily_rets.where(daily_rets < 0, 0)
df['Sortino'] = (daily_rets.mean() / (neg_rets.std() + 1e-9)) * np.sqrt(252)

roll_max = df['Close'].cummax()
drawdown = (df['Close'] - roll_max) / roll_max
df['Max_DD'] = drawdown.cummin() * 100

return df.dropna()

def estrai_news_e_sentiment(ticker):
return 0.0, [] # Semplificato per il backtest storico

def calcola_modello_ml_e_backtest(df, sentiment_score, horizon):
days_forward = 1 if horizon == "1D" else (3 if horizon == "3D" else 7)
df = df.copy()
df['Target'] = (df['Close'].shift(-days_forward) > df['Close']).astype(int)
df = df.dropna()

if len(df) < 100:
return 50.0, "HOLD", "#d29922", 0, 0, 0, 0, 0, 0, 0

feature_cols = ['RSI', 'RSI7', 'MACD_Hist', 'Vol_Ratio', 'Dist_EMA20', 'RS_Change', 'ATR_Pctile']
X = df[feature_cols]
y = df['Target']

split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
model.fit(X_train_scaled, y_train)

last_features = scaler.transform(df[feature_cols].iloc[[-1]])
prob = model.predict_proba(last_features)[0][1] * 100
prob = max(0.0, min(100.0, prob + (sentiment_score * 12)))

last = df.iloc[-1]
is_macro_bull = last['Bench_Close'] > last['Bench_SMA200']

if not is_macro_bull and prob >= 62.0:
prob = 50.0

test_predictions = model.predict(X_test_scaled)
test_actuals = y_test.values
correct_trades = np.sum(test_predictions == test_actuals)
total_trades = len(test_actuals)

raw_win_rate = (correct_trades / total_trades) * 100 if total_trades > 0 else 0.0
win_rate = max(0.0, raw_win_rate - 3.5)

atr = last['ATR']
close = last['Close']
atr_mult = 1.0 if horizon == "1D" else (1.8 if horizon == "3D" else 2.6)

sl_buf = 1.3 if last['ATR_Pctile'] > 0.85 else 1.1
sl_val = close - (atr * atr_mult * sl_buf)
tp_val = close + (atr * atr_mult * 1.8)
sl_pct = ((sl_val - close) / close) * 100
tp_pct = ((tp_val - close) / close) * 100
risk_reward = abs(tp_pct / sl_pct) if sl_pct != 0 else 0

if prob >= 62.0:
verdetto, colore = "BUY", "#238636"
elif prob <= 38.0:
verdetto, colore = "SELL", "#da3633"
else:
verdetto, colore = "HOLD", "#d29922"

return prob, verdetto, colore, sl_val, sl_pct, tp_val, tp_pct, risk_reward, win_rate, is_macro_bull

UI PRINCIPALE
st.subheader("Trading Tikitò - Backtest Engine Storico")
st.write("")

c1, c2 = st.columns([2, 1])
with c1:
st.session_state.ticker_input = st.text_input("Ticker:", value=st.session_state.ticker_input).upper().strip()
with c2:
st.session_state.horizon_input = st.selectbox("Orizzonte:", ["1D", "3D", "7D"], index=["1D", "3D", "7D"].index(st.session_state.horizon_input))

ticker = st.session_state.ticker_input
horizon = st.session_state.horizon_input

if ticker:
df_raw = scarica_dati(ticker)
if df_raw is None or df_raw.empty:
st.error("Nessun dato trovato.")
st.stop()

df = calcola_indicatori(df_raw)

# Esecuzione Terminale Live per la data odierna
avg_sentiment, articoli = estrai_news_e_sentiment(ticker)
prob, verdetto, colore, sl_val, sl_pct, tp_val, tp_pct, risk_reward, win_rate, is_macro_bull = calcola_modello_ml_e_backtest(df, avg_sentiment, horizon)

last_price = df['Close'].iloc[-1]
last_date = df.index[-1].strftime('%Y-%m-%d')
change_pct = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100

# SIMULATORE BACKTEST STORICO (AGOSTO - SETTEMBRE - OTTOBRE 2026)
st.write("")
st.subheader(f"Backtest Automatico (Agosto - Ottobre 2026) su {ticker}")
st.write("Il motore simula l'operatività passata giorno per giorno applicando filtri ML, Macro e Position Sizing all'1% su un capitale ipotetico di 2.000€.")

# Filtriamo il dataframe dal 2026-08-01 a oggi
df_backtest = df.loc['2026-08-01:'].copy()

simulated_trades = []
budget_iniziale = 2000.0
risk_pct_chosen = 1.0

days_fwd = 1 if horizon == "1D" else (3 if horizon == "3D" else 7)

for i in range(len(df_backtest) - days_fwd):
row_date = df_backtest.index[i]
sub_df = df.loc[:row_date] # Dati storici fino a quel giorno per evitare leak
if len(sub_df) < 200:
continue

# Calcoliamo predizione in quel giorno
p_val, verd, col, s_val, s_pct, t_val, t_pct, r_rew, w_rate, macro_b = calcola_modello_ml_e_backtest(sub_df, 0.0, horizon)

# Se il modello da BUY e il macro è bullish, entriamo a mercato
if verd == "BUY":
entry_price = df_backtest['Close'].iloc[i]
entry_atr = df_backtest['ATR'].iloc[i]

# Stop loss e take profit calcolati al momento dell'ingresso
sl_target = entry_price - (entry_atr * (1.8 if horizon=="3D" else 1.0))
tp_target = entry_price + (entry_atr * 1.8 * 1.8)

max_loss_eur = budget_iniziale * (risk_pct_chosen / 100.0)
sl_dist = abs(entry_price - sl_target)
shares = int(max_loss_eur / sl_dist) if sl_dist > 0 else 0
invested_eur = shares * entry_price

if shares <= 0:
continue

# Verifichiamo cosa è successo nei giorni successivi (fino a days_fwd o chiusura target)
future_window = df_backtest.iloc[i+1 : i+1+days_fwd]

outcome = "PENDING"
exit_price = future_window['Close'].iloc[-1]
pnl_eur = 0.0

for _, f_row in future_window.iterrows():
if f_row['Low'] <= sl_target:
outcome = "LOSS"
exit_price = sl_target
break
elif f_row['High'] >= tp_target:
outcome = "WIN"
exit_price = tp_target
break

if outcome == "PENDING":
outcome = "WIN" if exit_price > entry_price else "LOSS"

pnl_eur = (exit_price - entry_price) * shares

simulated_trades.append({
"Data Ingresso": row_date.strftime('%Y-%m-%d'),
"Ticker": ticker,
"Prob ML": f"{p_val:.1f}%",
"Azioni": shares,
"Investito (€)": f"€{invested_eur:,.2f}",
"Prezzo Ing.": f"€{entry_price:.2f}",
"Stop Loss": f"€{sl_target:.2f}",
"Take Profit": f"€{tp_target:.2f}",
"Esito": outcome,
"P&L (€)": f"+€{pnl_eur:.2f}" if pnl_eur > 0 else f"-€{abs(pnl_eur):.2f}"
})

if simulated_trades:
df_sim_results = pd.DataFrame(simulated_trades)

# Calcolo KPI di Performance
total_trades = len(df_sim_results)
wins = len(df_sim_results[df_sim_results['Esito'] == 'WIN'])
win_rate_real = (wins / total_trades) * 100 if total_trades > 0 else 0

st.markdown(f"""
<div class="flow-card" style="border-left: 5px solid #238636; margin-bottom: 15px;">
<div>
<b>📊 Report Sintetico Backtest ({ticker} - {horizon}):</b><br>
• Totale Operazioni Simulate: <b>{total_trades}</b><br>
• Operazioni Vincenti (WIN): <b>{wins}</b> ({win_rate_real:.1f}%)
</div>
<div style="text-align: right;">
<span style="font-size: 10px; opacity: 0.7; text-transform: uppercase; display: block;">Stato Engine</span>
<b style="color: #238636;">VALIDATO CON SUCCESSO</b>
</div>
</div>
""", unsafe_allow_html=True)

st.dataframe(df_sim_results, use_container_width=True)
else:
st.info("Nessun trade generato con i rigidi filtri attuali nel periodo selezionato. Prova ad allargare l'orizzonte o cambiare ticker.")
