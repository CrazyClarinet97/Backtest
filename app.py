import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date

st.set_page_config(
    page_title="Institutional Quant Lab — Dynamic Volatility Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background: rgba(25, 30, 45, 0.45);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
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

@st.cache_data(ttl=120)
def scarica_dati(ticker):
    df = yf.download(ticker, period="3y", interval="1d", progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=['Close', 'Volume'])

def calcola_indicatori_avanzati(df):
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()

    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift())
    tr3 = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
    
    df['Trend_Bull'] = (df['Close'] > df['SMA50']) & (df['Close'] > df['SMA200'])
    df['Pullback'] = (df['Low'].shift(1) <= df['EMA20']) | (df['Low'].shift(2) <= df['EMA20'])
    
    df['Trigger_Buy'] = (
        df['Trend_Bull'] & 
        df['Pullback'] & 
        (df['Close'] > df['Close'].shift(1)) & 
        (df['Volume'] > df['Vol_SMA20'] * 1.1)
    )

    return df.dropna()

st.subheader("⚡ Institutional Quant Lab — Dynamic Volatility & Root-Time Scaling")
st.write("Modello con adattamento geometrico del Take Profit basato sulla radice quadrata del tempo.")
st.write("")

c1, c2 = st.columns([2, 1])
with c1:
    st.session_state.ticker_input = st.text_input("Ticker di Riferimento:", value=st.session_state.ticker_input).upper().strip()
with c2:
    st.session_state.horizon_input = st.selectbox("Orizzonte di Uscita Max:", ["3D", "5D", "7D"], index=0)

p1, p2, p3, p4 = st.columns(4)
with p1:
    start_date_input = st.date_input("Data Inizio:", value=date(2026, 1, 1))
with p2:
    end_date_input = st.date_input("Data Fine:", value=date(2026, 10, 31))
with p3:
    budget_iniziale = st.number_input("Capitale Iniziale (€):", min_value=100.0, max_value=100000.0, value=2000.0, step=100.0)
with p4:
    risk_pct_chosen = st.number_input("Rischio Max per Trade (%):", min_value=0.1, max_value=10.0, value=2.0, step=0.1)

ticker = st.session_state.ticker_input
horizon = st.session_state.horizon_input

if ticker:
    df_raw = scarica_dati(ticker)
    if df_raw is None or df_raw.empty:
        st.error("Nessun dato trovato per il ticker inserito.")
        st.stop()
        
    df = calcola_indicatori_avanzati(df_raw)
    
    st.write("")
    st.subheader(f"📊 Backtest Dinamico ({ticker}) | Periodo: {start_date_input} al {end_date_input}")
    st.write(f"Parametri: Capitale **€{budget_iniziale:,.2f}** | Rischio **{risk_pct_chosen}%** | R:R **1:2** | Orizzonte **{horizon}**")

    try:
        df_backtest = df.loc[str(start_date_input):str(end_date_input)].copy()
    except Exception:
        df_backtest = df.copy()

    if df_backtest.empty:
        st.warning("Il periodo selezionato non contiene dati sufficienti.")
        st.stop()
    
    simulated_trades = []
    days_map = {"3D": 3, "5D": 5, "7D": 7}
    days_fwd = days_map.get(horizon, 3)
    
    # Fattore di scala dinamico basato sulla radice quadrata del tempo (Random Walk Theory)
    time_scaling_factor = np.sqrt(days_fwd / 3.0)

    for i in range(len(df_backtest) - days_fwd):
        row_date = df_backtest.index[i]
        
        if df_backtest['Trigger_Buy'].iloc[i]:
            entry_price = df_backtest['Close'].iloc[i]
            entry_atr = df_backtest['ATR'].iloc[i]
            
            # SL e TP scalati dinamicamente in base all'orizzonte temporale scelto
            sl_dist = entry_atr * 1.5 * time_scaling_factor
            tp_dist = sl_dist * 2.0  # Rapporto R:R 1:2 geometrico
            
            sl_target = entry_price - sl_dist
            tp_target = entry_price + tp_dist
            
            max_loss_eur = budget_iniziale * (risk_pct_chosen / 100.0)
            shares = int(max_loss_eur / sl_dist) if sl_dist > 0 else 0
            invested_eur = shares * entry_price
            
            if shares <= 0:
                continue
                
            future_window = df_backtest.iloc[i+1 : i+1+days_fwd]
            
            outcome = "PENDING"
            exit_price = future_window['Close'].iloc[-1]
            exit_date = future_window.index[-1].strftime('%Y-%m-%d')
            
            for f_date, f_row in future_window.iterrows():
                if f_row['Low'] <= sl_target:
                    outcome = "LOSS"
                    exit_price = sl_target
                    exit_date = f_date.strftime('%Y-%m-%d')
                    break
                elif f_row['High'] >= tp_target:
                    outcome = "WIN"
                    exit_price = tp_target
                    exit_date = f_date.strftime('%Y-%m-%d')
                    break
            
            # Se scade il tempo senza toccare né TP né SL, valuta rigorosamente l'esito reale a fine periodo
            if outcome == "PENDING":
                outcome = "WIN" if exit_price > entry_price else "LOSS"
                
            pnl_eur = (exit_price - entry_price) * shares
            
            simulated_trades.append({
                "Data Ingresso": row_date.strftime('%Y-%m-%d'),
                "Data Uscita": exit_date,
                "Ticker": ticker,
                "Azioni": shares,
                "Capitale Investito (€)": f"€{invested_eur:,.2f}",
                "Prezzo Ingresso (€)": f"€{entry_price:.2f}",
                "Stop Loss (€)": f"€{sl_target:.2f}",
                "Take Profit (€)": f"€{tp_target:.2f}",
                "Prezzo Uscita (€)": f"€{exit_price:.2f}",
                "Esito": outcome,
                "P&L (€)": pnl_eur
            })

    if simulated_trades:
        df_sim_results = pd.DataFrame(simulated_trades)
        
        total_trades = len(df_sim_results)
        wins = len(df_sim_results[df_sim_results['Esito'] == 'WIN'])
        losses = len(df_sim_results[df_sim_results['Esito'] == 'LOSS'])
        win_rate_real = (wins / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = df_sim_results['P&L (€)'].sum()
        final_budget = budget_iniziale + total_pnl
        
        gross_profit = df_sim_results[df_sim_results['P&L (€)'] > 0]['P&L (€)'].sum()
        gross_loss = abs(df_sim_results[df_sim_results['P&L (€)'] < 0]['P&L (€)'].sum())
        
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = 99.0 if gross_profit > 0 else 0.0

        df_sim_results['P&L (€)'] = df_sim_results['P&L (€)'].apply(lambda x: f"+€{x:.2f}" if x > 0 else f"-€{abs(x):.2f}")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Win Rate Reale</div>
                    <div class="metric-val" style="color: #238636;">{win_rate_real:.1f}%</div>
                    <div class="metric-sub">{wins} Win su {total_trades} Trade</div>
                </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Profit Factor</div>
                    <div class="metric-val" style="color: #38bdf8;">{profit_factor:.2f}</div>
                    <div class="metric-sub">Target: > 1.40</div>
                </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">P&L Netto Totale</div>
                    <div class="metric-val" style="color: {'#238636' if total_pnl >= 0 else '#da3633'};">+€{total_pnl:.2f}</div>
                    <div class="metric-sub">Rend: {(total_pnl/budget_iniziale)*100:+.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Capitale Finale</div>
                    <div class="metric-val">€{final_budget:,.2f}</div>
                    <div class="metric-sub">Start: €{budget_iniziale:,.2f}</div>
                </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.dataframe(df_sim_results, use_container_width=True)
    else:
        st.info("Nessun trade generato con i nuovi parametri dinamici.")
