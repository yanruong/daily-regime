import os, json, numpy as np, pandas as pd 
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
import requests

# === DEBUG MARKER ===
print("DEBUG: Running main.py revision marker v2025-08-26-B (with correct ADX)")

# === TELEGRAM ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")  # set in GitHub Secrets

def send_message(msg: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": msg})
        r.raise_for_status()
    except Exception as e:
        print("⚠️ Failed to send Telegram message:", str(e))

def send_file(path: Path):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    try:
        with open(path, "rb") as f:
            r = requests.post(url, data={"chat_id": CHAT_ID}, files={"document": f})
            r.raise_for_status()
    except Exception as e:
        print("⚠️ Failed to send Telegram file:", str(e))

# === GOOGLE SHEETS LOADER ===
def load_sheet(sheet_id, range_name):
    with open("creds.json", "w") as f:
        f.write(os.getenv("GOOGLE_CREDS"))

    creds = service_account.Credentials.from_service_account_file(
        "creds.json", scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
    )
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()

    result = sheet.values().get(spreadsheetId=sheet_id, range=range_name).execute()
    values = result.get("values", [])

    if not values:
        raise ValueError("No data found in Google Sheet")

    return pd.DataFrame(values[1:], columns=values[0])

# === CONFIG ===
TZ = "Asia/Singapore"
SHEET_ID = "1MwTlXHwGt10Somh4v2sRJjD1VWCvdCYyS5JzLEApN0c"
RANGE_NAME = "new!A:E"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)
MIN_HIST_DAYS = 60
EXCLUDED_CELLS = {(0,1), (2,0), (1,0), (3,2), (1,2)}

# === FUNCTIONS ===
def load_intraday_epoch_s(df_in, tz=TZ, time_col="time"):
    df = df_in.copy()
    df[time_col] = df[time_col].astype(str).str.strip().astype(float)
    t = pd.to_datetime(df[time_col], unit="s", utc=True, errors="coerce")
    if t.isna().all():
        raise ValueError("All time values failed to parse. Check 'time' column.")
    df = df.drop(columns=[time_col]).set_index(t).sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.tz_convert(tz)
    for col in ["open","high","low","close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.columns = [c.lower() for c in df.columns]
    return df

def daily_prevday_features(df, tz=TZ, atr_n=14, vol_n=20):
    day = df.index.tz_convert(tz).normalize()
    g = (df.assign(_day=day)
           .groupby('_day')
           .agg(high=('high','max'), low=('low','min'), close=('close','last')))
    g['ret']   = g['close'].pct_change()
    g['vol20'] = g['ret'].rolling(vol_n, min_periods=1).std()
    prev_c = g['close'].shift(1)
    tr = pd.concat([(g['high']-g['low']).abs(),
                    (g['high']-prev_c).abs(),
                    (g['low'] -prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_n, adjust=False).mean()
    g['atr_pct'] = atr / g['close'].replace(0, np.nan)
    g['rolling_range_prevday'] = g['atr_pct'].shift(1)
    g['daily_vol20_prevday']   = g['vol20'].shift(1)
    return g[['rolling_range_prevday','daily_vol20_prevday']]

def label_walkforward_quartiles_generic(df_in, range_col, vol_col, out_prefix='roll_', min_hist_days=MIN_HIST_DAYS, tz=TZ):
    df = df_in.copy()
    out_R, out_V = f'{out_prefix}Range_Q', f'{out_prefix}Vol_Q'
    df[out_R] = pd.Series(pd.NA, index=df.index, dtype='Int64')
    df[out_V] = pd.Series(pd.NA, index=df.index, dtype='Int64')
    idx_local = df.index.tz_convert(tz)
    if idx_local.empty:
        raise ValueError("Datetime index is empty — check your 'time' column parsing.")
    first_ms  = pd.Timestamp(idx_local.min().year, idx_local.min().month, 1, tz=tz)
    last_ms   = pd.Timestamp(idx_local.max().year,  idx_local.max().month,  1, tz=tz)
    month_starts = pd.date_range(first_ms, last_ms, freq='MS', tz=tz)
    for i in range(1, len(month_starts)):
        m0 = month_starts[i]
        m1 = month_starts[i+1] if i+1 < len(month_starts) else (m0 + pd.offsets.MonthBegin(1))
        hist = df.loc[:m0 - pd.Timedelta('1ns'), [range_col, vol_col]].dropna()
        if hist.index.normalize().nunique() < min_hist_days:
            continue
        rq = hist[range_col].quantile([.25,.50,.75]).to_list()
        vq = hist[vol_col].quantile([.25,.50,.75]).to_list()
        r_bins = np.array([-np.inf, *rq, np.inf], float)
        v_bins = np.array([-np.inf, *vq, np.inf], float)
        sel = (df.index >= m0) & (df.index < m1)
        df.loc[sel, out_R] = pd.cut(df.loc[sel, range_col], r_bins, labels=False, include_lowest=True).astype('Int64')
        df.loc[sel, out_V] = pd.cut(df.loc[sel,  vol_col], v_bins,   labels=False, include_lowest=True).astype('Int64')
    return df

# === ADX (Wilder’s smoothing, TradingView match) ===
def compute_adx(df, n=14, high_col='high', low_col='low', close_col='close'):
    h, l, c = df[high_col], df[low_col], df[close_col]
    up_move   = h.diff()
    down_move = l.shift(1) - l

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = h - l
    tr2 = (h - c.shift()).abs()
    tr3 = (l - c.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    alpha = 1.0 / n
    atr      = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean()
    plus_dm_s  = pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()

    pdi = 100.0 * (plus_dm_s / atr).replace([np.inf, -np.inf], np.nan)
    mdi = 100.0 * (minus_dm_s / atr).replace([np.inf, -np.inf], np.nan)

    dx = 100.0 * (pdi.subtract(mdi).abs() / (pdi + mdi)).replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx

# === MAIN ===
def run_daily():
    df_raw = load_sheet(SHEET_ID, RANGE_NAME)
    df = load_intraday_epoch_s(df_raw, tz=TZ)

    # compute ADX(14) per 2h bar
    df['adx'] = compute_adx(df, n=14)

    daily = daily_prevday_features(df, tz=TZ)
    day_key = df.index.normalize()
    df_roll = df.copy()
    df_roll['rolling_range_prevday'] = day_key.map(daily['rolling_range_prevday'])
    df_roll['daily_vol20_prevday']   = day_key.map(daily['daily_vol20_prevday'])
    df_roll = label_walkforward_quartiles_generic(
        df_roll,
        range_col='rolling_range_prevday',
        vol_col='daily_vol20_prevday',
        out_prefix='roll_',
        min_hist_days=MIN_HIST_DAYS,
        tz=TZ
    )

    day = df_roll.index.normalize()
    daily_lbl = (
        df_roll.assign(_day=day)
               .groupby('_day')[['roll_Range_Q','roll_Vol_Q','rolling_range_prevday','daily_vol20_prevday']]
               .first()
               .astype({'roll_Range_Q':'Int64','roll_Vol_Q':'Int64'})
    )

    # yesterday’s last 2h ADX
    last_bar_adx = df.groupby(df.index.normalize())['adx'].last()
    last_bar_adx = last_bar_adx.shift(1)
    daily_lbl = daily_lbl.join(last_bar_adx.rename("adx_prevday"), how="left")

    print("DEBUG daily_lbl columns:", daily_lbl.columns.tolist())

    last10 = daily_lbl.tail(10)
    last10_records = []
    for idx, row in last10.iterrows():
        cell = (int(row["roll_Range_Q"]) if pd.notna(row["roll_Range_Q"]) else -1,
                int(row["roll_Vol_Q"]) if pd.notna(row["roll_Vol_Q"]) else -1)
        allow_trade = cell not in EXCLUDED_CELLS

        label_str = (
            f"R{int(row['roll_Range_Q'])}/V{int(row['roll_Vol_Q'])}"
            if pd.notna(row['roll_Range_Q']) and pd.notna(row['roll_Vol_Q'])
            else "NA"
        )

        rec = {
            "date": idx.strftime("%Y-%m-%d"),
            "roll_Range_Q": None if pd.isna(row["roll_Range_Q"]) else int(row["roll_Range_Q"]),
            "roll_Vol_Q": None if pd.isna(row["roll_Vol_Q"]) else int(row["roll_Vol_Q"]),
            "label": label_str,
            "range_value": None if pd.isna(row["rolling_range_prevday"]) else float(row["rolling_range_prevday"]),
            "vol_value": None if pd.isna(row["daily_vol20_prevday"]) else float(row["daily_vol20_prevday"]),
            "adx_value": None if pd.isna(row["adx_prevday"]) else float(row["adx_prevday"]),
            "trade": allow_trade
        }
        last10_records.append(rec)

    summary = {"last10": last10_records}
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Telegram summary
    last_days = last10_records[-3:]
    msg_lines = ["📊 Regime Bot Update\n"]
    for day in last_days:
        trade_msg = "✅ TRADE" if day["trade"] else "🚫 NO TRADE"
        msg_lines.append(
            f"📅 {day['date']} → {trade_msg}\n"
            f"   Regime: {day['label']}\n"
            f"   Range: {day['range_value']}\n"
            f"   Vol:   {day['vol_value']}\n"
            f"   ADX(14): {day['adx_value']}\n"
        )
    text_summary = "\n".join(msg_lines)

    send_message(text_summary)
    send_file(summary_path)

if __name__ == "__main__":
    try:
        run_daily()
        print("✅ Daily report completed and sent to Telegram.")
    except Exception as e:
        import traceback
        send_message(f"❌ Daily run failed: {type(e).__name__}: {str(e)}")
        print("⚠️ Error in run_daily:\n", traceback.format_exc())
        raise
