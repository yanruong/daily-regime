import os, json, numpy as np, pandas as pd
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
import requests

# === TELEGRAM ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_message(msg: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": msg})
        r.raise_for_status()
    except Exception as e:
        print("⚠️ Failed to send Telegram message:", str(e))

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

# === UTILS ===
def _serialize(arr): return [float(x) if np.isfinite(x) else ("inf" if x>0 else "-inf") for x in arr]
def _deserialize(lst): return np.array([np.inf if v=="inf" else -np.inf if v=="-inf" else float(v) for v in lst])

def save_month_bins(month_start, r_bins, v_bins, out_dir):
    key = pd.Timestamp(month_start.year, month_start.month, 1).strftime("%Y-%m")
    p = Path(out_dir) / "monthly_bins.json"
    store = {}
    if p.exists():
        store = json.loads(p.read_text())
    if key not in store:   # ✅ only save once
        store[key] = {"range_bins": _serialize(r_bins), "vol_bins": _serialize(v_bins)}
        p.write_text(json.dumps(store, indent=2))

def load_month_bins(month_start, out_dir):
    key = pd.Timestamp(month_start.year, month_start.month, 1).strftime("%Y-%m")
    p = Path(out_dir) / "monthly_bins.json"
    if not p.exists(): return None
    store = json.loads(p.read_text())
    if key in store:
        rb = _deserialize(store[key]["range_bins"])
        vb = _deserialize(store[key]["vol_bins"])
        return rb, vb
    return None

def save_daily_result(date, result, out_dir):
    p = Path(out_dir) / "regime_history.json"
    store = {}
    if p.exists():
        store = json.loads(p.read_text())
    if date not in store:   # ✅ don’t overwrite existing
        store[date] = result
        p.write_text(json.dumps(store, indent=2))

# === CORE FUNCTIONS ===
def load_intraday_epoch_s(df_in, tz=TZ, time_col="time"):
    df = df_in.copy()
    df[time_col] = df[time_col].astype(str).str.strip().astype(float)
    t = pd.to_datetime(df[time_col], unit="s", utc=True, errors="coerce")
    if t.isna().all(): raise ValueError("All time values failed to parse. Check 'time' column.")
    df = df.drop(columns=[time_col]).set_index(t).sort_index()
    if df.index.tz is None: df.index = df.index.tz_localize("UTC")
    df = df.tz_convert(tz)
    for col in ["open","high","low","close"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    df.columns = [c.lower() for c in df.columns]
    return df

def daily_prevday_features(df, atr_n=14, vol_n=20):
    day = df.index.normalize()
    g = (df.assign(_day=day)
           .groupby('_day')
           .agg(high=('high','max'), low=('low','min'), close=('close','last'))
           .dropna())
    g['ret'] = g['close'].pct_change()
    g['vol20'] = g['ret'].rolling(vol_n, min_periods=vol_n).std()
    prev_c = g['close'].shift(1)
    tr = pd.concat([(g['high']-g['low']).abs(), (g['high']-prev_c).abs(), (g['low']-prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_n, adjust=False).mean()
    g['atr_pct'] = atr / g['close'].replace(0, np.nan)
    g['range_prevday'] = g['atr_pct'].shift(1)
    g['vol20_prevday'] = g['vol20'].shift(1)
    return g

# === MAIN ===
def run_daily():
    df_raw = load_sheet(SHEET_ID, RANGE_NAME)
    df = load_intraday_epoch_s(df_raw, tz=TZ)
    daily = daily_prevday_features(df)

    today = df.index.max().normalize()
    month_start = pd.Timestamp(today.year, today.month, 1, tz=TZ)

    # load or fit bins
    bins = load_month_bins(month_start, OUT_DIR)
    if bins is None:
        hist = daily.loc[:month_start - pd.Timedelta("1ns"), ["range_prevday","vol20_prevday"]].dropna()
        if len(hist) < MIN_HIST_DAYS: raise ValueError("Insufficient history to fit bins")
        r_bins = hist["range_prevday"].quantile([.25,.5,.75]).to_numpy()
        v_bins = hist["vol20_prevday"].quantile([.25,.5,.75]).to_numpy()
        r_bins = np.array([-np.inf, *r_bins, np.inf])
        v_bins = np.array([-np.inf, *v_bins, np.inf])
        save_month_bins(month_start, r_bins, v_bins, OUT_DIR)
    else:
        r_bins, v_bins = bins

    # classify today
    rng = daily.loc[today, "range_prevday"]
    vol = daily.loc[today, "vol20_prevday"]

    R = pd.cut([rng], r_bins, labels=False, include_lowest=True)[0] if pd.notna(rng) else None
    V = pd.cut([vol], v_bins, labels=False, include_lowest=True)[0] if pd.notna(vol) else None

    cell = (int(R), int(V)) if R is not None and V is not None else None
    allow = (cell not in EXCLUDED_CELLS) if cell else False
    label = f"R{R}/V{V}" if cell else "NA"

    result = {
        "date": str(today.date()),
        "range_value": None if pd.isna(rng) else float(rng),
        "vol_value": None if pd.isna(vol) else float(vol),
        "Range_Q": None if R is None else int(R),
        "Vol_Q": None if V is None else int(V),
        "label": label,
        "trade": allow
    }

    # save daily result (frozen)
    save_daily_result(str(today.date()), result, OUT_DIR)

    # === load history and take last 3 days ===
    hist_path = OUT_DIR / "regime_history.json"
    history = {}
    if hist_path.exists():
        history = json.loads(hist_path.read_text())
    records = [history[d] for d in sorted(history.keys())]
    last3 = records[-3:]

    # build Telegram message
    msg_lines = ["📊 Regime Bot Update\n"]
    for rec in last3:
        trade_msg = "✅ TRADE" if rec["trade"] else "🚫 NO TRADE"
        msg_lines.append(
            f"📅 {rec['date']} → {trade_msg}\n"
            f"   Regime: {rec['label']}\n"
            f"   Range: {rec['range_value']:.4f}\n"
            f"   Vol:   {rec['vol_value']:.4f}\n"
        )
    text_summary = "\n".join(msg_lines)

    send_message(text_summary)

if __name__ == "__main__":
    try:
        run_daily()
        print("✅ Daily report done")
    except Exception as e:
        import traceback
        send_message(f"❌ Run failed: {type(e).__name__}: {str(e)}")
        print(traceback.format_exc())
        raise
