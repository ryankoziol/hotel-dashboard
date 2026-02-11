import streamlit as st
import pandas as pd
import os
import time
import re
import io
import json
import hashlib
import zipfile
import math 
import altair as alt
from urllib.parse import urlparse
from datetime import datetime, timedelta
from dotenv import load_dotenv
from serpapi import GoogleSearch
import openai

# ---------------------------
# 1) CONFIGURATION & SETUP
# ---------------------------
st.set_page_config(page_title="Reputation Dashboard", page_icon="🏨", layout="wide", initial_sidebar_state="collapsed")
load_dotenv()

# [FIX] PRIORITY SECRETS LOADING (Cloud Stability)
def get_secret(key):
    if key in st.secrets:
        return st.secrets[key].strip()
    return os.getenv(key, "").strip()

SERPAPI_KEY = get_secret("SERPAPI_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")

# -- FILE PATHS --
if not os.path.exists("data"):
    os.makedirs("data")

CACHE_FILE = "data/production_cache.csv"
USERS_FILE = "data/users.csv"
PORTFOLIOS_FILE = "data/portfolios.json"
HISTORY_FILE = "data/history.csv"
ACTIVITY_FILE = "data/activity_log.csv"

TTL_DAYS = 14

# --- [FIX] SAFETY HELPER FUNCTIONS ---
def safe_float(val):
    """Prevents crashes by forcing empty/None values to 0.0"""
    try:
        if pd.isna(val) or val == "" or val is None: return 0.0
        return float(val)
    except: return 0.0

def safe_int(val):
    """Prevents crashes by forcing empty/None values to 0"""
    try:
        if pd.isna(val) or val == "" or val is None: return 0
        return int(float(val))
    except: return 0

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    [data-testid="InputInstructions"] { display: none; }
    
    h1 {
        font-size: 2.5rem !important; 
        font-weight: 800 !important;
        color: #ffffff !important;
        padding-bottom: 10px;
    }
    
    /* SIDEBAR STYLES */
    section[data-testid="stSidebar"] .stButton button {
        width: 100%;
        border-radius: 4px;
        height: 3em;
        font-weight: 600;
        margin-bottom: 5px;
        border: 1px solid #4facfe;
    }

    /* GLOBAL BUTTON STYLES */
    div.stButton > button[kind="primary"] {
        background-color: #28a745 !important;
        border: 1px solid #28a745 !important;
        color: white !important;
        font-weight: bold;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #218838 !important;
        border-color: #1e7e34 !important;
    }
    div[data-testid="column"]:nth-of-type(2) div.stButton > button {
        border: 1px solid #ff4b4b !important;
        color: #ff4b4b !important;
        background-color: transparent !important;
    }
    div[data-testid="column"]:nth-of-type(2) div.stButton > button:hover {
        background-color: #ff4b4b !important;
        color: white !important;
    }

    /* METRIC BOXES */
    .metric-box {
        background-color: #262730;
        border: 1px solid #4facfe;
        border-radius: 6px;
        padding: 15px;
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-title {
        color: #bbb;
        font-size: 0.85rem;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* TOP/BOTTOM PERFORMERS */
    .performer-card {
        background-color: #1e2530;
        border-left: 4px solid #555;
        padding: 10px;
        margin-bottom: 8px;
        border-radius: 0 4px 4px 0;
    }
    .p-name {
        font-weight: bold;
        font-size: 0.95em;
        color: white;
    }
    .p-score {
        float: right;
        font-weight: bold;
        color: #ffd700;
    }
    .p-loc {
        font-size: 0.8em;
        color: #aaa;
    }

    /* HISTORY LOG */
    .hist-header {
        font-weight: bold;
        color: #4facfe;
        font-size: 0.75em;
        border-bottom: 1px solid #555;
        padding-bottom: 5px;
        margin-bottom: 5px;
        text-align: center;
        height: 40px;
        display: flex;
        align-items: end;
        justify-content: center;
    }
    .hist-val {
        font-size: 0.8em;
        text-align: center;
        color: #ddd;
        padding-top: 8px;
    }
    
    /* METHODOLOGY */
    .meth-header {
        color: #4facfe;
        font-weight: bold;
        margin-top: 10px;
        display: block;
    }
    .meth-body {
        margin-left: 15px;
        color: #ddd;
        font-size: 0.95em;
    }
    
    /* AI SUMMARY BOX */
    .ai-box {
        background-color: #1e2530;
        border: 1px solid #4facfe;
        border-radius: 8px;
        padding: 25px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .ai-title {
        color: #4facfe;
        font-weight: bold;
        font-size: 1.2em;
        margin-bottom: 15px;
        text-transform: uppercase;
        border-bottom: 1px solid #555;
        padding-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# 2) AUTHENTICATION & TRACKING
# ---------------------------

def hash_password(password):
    return hashlib.sha256(str(password).encode('utf-8')).hexdigest()

def log_activity(email, action, details=""):
    if not os.path.exists(ACTIVITY_FILE):
        df = pd.DataFrame(columns=["timestamp", "email", "action", "details"])
        df.to_csv(ACTIVITY_FILE, index=False)
    
    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "email": email,
        "action": action,
        "details": details
    }
    pd.DataFrame([new_row]).to_csv(ACTIVITY_FILE, mode='a', header=False, index=False)

def load_users():
    if not os.path.exists(USERS_FILE):
        return pd.DataFrame(columns=["email", "password", "first_name", "last_name", "created_at"])
    return pd.read_csv(USERS_FILE)

def save_user(email, password, first_name, last_name):
    df = load_users()
    if email in df['email'].values:
        return False
    
    new_user = pd.DataFrame([{
        "email": email, 
        "password": hash_password(password),
        "first_name": first_name,
        "last_name": last_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])
    pd.concat([df, new_user], ignore_index=True).to_csv(USERS_FILE, index=False)
    log_activity(email, "Account Created", f"Name: {first_name} {last_name}")
    return True

def check_login(email, password):
    df = load_users()
    hashed = hash_password(password)
    user = df[(df['email'] == email) & (df['password'] == hashed)]
    
    if not user.empty:
        log_activity(email, "User Login", "Success")
        return user.iloc[0]['first_name']
    return None

def load_portfolios():
    if not os.path.exists(PORTFOLIOS_FILE):
        return {}
    try:
        with open(PORTFOLIOS_FILE, 'r') as f:
            return json.load(f)
    except: return {}

def save_portfolio(email, portfolio_name, hotel_ids):
    data = load_portfolios()
    if email not in data:
        data[email] = {}
    data[email][portfolio_name] = hotel_ids
    with open(PORTFOLIOS_FILE, 'w') as f:
        json.dump(data, f)
    log_activity(email, "Portfolio Saved", f"Name: {portfolio_name} ({len(hotel_ids)} hotels)")

def rename_portfolio(email, old_name, new_name):
    data = load_portfolios()
    if email in data and old_name in data[email]:
        if new_name in data[email]:
            return False # Target name already exists
        # Move data to new key
        data[email][new_name] = data[email].pop(old_name)
        with open(PORTFOLIOS_FILE, 'w') as f:
            json.dump(data, f)
        log_activity(email, "Portfolio Renamed", f"{old_name} -> {new_name}")
        return True
    return False

def delete_portfolio(email, portfolio_name):
    data = load_portfolios()
    if email in data and portfolio_name in data[email]:
        del data[email][portfolio_name]
        with open(PORTFOLIOS_FILE, 'w') as f:
            json.dump(data, f)
        log_activity(email, "Portfolio Deleted", f"Name: {portfolio_name}")
        return True
    return False

# HISTORY LOGGING
def check_history_schema():
    expected_cols = [
        "timestamp", "email", "group_name", "hotel_count",
        "avg_google_norm", "avg_expedia_norm", "avg_ta_norm", "avg_booking_norm",
        "total_reviews_google", "total_reviews_expedia", "total_reviews_ta", "total_reviews_booking",
        "weighted_avg_score"
    ]
    if not os.path.exists(HISTORY_FILE):
        pd.DataFrame(columns=expected_cols).to_csv(HISTORY_FILE, index=False)
    else:
        try:
            pd.read_csv(HISTORY_FILE)
        except:
            pd.DataFrame(columns=expected_cols).to_csv(HISTORY_FILE, index=False)

def log_history(email, group_name, df_results):
    check_history_schema()
    df_clean = df_results[df_results['Name'] != "Total / Weighted Average"].copy()
    
    def safe_mean(col): return round(df_clean[col].mean(), 1) if col in df_clean.columns else 0
    def safe_sum(col): return int(df_clean[col].sum()) if col in df_clean.columns else 0
    
    total_reviews = 0
    weighted_sum = 0
    if "Weighted Avg. Review Score" in df_clean.columns:
        for _, row in df_clean.iterrows():
            revs = (safe_int(row.get("# Google Reviews")) + safe_int(row.get("# Expedia Reviews")) + 
                    safe_int(row.get("# TripAdvisor Reviews")) + safe_int(row.get("# Booking Reviews")))
            score = safe_float(row.get("Weighted Avg. Review Score"))
            if revs > 0 and score > 0:
                weighted_sum += (score * revs)
                total_reviews += revs
    
    final_weighted_score = round(weighted_sum / total_reviews, 1) if total_reviews > 0 else 0

    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "email": email,
        "group_name": group_name,
        "hotel_count": len(df_clean),
        "avg_google_norm": safe_mean("Google Score (Normalized)"),
        "avg_expedia_norm": safe_mean("Expedia Score (Normalized)"),
        "avg_ta_norm": safe_mean("TripAdvisor Score (Normalized)"),
        "avg_booking_norm": safe_mean("Booking Score (Normalized)"),
        "total_reviews_google": safe_sum("# Google Reviews"),
        "total_reviews_expedia": safe_sum("# Expedia Reviews"),
        "total_reviews_ta": safe_sum("# TripAdvisor Reviews"),
        "total_reviews_booking": safe_sum("# Booking Reviews"),
        "weighted_avg_score": final_weighted_score
    }
    pd.DataFrame([new_row]).to_csv(HISTORY_FILE, mode='a', header=False, index=False)
    log_activity(email, "Snapshot Saved", f"Group: {group_name}")

def delete_history_row(timestamp, group_name):
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        # Filter where NOT matching the criteria
        df = df[~((df['timestamp'] == timestamp) & (df['group_name'] == group_name) & (df['email'] == st.session_state['user_email']))]
        df.to_csv(HISTORY_FILE, index=False)
        return True
    return False

# ---------------------------
# 3) CORE ENGINE
# ---------------------------
CACHE_COLUMNS = [
    "Hotel_ID", "Name", "City", "State", 
    "Google_Raw", "Google_N",
    "Expedia_Raw", "Expedia_N", 
    "TA_Raw", "TA_N", 
    "Booking_Raw", "Booking_N",
    "Review_Text_Raw",
    "Is_Verified", "Last_Scraped", "_resolved_name", "Debug"
]

def load_cache():
    if os.path.exists(CACHE_FILE): 
        try:
            df = pd.read_csv(CACHE_FILE, on_bad_lines='skip', engine='python')
            for c in CACHE_COLUMNS:
                if c not in df.columns: df[c] = None
            return df
        except: return pd.DataFrame(columns=CACHE_COLUMNS)
    return pd.DataFrame(columns=CACHE_COLUMNS)

def save_to_cache(new_row_data):
    df = pd.DataFrame([new_row_data])
    for c in CACHE_COLUMNS:
        if c not in df.columns: df[c] = None
    df = df[CACHE_COLUMNS]
    if not os.path.exists(CACHE_FILE): df.to_csv(CACHE_FILE, index=False)
    else: df.to_csv(CACHE_FILE, mode='a', header=False, index=False)

def is_cache_fresh(last_scraped_str):
    if pd.isna(last_scraped_str) or str(last_scraped_str) == 'None': return False
    try:
        dt = datetime.strptime(str(last_scraped_str), "%Y-%m-%d")
        return dt >= (datetime.now() - timedelta(days=TTL_DAYS))
    except: return False

def find_header_row(uploaded_file, file_type='csv'):
    uploaded_file.seek(0)
    if file_type == 'xlsx': preview = pd.read_excel(uploaded_file, header=None, nrows=10)
    else: preview = pd.read_csv(uploaded_file, header=None, nrows=10)
    
    header_idx = 0
    found = False
    for idx, row in preview.iterrows():
        row_str = " ".join([str(val).lower() for val in row.values])
        if "name" in row_str and "city" in row_str:
            header_idx = idx
            found = True
            break
    uploaded_file.seek(0)
    return header_idx if found else 0

def normalize_to_10(val, scale=5):
    # [FIX] Use safe_float to handle None
    v = safe_float(val)
    if v == 0 and (val is None or pd.isna(val)): return None
    if scale == 5: return round(v * 2, 1)
    return round(v, 1)

def extract_rating_from_text(text):
    if not text: return None, None
    m5 = re.search(r'(\d(\.\d)?)\s*/\s*5', text)
    if m5: return float(m5.group(1)), 5
    m10 = re.search(r'(\d(\.\d)?)\s*/\s*10', text)
    if m10: return float(m10.group(1)), 10
    m_score = re.search(r'(?:Score|Rating)[:\s]+(\d(\.\d)?)', text, re.IGNORECASE)
    if m_score:
        val = float(m_score.group(1))
        return val, 10 if val > 5 else 5
    return None, None

def scavenge_knowledge_graph(kg, out, debug_notes):
    if not kg: return
    lists_to_check = []
    if "reviews_from_the_web" in kg: lists_to_check.append(kg["reviews_from_the_web"])
    if "ratings" in kg: lists_to_check.append(kg["ratings"])
    for lst in lists_to_check:
        for item in lst:
            source = str(item.get("source", "")).lower() + str(item.get("name", "")).lower()
            value = str(item.get("rating", "") or item.get("value", ""))
            try: score = float(re.split(r'[/ ]', value)[0])
            except: continue
            if "tripadvisor" in source and not out["TA_Raw"]:
                out["TA_Raw"] = score
            elif "expedia" in source and not out["Expedia_Raw"]:
                out["Expedia_Raw"] = score
            elif "booking" in source and not out["Booking_Raw"]:
                out["Booking_Raw"] = score

def scan_organic_results(results, out, debug_notes, snippet_collection):
    for res in results:
        link = res.get("link", "").lower()
        title = res.get("title", "")
        snippet = res.get("snippet", "")
        
        if snippet and len(snippet) > 15:
            snippet_collection.append(snippet)

        rich = res.get("rich_snippet", {}) 
        rating = None
        reviews = None
        if "top" in rich:
            ext = rich.get("top", {}).get("detected_extensions", {})
            rating = ext.get("rating")
            reviews = ext.get("reviews")
        if not rating:
            rating = res.get("rating")
            reviews = res.get("reviews")
        if not rating:
            rating, _ = extract_rating_from_text(title + " " + snippet)
        if rating:
            if "tripadvisor.com" in link and not out["TA_Raw"]:
                out["TA_Raw"] = rating
                if reviews: out["TA_N"] = reviews
            elif "booking.com" in link and not out["Booking_Raw"]:
                out["Booking_Raw"] = rating
                if reviews: out["Booking_N"] = reviews
            elif "expedia.com" in link and not out["Expedia_Raw"]:
                out["Expedia_Raw"] = rating
                if reviews: out["Expedia_N"] = reviews

def fetch_data(name, city, state, debug_mode):
    # [FIX] STRING CONVERSION: Ensures we never process a float/NaN as text
    s_name = str(name)
    s_city = str(city)
    s_state = str(state)

    out = {
        "Name": s_name, "City": s_city, "State": s_state, 
        "Google_Raw": None, "Google_N": None,
        "TA_Raw": None, "TA_N": None,
        "Expedia_Raw": None, "Expedia_N": None,
        "Booking_Raw": None, "Booking_N": None,
        "Review_Text_Raw": "",
        "Is_Verified": False,
        "_resolved_name": s_name,
        "Debug": ""
    }
    debug_notes = []
    snippet_collection = []

    clean_name = s_name.replace("[", "").replace("]", "").strip()
    
    # [FIX] SMART SEARCH: IGNORE "Unknown" STATE & CITY
    loc_str = ""
    if "Pending" not in s_city and s_city != "Unknown": loc_str += s_city + " "
    if "Pending" not in s_state and s_state != "Unknown": loc_str += s_state
    
    loc_str = loc_str.strip()
    query_1 = f"{clean_name} {loc_str} hotel reviews".strip()

    try:
        search = GoogleSearch({"engine": "google", "q": query_1, "api_key": SERPAPI_KEY, "num": 10, "gl": "us", "hl": "en"})
        results = search.get_dict()
        
        # --- FALLBACK LOGIC: If no KG found, try broader search ---
        if "knowledge_graph" not in results and "organic_results" in results:
             # Check if top organic result is generic or missing rating
             top_res = results["organic_results"][0] if results["organic_results"] else {}
             if not top_res.get("rating") and not top_res.get("rich_snippet"):
                 # TRIGGER ATTEMPT 2: BROAD SEARCH
                 debug_notes.append("Retry: Broad Search")
                 query_2 = f"{clean_name} google reviews"
                 search_2 = GoogleSearch({"engine": "google", "q": query_2, "api_key": SERPAPI_KEY, "num": 10, "gl": "us", "hl": "en"})
                 results = search_2.get_dict()

        if "knowledge_graph" in results:
            kg = results["knowledge_graph"]
            out["Is_Verified"] = True
            out["Google_Raw"] = kg.get("rating")
            out["Google_N"] = kg.get("reviews") or kg.get("review_count")
            
            # SMART BACKFILL from KG
            if kg.get("title"): 
                out["Name"] = kg.get("title")
                out["_resolved_name"] = kg.get("title")
            
            addr = kg.get("address", "")
            if isinstance(addr, str) and "," in addr:
                parts = [p.strip() for p in addr.split(",") if p.strip()]
                # Standard Google Address: "Street, City, State Zip"
                if len(parts) >= 2: 
                    state_zip = parts[-1].split(" ")
                    if len(state_zip) > 0:
                        out["State"] = state_zip[0]
                    out["City"] = parts[-2]

            scavenge_knowledge_graph(kg, out, debug_notes)
        
        scan_organic_results(results.get("organic_results", []), out, debug_notes, snippet_collection)
    except Exception as e: debug_notes.append(f"Err S1: {str(e)}")

    # SEARCH 2 & 3: Sniper
    search_name = out["_resolved_name"]
    # Re-apply Smart Search logic for sniper
    loc_str_snip = f"{out['City']} {out['State']}".replace("Pending", "").replace("Unknown", "").strip()
    
    targets = [("Booking", "booking.com", "Booking_Raw"), ("Expedia", "expedia.com", "Expedia_Raw"), ("TA", "tripadvisor.com", "TA_Raw")]
    for ota_name, domain, key in targets:
        if not out[key]:
            q_sniper = f"{search_name} {loc_str_snip} reviews {domain}".strip()
            try:
                search_t = GoogleSearch({"engine": "google", "q": q_sniper, "api_key": SERPAPI_KEY, "num": 5})
                res_t = search_t.get_dict()
                scan_organic_results(res_t.get("organic_results", []), out, debug_notes, snippet_collection)
            except: pass

    if snippet_collection:
        unique_snips = list(set(snippet_collection))
        out["Review_Text_Raw"] = " || ".join(unique_snips[:20])

    if debug_mode: out["Debug"] = "; ".join(debug_notes)
    return out

def calculate_metrics(row):
    row["Google_10"] = normalize_to_10(row["Google_Raw"], 5)
    row["TA_10"] = normalize_to_10(row["TA_Raw"], 5)
    for k in ["Booking", "Expedia"]:
        # [FIX] Safe float conversion
        v = safe_float(row[f"{k}_Raw"])
        if v > 0:
            row[f"{k}_10"] = v if v > 5 else normalize_to_10(v, 5)
        else: row[f"{k}_10"] = None

    total_score = 0
    total_count = 0
    platforms = [("Google", 5), ("TA", 5), ("Expedia", 5), ("Booking", 10)] 
    
    for plt, _ in platforms:
        score_10 = safe_float(row.get(f"{plt}_10"))
        count = safe_int(row.get(f"{plt}_N"))
        
        if score_10 > 0 and count > 0:
            total_score += (score_10 * count)
            total_count += count
            
    if total_count > 0:
        row["Weighted_Avg"] = round(total_score / total_count, 1)
    else: row["Weighted_Avg"] = None
    return row

def calculate_portfolio_averages(df):
    # [FIX] Complete Rewrite for Null Safety - Fixes TypeError crash
    if df.empty: return None
    summary = {"Name": "Total / Weighted Average", "City": "-", "State": "-", "Last Scraped": datetime.now().strftime("%Y-%m-%d")}
    platforms = ["Google", "Expedia", "TripAdvisor", "Booking"]
    
    for plt in platforms:
        col_name = f"# {plt} Reviews"
        if col_name in df.columns:
            # Use safe_int to handle any NaNs in the column
            total = df[col_name].apply(safe_int).sum()
            summary[col_name] = int(total)
    
    for plt in platforms:
        score_col = f"{plt} Score"
        norm_col = f"{plt} Score (Normalized)"
        count_col = f"# {plt} Reviews"
        
        if score_col in df.columns and count_col in df.columns:
            weighted_sum = 0
            total_count = 0
            for _, row in df.iterrows():
                s = safe_float(row.get(score_col))
                c = safe_int(row.get(count_col))
                if s > 0 and c > 0:
                    weighted_sum += (s * c)
                    total_count += c
            summary[score_col] = round(weighted_sum / total_count, 1) if total_count > 0 else None

        if norm_col in df.columns and count_col in df.columns:
            weighted_sum = 0
            total_count = 0
            for _, row in df.iterrows():
                s = safe_float(row.get(norm_col))
                c = safe_int(row.get(count_col))
                if s > 0 and c > 0:
                    weighted_sum += (s * c)
                    total_count += c
            summary[norm_col] = round(weighted_sum / total_count, 1) if total_count > 0 else None

    if "Weighted Avg. Review Score" in df.columns:
        weighted_sum = 0
        total_revs = 0
        for _, row in df.iterrows():
            row_revs = 0
            for plt in platforms: 
                # [FIX] safe_int here prevents the "int + NoneType" crash
                row_revs += safe_int(row.get(f"# {plt} Reviews"))
            
            w_avg = safe_float(row.get("Weighted Avg. Review Score"))
            if w_avg > 0 and row_revs > 0:
                weighted_sum += (w_avg * row_revs)
                total_revs += row_revs
        summary["Weighted Avg. Review Score"] = round(weighted_sum / total_revs, 1) if total_revs > 0 else None

    return summary

# ---------------------------
# 5) AI ANALYSIS FUNCTION (FACT-BASED, NO MATH)
# ---------------------------
def generate_portfolio_summary(data_df):
    if not OPENAI_API_KEY:
        return "⚠️ API Key Missing. Please add OPENAI_API_KEY to your Secrets."
    
    if data_df.empty:
        return "No data available to analyze."

    # --- SINGLE SOURCE OF TRUTH: CALCULATE METRICS IN PYTHON ---
    internal_vol_cols = ['Google_N', 'Expedia_N', 'TA_N', 'Booking_N']
    for vc in internal_vol_cols:
        if vc not in data_df.columns: data_df[vc] = 0
        else: data_df[vc] = data_df[vc].apply(safe_int)
            
    total_reviews_all = int(data_df[internal_vol_cols].sum().sum())
    hotel_count = len(data_df)
    
    w_sum = 0
    w_vol = 0
    if "Weighted_Avg" in data_df.columns:
        for _, r in data_df.iterrows():
            row_vol = sum(safe_int(r.get(vc)) for vc in internal_vol_cols)
            score = safe_float(r.get("Weighted_Avg"))
            if score > 0 and row_vol > 0:
                w_sum += (score * row_vol)
                w_vol += row_vol
    overall_score = round(w_sum / w_vol, 1) if w_vol > 0 else 0

    # PERCENTILES
    p90 = round(data_df['Weighted_Avg'].quantile(0.9), 1) if 'Weighted_Avg' in data_df.columns else 0
    p50 = round(data_df['Weighted_Avg'].median(), 1) if 'Weighted_Avg' in data_df.columns else 0

    # PLATFORM BREAKDOWN (PRE-CALC)
    plat_stats = []
    for p, n in [('Google_10', 'Google_N'), ('Expedia_10', 'Expedia_N'), ('TA_10', 'TA_N'), ('Booking_10', 'Booking_N')]:
        if p in data_df.columns:
            valid = data_df[data_df[p].notnull()]
            if not valid.empty:
                avg = valid[p].mean()
                plat_stats.append(f"{p.split('_')[0]}: {avg:.1f}/10")

    # GEO BREAKDOWN (PRE-CALC)
    geo_stats = "N/A"
    if 'State' in data_df.columns:
        try:
            best_state = data_df.groupby('State')['Weighted_Avg'].mean().idxmax()
            worst_state = data_df.groupby('State')['Weighted_Avg'].mean().idxmin()
            geo_stats = f"Best Performing Region: {best_state}, Lowest Performing: {worst_state}"
        except: geo_stats = "Insufficient Data"

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    prompt = f"""
    SYSTEM ROLE:
    You are an expert Hotel Asset Manager. 
    You do NOT calculate numbers. You only interpret the facts provided below.

    ---
    
    **FACT SHEET (DO NOT RE-CALCULATE):**
    * **Total Hotels:** {hotel_count}
    * **Total Portfolio Reviews:** {total_reviews_all:,}
    * **Overall Weighted Average Score:** {overall_score}/10
    * **Median Score (50th Percentile):** {p50}/10
    * **Top 10% Threshold (90th Percentile):** {p90}/10
    * **Platform Averages:** {', '.join(plat_stats)}
    * **Geographic Trends:** {geo_stats}
    
    ---

    OBJECTIVE:
    Provide a professional, executive-level summary based ONLY on the facts above.

    OUTPUT FORMAT (STRICT):

    1. Portfolio Overview
    [One sentence summarizing the total hotels, reviews, and overall weighted score.]

    2. Strategic Insights
    [4-5 bullet points interpreting the facts.]
    * **Platform Performance:** Mention which platforms are driving the score based on the Platform Averages provided.
    * **Geographic Variance:** Mention the Best/Lowest regions provided in the facts.
    * **Benchmarking:** Compare the Median ({p50}) to the Average ({overall_score}).
    * **Top Tier Performance:** State that assets must score above {p90} to reach the top 10%.

    CONSTRAINTS:
    * Do NOT hallucinate numbers. 
    * Do NOT calculate new averages. Use the ones provided.
    * Do NOT use emojis.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# ---------------------------
# 6) MAIN APPLICATION FLOW
# ---------------------------

if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'user_email' not in st.session_state: st.session_state['user_email'] = None
if 'first_name' not in st.session_state: st.session_state['first_name'] = None
if 'working_df' not in st.session_state: st.session_state['working_df'] = pd.DataFrame(columns=['Name', 'City', 'State', 'URL', 'Hotel_ID', 'Is_Verified'])
if 'confirm_reset' not in st.session_state: st.session_state['confirm_reset'] = False
if 'active_portfolio_name' not in st.session_state: st.session_state['active_portfolio_name'] = None
if 'original_portfolio_ids' not in st.session_state: st.session_state['original_portfolio_ids'] = []
if 'show_portfolio_dropdown' not in st.session_state: st.session_state['show_portfolio_dropdown'] = False
if 'show_admin' not in st.session_state: st.session_state['show_admin'] = False
if 'hidden_history_ids' not in st.session_state: st.session_state['hidden_history_ids'] = set()
if 'confirm_delete' not in st.session_state: st.session_state['confirm_delete'] = False
if 'ai_result' not in st.session_state: st.session_state['ai_result'] = None 

check_history_schema()

export_summary_list = []
export_hotel_data = pd.DataFrame()

if not st.session_state['logged_in']:
    c_login_main = st.columns([1, 2])[0]
    with c_login_main:
        st.title("Reputation Dashboard")
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Create Account"])
        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Login", use_container_width=True):
                    fname = check_login(email, password)
                    if fname:
                        st.session_state['logged_in'] = True
                        st.session_state['user_email'] = email
                        st.session_state['first_name'] = fname
                        st.rerun()
                    else: st.error("Invalid credentials")
        with tab2:
            first_name = st.text_input("First Name")
            last_name = st.text_input("Last Name")
            new_email = st.text_input("Email")
            new_pass = st.text_input("Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            if st.button("Create Account", use_container_width=True):
                if new_pass != confirm_pass: st.error("Passwords do not match!")
                elif not first_name or not last_name or not new_email: st.error("Please fill in all fields.")
                else:
                    if save_user(new_email, new_pass, first_name, last_name): st.success("Account created! Please login.")
                    else: st.error("Email already exists.")
else:
    with st.sidebar:
        st.markdown("### ☰ MENU")
        if st.button("Start New Analysis"):
            st.session_state['working_df'] = pd.DataFrame(columns=['Name', 'City', 'State', 'URL', 'Hotel_ID', 'Is_Verified'])
            st.session_state['active_portfolio_name'] = None
            st.session_state['original_portfolio_ids'] = []
            st.session_state['show_portfolio_dropdown'] = False
            st.session_state['hidden_history_ids'] = set()
            st.session_state['confirm_delete'] = False
            st.session_state['ai_result'] = None
            st.session_state.pop('final_results', None)
            st.rerun()
        
        if st.button("Load Saved Portfolio"):
            st.session_state['show_portfolio_dropdown'] = not st.session_state['show_portfolio_dropdown']
            st.session_state['confirm_delete'] = False 
        
        if st.session_state['show_portfolio_dropdown']:
            all_portfolios = load_portfolios()
            user_portfolios = all_portfolios.get(st.session_state['user_email'], {})
            portfolio_names = list(user_portfolios.keys())
            selected_portfolio = st.selectbox("Select Group", ["Select..."] + portfolio_names, label_visibility="collapsed")
            
            if selected_portfolio != "Select...":
                c_load, c_del = st.columns([3, 1])
                if c_load.button("Load Portfolio", type="primary", use_container_width=True):
                    hotel_ids = user_portfolios[selected_portfolio]
                    cache = load_cache()
                    loaded_rows = []
                    for hid in hotel_ids:
                        match = cache[cache['Hotel_ID'] == hid]
                        if not match.empty:
                            row = match.iloc[0][['Name', 'City', 'State', 'Hotel_ID', 'Is_Verified']].to_dict()
                            loaded_rows.append(row)
                        else:
                            parts = hid.split('_')
                            if len(parts) >= 3: loaded_rows.append({'Name': parts[0], 'City': parts[1], 'State': parts[2], 'Hotel_ID': hid, 'Is_Verified': False})
                            elif len(parts) == 2: loaded_rows.append({'Name': parts[0], 'City': parts[1], 'State': 'Pending', 'Hotel_ID': hid, 'Is_Verified': False})
                    
                    st.session_state['working_df'] = pd.DataFrame(loaded_rows)
                    st.session_state['active_portfolio_name'] = selected_portfolio
                    st.session_state['original_portfolio_ids'] = hotel_ids 
                    st.session_state['show_portfolio_dropdown'] = False
                    st.session_state['hidden_history_ids'] = set()
                    st.session_state['confirm_delete'] = False
                    st.session_state['ai_result'] = None
                    log_activity(st.session_state['user_email'], "Portfolio Loaded", f"Name: {selected_portfolio}")
                    st.rerun()

                if c_del.button("🗑️", type="secondary", use_container_width=True):
                    st.session_state['confirm_delete'] = True
                
                if st.session_state.get('confirm_delete'):
                    st.warning(f"Are you sure you want to delete {selected_portfolio}?")
                    c_yes, c_no = st.columns(2)
                    if c_yes.button("Yes, Delete", type="primary", use_container_width=True):
                        delete_portfolio(st.session_state['user_email'], selected_portfolio)
                        st.success("Deleted!")
                        st.session_state['confirm_delete'] = False
                        st.session_state['active_portfolio_name'] = None
                        time.sleep(1)
                        st.rerun()
                    if c_no.button("No, Don't Delete", type="secondary", use_container_width=True):
                        st.session_state['confirm_delete'] = False
                        st.rerun()

        if st.button("Admin Controls"): st.session_state['show_admin'] = not st.session_state['show_admin']
        if st.session_state['show_admin']:
            admin_password = st.text_input("Password", type="password")
            if admin_password == "admin" or admin_password == "masteradmin":
                if admin_password == "masteradmin":
                    st.success("🔓 Access Granted")
                    st.markdown("### Activity Log")
                    if os.path.exists(ACTIVITY_FILE):
                        act_df = pd.read_csv(ACTIVITY_FILE)
                        st.dataframe(act_df.sort_values(by="timestamp", ascending=False), height=150, use_container_width=True, hide_index=True)
                
                st.markdown("""To minimize API usage and prevent hitting API rate limits, data is stored locally
for 14 days. Warning: Purging this cache will force a fresh scrape for every
property on your list, which will consume new API credits.""")
                if not st.session_state['confirm_reset']:
                    if st.button("Purge Cache Database", type="primary", use_container_width=True):
                        st.session_state['confirm_reset'] = True
                        st.rerun()
                else:
                    st.markdown("**Are you sure?**")
                    c1, c2 = st.columns(2)
                    if c1.button("✅ Yes, I am Sure", type="primary", use_container_width=True):
                        if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
                        st.session_state['confirm_reset'] = False
                        st.success("Cache Cleared.")
                        time.sleep(1)
                        st.rerun()
                    if c2.button("❌ No, Do Not Purge", type="secondary", use_container_width=True):
                        st.session_state['confirm_reset'] = False
                        st.rerun()

        if st.button("Logout"):
            log_activity(st.session_state['user_email'], "User Logout")
            st.session_state['logged_in'] = False
            st.session_state['user_email'] = None
            st.rerun()

    st.title("Hotel Reputation Dashboard")
    tab_live, tab_hist = st.tabs(["📊 Live Dashboard", "📈 History & Trends"])

    # --- TAB 1: LIVE DASHBOARD ---
    with tab_live:
        col_upload, col_input = st.columns([1, 1], gap="large")
        
        # [FIX] COMPREHENSIVE DATA AUDIT & SANITIZATION
        def process_file_upload():
            uploaded_file = st.session_state.get('file_uploader')
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        header_idx = find_header_row(uploaded_file, 'csv')
                        df = pd.read_csv(uploaded_file, header=header_idx)
                    else:
                        header_idx = find_header_row(uploaded_file, 'xlsx')
                        df = pd.read_excel(uploaded_file, header=header_idx)
                    
                    df.columns = df.columns.str.strip().str.title()
                    
                    # 1. Sanitize: Fill blanks with 'Unknown'
                    if 'Name' not in df.columns:
                        st.error("File missing 'Name' column.")
                        return
                    
                    if 'City' not in df.columns: df['City'] = 'Unknown'
                    if 'State' not in df.columns: df['State'] = 'Unknown'
                    
                    df['Name'] = df['Name'].fillna('Unknown').astype(str)
                    df['City'] = df['City'].fillna('Unknown').astype(str)
                    df['State'] = df['State'].fillna('Unknown').astype(str)
                    
                    # 2. Audit: Check for specific failures
                    warnings = []
                    
                    # A. Fatal: Missing Name (Skip these rows)
                    missing_name_mask = (df['Name'] == 'Unknown') | (df['Name'].str.strip() == '')
                    skipped_count = missing_name_mask.sum()
                    if skipped_count > 0:
                        warnings.append(f"Skipped {skipped_count} rows missing 'Name' data.")
                        df = df[~missing_name_mask].copy() # Filter them out
                    
                    if df.empty:
                        st.warning("⚠️ No valid rows found after filtering missing names.")
                        return

                    # B. Warning: Missing City or State
                    missing_city = df[df['City'] == 'Unknown'].index.tolist()
                    missing_state = df[df['State'] == 'Unknown'].index.tolist()
                    
                    if missing_city:
                        rows_disp = [x + header_idx + 2 for x in missing_city][:5] # +2 accounts for 0-index and header
                        warnings.append(f"{len(missing_city)} rows missing 'City' (e.g. Row {rows_disp}...)")
                    
                    if missing_state:
                        rows_disp = [x + header_idx + 2 for x in missing_state][:5]
                        warnings.append(f"{len(missing_state)} rows missing 'State' (e.g. Row {rows_disp}...)")
                    
                    if warnings:
                        st.warning("⚠️ **Data Quality Report:** " + " | ".join(warnings))
                    
                    # 3. Construct ID and proceed
                    df['Hotel_ID'] = df['Name'] + "_" + df['City'] + "_" + df['State']
                    df['Is_Verified'] = False
                    cols = ['Name', 'City', 'State', 'Hotel_ID', 'Is_Verified']
                    
                    if 'Url' in df.columns: df['URL'] = df['Url']; cols.append('URL')
                    elif 'URL' in df.columns: cols.append('URL')
                    
                    clean_df = df[cols].copy()
                    st.session_state['working_df'] = pd.concat([st.session_state['working_df'], clean_df], ignore_index=True).drop_duplicates(subset=['Hotel_ID'])
                    
                    st.toast("✅ Upload Processed")
                except Exception as e: st.error(f"Error: {e}")

        with col_upload:
            st.markdown("**📁 Bulk Upload**")
            st.caption("⚠️ File Requirement: Your file must contain columns for **Name**, **City**, and **State**.")
            st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"], label_visibility="collapsed", key="file_uploader", on_change=process_file_upload)

        with col_input:
            st.markdown("**✍️ Manual Entry**")
            with st.form("manual"):
                c1, c2, c3 = st.columns(3)
                with c1: f_name = st.text_input("Name", placeholder="Name")
                with c2: f_city = st.text_input("City", placeholder="City")
                with c3: f_state = st.text_input("State", placeholder="TX")
                if st.form_submit_button("Add"):
                    # [FIX] STRICT MANUAL ENTRY: Require all 3 fields
                    if f_name and f_city and f_state:
                        st.session_state['working_df'] = pd.concat([st.session_state['working_df'], pd.DataFrame([{
                            'Name': f_name, 'City': f_city, 'State': f_state, 
                            'Hotel_ID': f"{f_name}_{f_city}_{f_state}", 'Is_Verified': False
                        }])], ignore_index=True)
                        st.toast("✅ Hotel added to queue.")
                        st.rerun()
                    else:
                        st.error("⚠️ Please fill in all 3 fields (Name, City, State).")

        if not st.session_state['working_df'].empty:
            st.markdown("---")
            has_results = 'final_results' in st.session_state
            queue_count = len(st.session_state['working_df'])
            if has_results: st.markdown(f"<div class='status-complete'><b>✅ Analysis Complete!</b><br>Scroll down to view the results table, filter data, and save your report.</div>", unsafe_allow_html=True)
            else: st.markdown(f"<div class='status-ready'><b>📋 Queue Ready: {queue_count} hotels pending.</b><br>Click 'Run Analysis' below to fetch data.</div>", unsafe_allow_html=True)
            st.caption("To prevent exceeding API rate-limits, data is cached for 14 days.")

            if st.button("Run Analysis", type="primary", use_container_width=True):
                log_activity(st.session_state['user_email'], "Analysis Run", f"{len(st.session_state['working_df'])} hotels")
                progress_bar = st.progress(0, text="Starting analysis...")
                cache_db = load_cache()
                results = []
                target_df = st.session_state['working_df']
                total_rows = len(target_df)

                for index, row in target_df.iterrows():
                    progress_bar.progress((index + 1) / total_rows, text=f"Processing hotel {index+1} of {total_rows}")
                    
                    # URL Handling: If row has a URL and ID matches URL, it's a raw entry
                    input_url = row.get('URL') if row.get('URL') and str(row.get('URL')).startswith('http') else None
                    
                    cached_row = cache_db[cache_db['Hotel_ID'] == row['Hotel_ID']]
                    if not cached_row.empty and not input_url: 
                        last = cached_row.iloc[0]['Last_Scraped']
                        if is_cache_fresh(last): result_row = cached_row.iloc[0].to_dict()
                        else:
                            data = fetch_data(row["Name"], row["City"], row["State"], False)
                            data["Hotel_ID"] = row["Hotel_ID"]; data["Last_Scraped"] = datetime.now().strftime("%Y-%m-%d")
                            save_to_cache(data); result_row = data
                    else:
                        data = fetch_data(row["Name"], row["City"], row["State"], False)
                        # If we found a real name, update ID to prevent re-searching URL next time
                        if data['Is_Verified'] and input_url:
                            data["Hotel_ID"] = f"{data['Name']}_{data['City']}_{data['State']}"
                        else:
                            data["Hotel_ID"] = row["Hotel_ID"]
                        
                        data["Last_Scraped"] = datetime.now().strftime("%Y-%m-%d")
                        save_to_cache(data); result_row = data
                    
                    st.session_state['working_df'].at[index, 'Name'] = result_row['Name']
                    st.session_state['working_df'].at[index, 'City'] = result_row['City']
                    st.session_state['working_df'].at[index, 'State'] = result_row['State']
                    st.session_state['working_df'].at[index, 'Is_Verified'] = result_row['Is_Verified']
                    st.session_state['working_df'].at[index, 'Hotel_ID'] = result_row['Hotel_ID']
                    
                    result_row = calculate_metrics(result_row)
                    results.append(result_row)
                
                st.session_state['final_results'] = pd.DataFrame(results)
                st.rerun()

            # --- AUTO-MINIMIZE QUEUE ---
            # Collapse if results exist, otherwise expand
            expander_state = not has_results
            
            with st.expander("Analysis Queue (Edit)", expanded=expander_state):
                failed_ids = set()
                if 'final_results' in st.session_state:
                    bad = st.session_state['final_results'][(st.session_state['final_results']['Google_Raw'].isna())]
                    failed_ids = set(bad['Hotel_ID'].tolist())
                
                for index, row in st.session_state['working_df'].iterrows():
                    c1, c2, c3 = st.columns([4, 3, 1])
                    if row['Hotel_ID'] in failed_ids: c1.markdown(f"🔴 **{row['Name']}**", unsafe_allow_html=True)
                    else: c1.markdown(f"<b>{row['Name']}</b>", unsafe_allow_html=True)
                    c2.markdown(f"{row['City']}, {row['State']}", unsafe_allow_html=True)
                    if c3.button("✕", key=f"del_{index}"):
                            st.session_state['working_df'] = st.session_state['working_df'].drop(index).reset_index(drop=True)
                            st.rerun()
            
            # --- TROUBLESHOOTING BOX (BELOW MINIMIZED QUEUE) ---
            if 'failed_ids' in locals() and failed_ids:
                st.warning("⚠️ **Troubleshooting:** Hotels marked with 🔴 failed the primary Google score verification but can be manually updated in 'Verified Performance Data' section below (scroll down) by toggling 'Enable Editing'.")

        if 'final_results' in st.session_state:
            st.divider()
            st.subheader("Verified Performance Data")
            
            # --- DATA EDITOR IMPLEMENTATION ---
            enable_edit = st.toggle("✏️ Enable Editing (Manual Override)")
            
            res_df = st.session_state['final_results'].copy()
            
            # Helper to map display names back to internal names for saving
            internal_cols = ["Google_Raw", "Google_N", "Expedia_Raw", "Expedia_N", "TA_Raw", "TA_N", "Booking_Raw", "Booking_N"]

            if enable_edit:
                # Show editable grid for scores AND review counts
                st.caption("Edit values directly below. Press Enter to apply. (Scores 0-10, Counts must be integers)")
                edited_df = st.data_editor(
                    res_df[['Name', 'City', 'State', 
                            'Google_Raw', 'Google_N', 
                            'Expedia_Raw', 'Expedia_N', 
                            'TA_Raw', 'TA_N', 
                            'Booking_Raw', 'Booking_N']],
                    key="data_editor",
                    num_rows="fixed",
                    column_config={
                        "Google_Raw": st.column_config.NumberColumn("Google Score", min_value=0, max_value=5, step=0.1),
                        "Google_N": st.column_config.NumberColumn("Google Reviews", min_value=0, step=1),
                        "Expedia_Raw": st.column_config.NumberColumn("Expedia Score", min_value=0, max_value=10, step=0.1),
                        "Expedia_N": st.column_config.NumberColumn("Expedia Reviews", min_value=0, step=1),
                        "TA_Raw": st.column_config.NumberColumn("TripAdvisor Score", min_value=0, max_value=5, step=0.1),
                        "TA_N": st.column_config.NumberColumn("TA Reviews", min_value=0, step=1),
                        "Booking_Raw": st.column_config.NumberColumn("Booking Score", min_value=0, max_value=10, step=0.1),
                        "Booking_N": st.column_config.NumberColumn("Booking Reviews", min_value=0, step=1),
                    }
                )
                
                # Check for changes and trigger update
                if not edited_df.equals(res_df[['Name', 'City', 'State', 'Google_Raw', 'Google_N', 'Expedia_Raw', 'Expedia_N', 'TA_Raw', 'TA_N', 'Booking_Raw', 'Booking_N']]):
                    # Merge edits back into main dataframe
                    for idx in edited_df.index:
                        for col in internal_cols:
                            res_df.at[idx, col] = edited_df.at[idx, col]
                            
                    # Re-Calculate Metrics
                    updated_results = []
                    for _, row in res_df.iterrows():
                        updated_results.append(calculate_metrics(row))
                    
                    st.session_state['final_results'] = pd.DataFrame(updated_results)
                    # Update cache with manual overrides
                    for _, row in st.session_state['final_results'].iterrows():
                         save_to_cache(row.to_dict())
                         
                    st.rerun()
            
            # PREPARE DISPLAY DF
            final_df = st.session_state['final_results'].copy()
            rename_map = {
                "Google_Raw": "Google Score", "Expedia_Raw": "Expedia Score",
                "TA_Raw": "TripAdvisor Score", "Booking_Raw": "Booking Score",
                "Google_10": "Google Score (Normalized)", "Expedia_10": "Expedia Score (Normalized)",
                "TA_10": "TripAdvisor Score (Normalized)", "Booking_10": "Booking Score (Normalized)",
                "Google_N": "# Google Reviews", "Expedia_N": "# Expedia Reviews",
                "TA_N": "# TripAdvisor Reviews", "Booking_N": "# Booking Reviews",
                "Weighted_Avg": "Weighted Avg. Review Score", "Last_Scraped": "Last Scraped",
                "_resolved_name": "Matched Name (Source)"
            }
            final_df = final_df.rename(columns=rename_map)
            
            desired_order = [
                "Name", "City", "State", 
                "Google Score", "Expedia Score", "TripAdvisor Score", "Booking Score",
                "Google Score (Normalized)", "Expedia Score (Normalized)", "TripAdvisor Score (Normalized)", "Booking Score (Normalized)",
                "# Google Reviews", "# Expedia Reviews", "# TripAdvisor Reviews", "# Booking Reviews",
                "Weighted Avg. Review Score", "Last Scraped"
            ]
            for c in desired_order: 
                if c not in final_df.columns: final_df[c] = None
            final_df = final_df[desired_order]
            
            summary_row = calculate_portfolio_averages(final_df)
            if summary_row: final_df = pd.concat([final_df, pd.DataFrame([summary_row])], ignore_index=True)

            fmt = {}
            for c in final_df.columns:
                if "Score" in c: fmt[c] = "{:.1f}"
                if "#" in c and "Reviews" in c: fmt[c] = "{:,.0f}"

            def highlight_total_row(row):
                if row['Name'] == "Total / Weighted Average": return ['background-color: #e0e0e0; font-weight: bold; color: black'] * len(row)
                return [''] * len(row)

            # Display table (Non-editable view unless toggle is on)
            if not enable_edit:
                display_cols = [c for c in final_df.columns if c not in ["Google Score", "Expedia Score", "TripAdvisor Score", "Booking Score", "Matched Name (Source)"]]
                # [FIX] NULL-SAFE STYLING: na_rep="-" prevents crashes on empty cells
                st.dataframe(final_df[display_cols].style.format(fmt, na_rep="-").apply(highlight_total_row, axis=1), use_container_width=True, hide_index=True)

            # --- METHODOLOGY BOX (RESTORED HERE) ---
            with st.expander("Methodology & Assumptions", expanded=False):
                st.markdown("""
                <span class='meth-header'>Normalization:</span>
                <div class='meth-body'>
                • <b>Google & TripAdvisor:</b> Scores are on a 5-point scale. Scores are multiplied by 2 to convert to a 10-point scale.<br>
                • <b>Booking.com & Expedia:</b> Scores are generally on a 10-point scale. If a legacy 5-point score is detected, it is doubled.
                </div>
                <span class='meth-header'>Weighted Average Calculation:</span>
                <div class='meth-body'>
                • The Weighted Average Review Score is calculated by taking the sum of (Normalized Score × Review Count) for each platform, divided by the Total Review Count across all platforms.
                </div>
                <span class='meth-header'>Portfolio Totals (Bottom Row):</span>
                <div class='meth-body'>
                • <b>Review Counts:</b> Total sum of all reviews for that platform across the entire portfolio.<br>
                • <b>Scores:</b> Calculated as a weighted average based on review volume.
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""<div class='save-section'><div class='save-header'>Save Portfolio</div><div class='save-text'>Save this analysis to your account history.</div></div>""", unsafe_allow_html=True)
            
            # PREPARE RAW SAVE DATA
            raw_results_save = st.session_state['final_results'].rename(columns=rename_map)
            for c in ["# Google Reviews", "# Expedia Reviews", "# TripAdvisor Reviews", "# Booking Reviews"]:
                if c in raw_results_save.columns: raw_results_save[c] = raw_results_save[c].fillna(0).astype(int)

            current_ids = set(st.session_state['working_df']['Hotel_ID'].tolist())
            is_modified = current_ids != set(st.session_state.get('original_portfolio_ids', []))
            
            # --- SAVE & RENAME SECTION (COMPRESSED) ---
            if st.session_state['active_portfolio_name']:
                if not is_modified:
                    st.info(f"Active: **{st.session_state['active_portfolio_name']}**")
                    # Use columns to constrain width to ~1/3 (1, 1, 1)
                    c_btn1, c_btn2, c_buffer = st.columns([1, 1, 1])
                    
                    with c_btn1:
                        if st.button("📸 Save Snapshot", use_container_width=True):
                            log_history(st.session_state['user_email'], st.session_state['active_portfolio_name'], raw_results_save)
                            st.toast("Snapshot Saved!", icon="✅")
                    
                    with c_btn2:
                        with st.expander("Rename Portfolio"):
                            ren_name = st.text_input("New Name", value=st.session_state['active_portfolio_name'], label_visibility="collapsed")
                            if st.button("Rename", use_container_width=True):
                                if ren_name and ren_name != st.session_state['active_portfolio_name']:
                                    success = rename_portfolio(st.session_state['user_email'], st.session_state['active_portfolio_name'], ren_name)
                                    if success:
                                        st.session_state['active_portfolio_name'] = ren_name
                                        st.success("Renamed!")
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error("Name already exists.")
                else:
                    st.warning("Modified List")
                    c_mod1, c_mod2 = st.columns(2)
                    with c_mod1:
                        if st.button(f"Update", use_container_width=True):
                            save_portfolio(st.session_state['user_email'], st.session_state['active_portfolio_name'], list(current_ids))
                            st.session_state['original_portfolio_ids'] = list(current_ids)
                            log_history(st.session_state['user_email'], st.session_state['active_portfolio_name'], raw_results_save)
                            st.rerun()
                    with c_mod2:
                        new_name = st.text_input("New Name", label_visibility="collapsed")
                        if st.button("Save New", use_container_width=True):
                            if new_name:
                                save_portfolio(st.session_state['user_email'], new_name, list(current_ids))
                                st.session_state['original_portfolio_ids'] = list(current_ids)
                                st.session_state['active_portfolio_name'] = new_name
                                log_history(st.session_state['user_email'], new_name, raw_results_save)
                                st.rerun()
            else:
                save_name = st.text_input("Portfolio Name", placeholder="e.g. NYC Competitive Set", label_visibility="collapsed")
                c_save_btn, c_buff = st.columns([1, 2])
                with c_save_btn:
                    if st.button("💾 Save", use_container_width=True):
                        if save_name:
                            save_portfolio(st.session_state['user_email'], save_name, list(current_ids))
                            st.session_state['original_portfolio_ids'] = list(current_ids)
                            st.session_state['active_portfolio_name'] = save_name
                            log_history(st.session_state['user_email'], save_name, raw_results_save)
                            st.rerun()

    with tab_hist:
        st.subheader("Current Snapshot Analysis")
        if 'final_results' in st.session_state:
            curr_df = st.session_state['final_results'].copy()
            curr_calc = curr_df[curr_df['Name'] != "Total / Weighted Average"]
            
            # HERO METRICS
            w_sum = 0
            total_vol = 0
            for _, r in curr_calc.iterrows():
                revs = (safe_int(r.get('Google_N')) + safe_int(r.get('Expedia_N')) + 
                        safe_int(r.get('TA_N')) + safe_int(r.get('Booking_N')))
                w_avg = safe_float(r.get('Weighted_Avg'))
                if w_avg > 0 and revs > 0:
                    w_sum += (w_avg * revs)
                    total_vol += int(revs)
            port_score = round(w_sum / total_vol, 1) if total_vol > 0 else 0
            
            c_h1, c_h2, c_h3, c_h4 = st.columns(4)
            with c_h1: st.markdown(f"<div class='metric-box'><div class='metric-title'>Portfolio Weighted Avg. Score</div><div class='metric-value'>{port_score}</div></div>", unsafe_allow_html=True)
            with c_h2: st.markdown(f"<div class='metric-box'><div class='metric-title'>Total Reviews</div><div class='metric-value'>{total_vol:,}</div></div>", unsafe_allow_html=True)
            with c_h3: st.markdown(f"<div class='metric-box'><div class='metric-title'>Hotel Count</div><div class='metric-value'>{len(curr_calc)}</div></div>", unsafe_allow_html=True)
            with c_h4: st.markdown(f"<div class='metric-box'><div class='metric-title'>Market Coverage</div><div class='metric-value'>{curr_calc['City'].nunique()} Cities</div></div>", unsafe_allow_html=True)
            st.write("")

            # EXPORT PREP - HERO
            export_summary_list.append(pd.DataFrame({'Metric': ['Score', 'Reviews', 'Hotels'], 'Value': [port_score, total_vol, len(curr_calc)]}))

            # PLATFORM PERFORMANCE CHART
            st.markdown("##### Platform Performance")
            def calc_plat_stats(key_norm, key_n):
                p_w_sum = 0; p_vol = 0
                for _, r in curr_calc.iterrows():
                    s = safe_float(r.get(key_norm))
                    c = safe_int(r.get(key_n))
                    if s > 0 and c > 0:
                        p_w_sum += (s * c)
                        p_vol += c
                return (round(p_w_sum / p_vol, 1) if p_vol > 0 else 0), int(p_vol)
            
            g_s, g_v = calc_plat_stats('Google_10', 'Google_N')
            e_s, e_v = calc_plat_stats('Expedia_10', 'Expedia_N')
            t_s, t_v = calc_plat_stats('TA_10', 'TA_N')
            b_s, b_v = calc_plat_stats('Booking_10', 'Booking_N')

            plat_data = pd.DataFrame({
                'Platform': ['Google', 'Expedia', 'TripAdvisor', 'Booking'],
                'Score': [g_s, e_s, t_s, b_s],
                'Reviews': [g_v, e_v, t_v, b_v],
                'Color': ['#DB4437', '#F4B400', '#00AA6C', '#003580'],
                'Label_Score': [str(g_s), str(e_s), str(t_s), str(b_s)],
                'Label_Reviews': [f"({g_v:,} reviews)", f"({e_v:,} reviews)", f"({t_v:,} reviews)", f"({b_v:,} reviews)"]
            })
            
            export_summary_list.append(plat_data[['Platform', 'Score', 'Reviews']])

            base = alt.Chart(plat_data).encode(
                x=alt.X('Platform', sort=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Score', scale=alt.Scale(domain=[0, 10]), title="Weighted Avg. Review Score"),
                tooltip=['Platform', 'Score', 'Reviews']
            )
            bars = base.mark_bar().encode(color=alt.Color('Color', scale=None))
            text_score = base.mark_text(dy=15, color='white', fontSize=18, fontWeight='bold').encode(text='Label_Score')
            text_revs = base.mark_text(dy=35, color='white', fontSize=12, fontWeight='normal').encode(text='Label_Reviews')
            st.altair_chart(bars + text_score + text_revs, use_container_width=True)
            st.write("")

            # OUTLIERS
            st.markdown("##### Performance Outliers")
            c_best, c_worst = st.columns(2)
            sorted_df = curr_calc.sort_values(by="Weighted_Avg", ascending=False)
            with c_best:
                st.caption("🏆 Top 5 Performing Hotels")
                for _, r in sorted_df.head(5).iterrows(): st.markdown(f"<div class='performer-card'><div class='p-name'>{r['Name']}<span class='p-score'>{r['Weighted_Avg']}</span></div><div class='p-loc'>{r['City']}, {r['State']}</div></div>", unsafe_allow_html=True)
            with c_worst:
                st.caption("⚠️ Bottom 5 Performing Hotels")
                for _, r in sorted_df.tail(5).sort_values(by="Weighted_Avg", ascending=True).iterrows(): st.markdown(f"<div class='performer-card'><div class='p-name'>{r['Name']}<span class='p-score'>{r['Weighted_Avg']}</span></div><div class='p-loc'>{r['City']}, {r['State']}</div></div>", unsafe_allow_html=True)
            st.write("")

            # BENCHMARKS TABLE
            st.markdown("##### Percentile Benchmarks")
            def get_percentile(col_norm, quantile):
                valid = curr_calc[curr_calc[col_norm].notnull()][col_norm]
                if valid.empty: return "-"
                return round(valid.quantile(quantile), 1)
            bench_df = pd.DataFrame({
                "Metric": ["Portfolio Weighted Avg", "Google Score", "Expedia Score", "TripAdvisor Score", "Booking Score"],
                "Top 50%": [get_percentile('Weighted_Avg', 0.5), get_percentile('Google_10', 0.5), get_percentile('Expedia_10', 0.5), get_percentile('TA_10', 0.5), get_percentile('Booking_10', 0.5)],
                "Top 25%": [get_percentile('Weighted_Avg', 0.75), get_percentile('Google_10', 0.75), get_percentile('Expedia_10', 0.75), get_percentile('TA_10', 0.75), get_percentile('Booking_10', 0.75)],
                "Top 10%": [get_percentile('Weighted_Avg', 0.90), get_percentile('Google_10', 0.90), get_percentile('Expedia_10', 0.90), get_percentile('TA_10', 0.90), get_percentile('Booking_10', 0.90)],
                "Top 5%": [get_percentile('Weighted_Avg', 0.95), get_percentile('Google_10', 0.95), get_percentile('Expedia_10', 0.95), get_percentile('TA_10', 0.95), get_percentile('Booking_10', 0.95)],
                "Top 1%": [get_percentile('Weighted_Avg', 0.99), get_percentile('Google_10', 0.99), get_percentile('Expedia_10', 0.99), get_percentile('TA_10', 0.99), get_percentile('Booking_10', 0.99)]
            })
            st.dataframe(bench_df.style.format({c: "{:.1f}" for c in bench_df.columns if c != "Metric"}), use_container_width=True, hide_index=True)
            
            # EXPORT PREP - BENCHMARKS
            export_summary_list.append(bench_df)

            # CALCULATOR
            with st.expander("🧮 Percentile Calculator", expanded=False):
                st.caption("Instructions: Input review scores for a specific property to see how it compares against the dataset. Scores should be inputted on a standardized 10-point scale (e.g., a Google Rating of 4.5 / 5.0 should be inputted as 9.0).")
                cc1, cc2, cc3, cc4 = st.columns(4)
                with cc1: inp_g = st.number_input("Google (0-10)", 0.0, 10.0, 0.0, 0.1, key='calc_g')
                with cc2: inp_e = st.number_input("Expedia (0-10)", 0.0, 10.0, 0.0, 0.1, key='calc_e')
                with cc3: inp_t = st.number_input("TripAdvisor (0-10)", 0.0, 10.0, 0.0, 0.1, key='calc_t')
                with cc4: inp_b = st.number_input("Booking (0-10)", 0.0, 10.0, 0.0, 0.1, key='calc_b')
                if st.button("Calculate Rank"):
                    def calculate_percentile(data, value):
                        if len(data) == 0: return 0
                        count = sum(1 for x in data if x <= value)
                        return (count / len(data)) * 100
                    def calc_rank(val, col):
                        if val == 0: return "N/A"
                        valid = curr_calc[curr_calc[col].notnull()][col].tolist()
                        if not valid: return "No Data"
                        pct = calculate_percentile(valid, val)
                        return int(100 - pct) if (100 - pct) >= 1 else 1
                    r_g = calc_rank(inp_g, 'Google_10'); r_e = calc_rank(inp_e, 'Expedia_10'); r_t = calc_rank(inp_t, 'TA_10'); r_b = calc_rank(inp_b, 'Booking_10')
                    c_r1, c_r2, c_r3, c_r4 = st.columns(4)
                    def render_result(col, label, rank):
                        if rank == "N/A" or rank == "No Data": col.metric(label, "-")
                        else: col.markdown(f"**{label}**<br><span style='font-size:1.2em; color:#ffd700'>Top {rank}%</span><br><span style='font-size:0.8em; color:#bbb'>Your score ranks in the top {rank}% of this dataset.</span>", unsafe_allow_html=True)
                    render_result(c_r1, "Google Rank", r_g); render_result(c_r2, "Expedia Rank", r_e); render_result(c_r3, "TA Rank", r_t); render_result(c_r4, "Booking Rank", r_b)
            st.write("")

            # SCATTER PLOT
            st.markdown("##### Review Volume vs. Quality")
            curr_calc['Total_Reviews'] = curr_calc['Google_N'].fillna(0) + curr_calc['Expedia_N'].fillna(0) + curr_calc['TA_N'].fillna(0) + curr_calc['Booking_N'].fillna(0)
            scatter = alt.Chart(curr_calc).mark_circle(size=60).encode(
                x=alt.X('Total_Reviews', title='Total Reviews', scale=alt.Scale(domainMin=0)),
                y=alt.Y('Weighted_Avg', scale=alt.Scale(domain=[0, 10]), title='Weighted Score'),
                tooltip=['Name', 'City', 'Weighted_Avg', 'Total_Reviews']
            ).interactive()
            st.altair_chart(scatter, use_container_width=True)
            st.write("")

            # GEO BREAKDOWN
            st.markdown("##### Geographic Breakdown (Top 10 by Score)")
            group_by = st.radio("Group by:", ["State", "City"], horizontal=True)
            geo_rows = []
            for loc, group in curr_calc.groupby([group_by]):
                state_w_sum = 0; state_vol = 0
                for _, r in group.iterrows():
                    revs = (safe_int(r.get('Google_N')) + safe_int(r.get('Expedia_N')) + safe_int(r.get('TA_N')) + safe_int(r.get('Booking_N')))
                    w_avg = safe_float(r.get('Weighted_Avg'))
                    if w_avg > 0 and revs > 0: state_w_sum += (w_avg * revs); state_vol += revs
                geo_score = round(state_w_sum / state_vol, 1) if state_vol > 0 else 0
                clean_loc = str(loc).replace("('", "").replace("',)", "").replace("[", "").replace("]", "")
                geo_rows.append({group_by: clean_loc, "Hotels": len(group), "Reviews": state_vol, "Score": geo_score})
            
            geo_df = pd.DataFrame(geo_rows).sort_values(by="Score", ascending=False)
            
            if len(geo_df) > 10:
                top_10 = geo_df.iloc[:10]
                others = geo_df.iloc[10:]
                other_vol = others["Reviews"].sum()
                other_score = round((others["Score"] * others["Reviews"]).sum() / other_vol, 1) if other_vol > 0 else 0
                other_row = {group_by: "All Others", "Hotels": others["Hotels"].sum(), "Reviews": other_vol, "Score": other_score}
                geo_df = pd.concat([top_10, pd.DataFrame([other_row])], ignore_index=True)

            geo_df['Label_Hotels'] = "Hotels: " + geo_df['Hotels'].astype(str)
            geo_df['Label_Reviews'] = "Reviews: " + geo_df['Reviews'].apply(lambda x: f"{x:,.0f}")

            sort_order = geo_df[group_by].tolist()

            base_geo = alt.Chart(geo_df).encode(
                x=alt.X(group_by, sort=sort_order, axis=alt.Axis(labelAngle=0), scale=alt.Scale(padding=0.3), title=group_by), 
                y=alt.Y('Score', scale=alt.Scale(domain=[0, 10]), title="Weighted Avg Score"),
                tooltip=[group_by, 'Score', 'Reviews', 'Hotels']
            )
            bars_geo = base_geo.mark_bar(color="#1f77b4").encode() 
            text_geo_score = base_geo.mark_text(dy=15, color='white', fontSize=14, fontWeight='bold').encode(text=alt.Text('Score', format='.1f'))
            text_geo_hotels = base_geo.mark_text(dy=30, color='white', fontSize=10).encode(text='Label_Hotels')
            text_geo_revs = base_geo.mark_text(dy=42, color='white', fontSize=10).encode(text='Label_Reviews')

            st.altair_chart(bars_geo + text_geo_score + text_geo_hotels + text_geo_revs, use_container_width=True)
            export_summary_list.append(geo_df)
            
            # --- AI SECTION ---
            st.divider()
            st.markdown("### Portfolio Insights")
            st.caption("Generate a strategic summary based on the calculated metrics above.")
            
            if st.button("Generate Portfolio Report", type="primary"):
                with st.spinner("Analyzing portfolio data..."):
                    summary = generate_portfolio_summary(curr_calc)
                    st.session_state['ai_result'] = summary
            
            if st.session_state['ai_result']:
                st.markdown(f"<div class='ai-box'><div class='ai-title'>Portfolio Executive Summary</div>{st.session_state['ai_result']}</div>", unsafe_allow_html=True)

        else: st.info("Run an analysis on the Live Dashboard to see Current Analysis stats here.")

        st.divider()
        if st.session_state['active_portfolio_name']:
            st.subheader(f"Historical Log: {st.session_state['active_portfolio_name']}")
            
            if os.path.exists(HISTORY_FILE):
                hist_df = pd.read_csv(HISTORY_FILE)
                hist_df['unique_key'] = hist_df['timestamp'] + "_" + hist_df['group_name']
                hist_df_display = hist_df[(hist_df['email'] == st.session_state['user_email']) & (hist_df['group_name'] == st.session_state['active_portfolio_name']) & (~hist_df['unique_key'].isin(st.session_state['hidden_history_ids']))].copy().sort_values(by="timestamp", ascending=False)
                
                if not hist_df_display.empty:
                    h_cols = st.columns([1.5, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6])
                    headers = ["Timestamp", "Count", "Google Score", "Expedia Score", "TripAdvisor Score", "Booking Score", "Google #", "Expedia #", "TripAdvisor #", "Booking #", "Weighted Avg.", ""]
                    for c, h in zip(h_cols, headers): c.markdown(f"<div class='hist-header'>{h}</div>", unsafe_allow_html=True)
                    for index, row in hist_df_display.iterrows():
                        c_cols = st.columns([1.5, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6])
                        c_cols[0].markdown(f"<div class='hist-val'>{row['timestamp']}</div>", unsafe_allow_html=True)
                        c_cols[1].markdown(f"<div class='hist-val'>{row['hotel_count']}</div>", unsafe_allow_html=True)
                        c_cols[2].markdown(f"<div class='hist-val'>{row.get('avg_google_norm', '-') or '-'}</div>", unsafe_allow_html=True)
                        c_cols[3].markdown(f"<div class='hist-val'>{row.get('avg_expedia_norm', '-') or '-'}</div>", unsafe_allow_html=True)
                        c_cols[4].markdown(f"<div class='hist-val'>{row.get('avg_ta_norm', '-') or '-'}</div>", unsafe_allow_html=True)
                        c_cols[5].markdown(f"<div class='hist-val'>{row.get('avg_booking_norm', '-') or '-'}</div>", unsafe_allow_html=True)
                        c_cols[6].markdown(f"<div class='hist-val'>{int(row.get('total_reviews_google', 0)):,}</div>", unsafe_allow_html=True)
                        c_cols[7].markdown(f"<div class='hist-val'>{int(row.get('total_reviews_expedia', 0)):,}</div>", unsafe_allow_html=True)
                        c_cols[8].markdown(f"<div class='hist-val'>{int(row.get('total_reviews_ta', 0)):,}</div>", unsafe_allow_html=True)
                        c_cols[9].markdown(f"<div class='hist-val'>{int(row.get('total_reviews_booking', 0)):,}</div>", unsafe_allow_html=True)
                        c_cols[10].markdown(f"<div class='hist-val'><b>{row['weighted_avg_score']}</b></div>", unsafe_allow_html=True)
                        if c_cols[11].button("👁️", key=f"hide_h_{index}"): st.session_state['hidden_history_ids'].add(row['unique_key']); st.rerun()
                        st.markdown("<hr style='margin:2px 0; border:0; border-top:1px solid #333'>", unsafe_allow_html=True)
                else: st.info("No snapshots found.")
            else: st.info("No history logs found yet.")

        # --- CONSOLIDATED EXPORTS ---
        # PREPARE EXPORT DATAFRAME WITH REQUESTED COLUMNS & FIXED DATE
        if 'curr_calc' in locals():
            export_hotel_data = curr_calc.copy()
            rename_map_export = {
                "Google_Raw": "Google Score", "Expedia_Raw": "Expedia Score", "TA_Raw": "TripAdvisor Score", "Booking_Raw": "Booking Score",
                "Google_10": "Google Score (Normalized)", "Expedia_10": "Expedia Score (Normalized)",
                "TA_10": "TripAdvisor Score (Normalized)", "Booking_10": "Booking Score (Normalized)",
                "Google_N": "# Google Reviews", "Expedia_N": "# Expedia Reviews",
                "TA_N": "# TripAdvisor Reviews", "Booking_N": "# Booking Reviews",
                "Weighted_Avg": "Weighted Avg. Review Score", "Last_Scraped": "Last Scraped"
            }
            export_hotel_data = export_hotel_data.rename(columns=rename_map_export)
            
            # --- DATE CLEANUP LOGIC ---
            today_str = datetime.now().strftime("%Y-%m-%d")
            if 'Last Scraped' not in export_hotel_data.columns: export_hotel_data['Last Scraped'] = None
            if 'Last_Scraped' in export_hotel_data.columns:
                export_hotel_data['Last Scraped'] = export_hotel_data['Last Scraped'].fillna(export_hotel_data['Last_Scraped'])
            export_hotel_data['Last Scraped'] = export_hotel_data['Last Scraped'].fillna(today_str)
            
            # Define exact 17 columns requested
            final_export_cols = [
                "Name", "City", "State", 
                "Google Score", "Expedia Score", "TripAdvisor Score", "Booking Score", 
                "Google Score (Normalized)", "Expedia Score (Normalized)", "TripAdvisor Score (Normalized)", "Booking Score (Normalized)",
                "# Google Reviews", "# Expedia Reviews", "# TripAdvisor Reviews", "# Booking Reviews", 
                "Weighted Avg. Review Score", "Last Scraped"
            ]
            
            for c in final_export_cols:
                if c not in export_hotel_data.columns: export_hotel_data[c] = None
            
            export_hotel_data = export_hotel_data[final_export_cols]

        st.write("")
        c_ex1, c_ex2 = st.columns(2)
        if not export_hotel_data.empty:
            out_xl = io.BytesIO()
            with pd.ExcelWriter(out_xl, engine='openpyxl') as writer:
                if export_summary_list:
                    row_idx = 0
                    for i, df_chunk in enumerate(export_summary_list):
                        if not df_chunk.empty:
                            df_chunk.to_excel(writer, sheet_name='Executive Summary', startrow=row_idx, index=False)
                            row_idx += len(df_chunk) + 3
                
                export_hotel_data.to_excel(writer, sheet_name='Current Hotel Data', index=False)
                
                if st.session_state['active_portfolio_name'] and 'hist_df_display' in locals() and not hist_df_display.empty:
                    export_hist = hist_df_display.drop(columns=['unique_key']) if 'unique_key' in hist_df_display.columns else hist_df_display
                    export_hist.to_excel(writer, sheet_name='Historical Log', index=False)
            c_ex1.download_button("📥 Download Excel Report", out_xl.getvalue(), "Portfolio_Report.xlsx", use_container_width=True)

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                zf.writestr("Current_Hotel_Data.csv", export_hotel_data.to_csv(index=False))
                if st.session_state['active_portfolio_name'] and 'hist_df_display' in locals() and not hist_df_display.empty:
                    export_hist = hist_df_display.drop(columns=['unique_key']) if 'unique_key' in hist_df_display.columns else hist_df_display
                    zf.writestr("Historical_Log.csv", export_hist.to_csv(index=False))
            c_ex2.download_button("📄 Download CSVs (.zip)", zip_buffer.getvalue(), "Portfolio_Data.zip", mime="application/zip", use_container_width=True)
