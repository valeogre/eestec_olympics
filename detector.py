#!/usr/bin/env python3
"""
fraud_detector.py — combined live stream + offline CSV detector with DB logging

Modes:
 [1] Live stream mode  — connects to transaction stream and flags frauds in real time.
 [2] Offline CSV mode  — loads local CSV file, runs the model once, prints summary.

Adds:
  • SQLite logging of predictions (transactions.db)
  • Batch predictions (fast)
  • Encoder mapping (no per-row LabelEncoder transform)
  • Threaded flag posting for live mode
"""

import json
import time
import joblib
import requests
import threading
import queue
import concurrent.futures
import pandas as pd
import numpy as np
import urllib3
import os
import sqlite3
from sklearn.metrics import classification_report, roc_auc_score

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -------- CONFIG ----------
STREAM_URL = "https://95.217.75.14:8443/stream"
FLAG_URL = "https://95.217.75.14:8443/api/flag"
API_KEY = "fabfdad781aea9c5a836e653fabffb6fcc66511fc6db870dc212ca8959a5e82d"
MODEL_PATH = "fraud_model.pkl"
ENCODER_PREFIX = "le_"
CAT_COLS = ['category', 'merchant', 'state', 'gender', 'job']

MAX_WORKERS_FLAG = 8
PREDICT_BATCH_SIZE = 64
PREDICT_BATCH_TIMEOUT = 0.35
QUEUE_MAXSIZE = 10000
FRAUD_THRESHOLD = 0.5
FLAG_TIMEOUT = 6
DB_PATH = "transactions.db"
# ------------------------

headers = {"X-API-Key": API_KEY}

# ---------- Helpers ----------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def safe_get_float(d, key, default=0.0):
    try:
        return float(d.get(key, default) or default)
    except Exception:
        return default

# ---------- DB Setup ----------
def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            ssn TEXT,
            cc_num TEXT,
            first TEXT,
            last TEXT,
            gender TEXT,
            street TEXT,
            city TEXT,
            state TEXT,
            zip TEXT,
            lat REAL,
            long REAL,
            city_pop REAL,
            job TEXT,
            dob TEXT,
            acct_num TEXT,
            trans_num TEXT PRIMARY KEY,
            trans_date TEXT,
            trans_time TEXT,
            unix_time REAL,
            category TEXT,
            amt REAL,
            merchant TEXT,
            merch_lat REAL,
            merch_long REAL,
            distance REAL,
            age REAL,
            hour INTEGER,
            prob REAL,
            is_fraud INTEGER,
            timestamp TEXT
        )
    """)
    conn.commit()
    return conn

# ---------- Load model & encoders ----------
def load_artifacts_and_build_maps():
    print("Loading model and encoders...")
    model = joblib.load(MODEL_PATH)
    enc_map, unk_index = {}, {}
    for col in CAT_COLS:
        try:
            le = joblib.load(f"{ENCODER_PREFIX}{col}.pkl")
            classes = list(map(str, le.classes_))
            mapping = {c: i for i, c in enumerate(classes)}
            unk_index[col] = mapping.get('<UNK>', 0)
            enc_map[col] = mapping
        except Exception as e:
            print(f"Warning: can't load encoder for {col}: {e}")
            enc_map[col], unk_index[col] = {}, 0
    return model, enc_map, unk_index

# ---------- Convert transaction to feature vector ----------
def tx_to_features(tx, enc_map, unk_index):
    amt = safe_get_float(tx, 'amt', 0.0)
    dob = tx.get('dob')
    trans_date = tx.get('trans_date')
    trans_time = tx.get('trans_time')
    dt = None
    try:
        if trans_date and trans_time:
            dt = pd.to_datetime(trans_date + ' ' + trans_time, errors='coerce')
        elif trans_date:
            dt = pd.to_datetime(trans_date, errors='coerce')
    except:
        dt = pd.NaT
    try:
        dob_dt = pd.to_datetime(dob, errors='coerce') if dob else pd.NaT
    except:
        dob_dt = pd.NaT

    if dt is pd.NaT or dob_dt is pd.NaT or pd.isna(dt) or pd.isna(dob_dt):
        age = 30.0
    else:
        age = max(0.0, (dt - dob_dt).days / 365.25)

    hour = int(dt.hour) if (dt is not None and not pd.isna(dt)) else 12
    lat = safe_get_float(tx, 'lat', 0.0)
    lon = safe_get_float(tx, 'long', 0.0)
    merch_lat = safe_get_float(tx, 'merch_lat', 0.0)
    merch_long = safe_get_float(tx, 'merch_long', 0.0)
    distance = haversine(lat, lon, merch_lat, merch_long)

    cat_codes = []
    for col in CAT_COLS:
        sv = str(tx.get(col, '<UNK>')) if tx.get(col) is not None else '<UNK>'
        code = enc_map.get(col, {}).get(sv, unk_index.get(col, 0))
        cat_codes.append(code)

    return [amt, age, hour, distance] + cat_codes

# ---------- Flag POST ----------
def flag_transaction_post(trans_num, flag_value):
    payload = {"trans_num": trans_num, "flag_value": int(flag_value)}
    try:
        r = requests.post(FLAG_URL, headers=headers, json=payload, timeout=FLAG_TIMEOUT, verify=False)
        try:
            return r.status_code, r.json()
        except:
            return r.status_code, r.text
    except Exception as e:
        return None, str(e)

# ---------- Batch predictor ----------
# ---------- Batch predictor ----------  
class BatchPredictor(threading.Thread):
    def __init__(self, model, enc_map, unk_index, input_queue, flag_executor):
        super().__init__(daemon=True)
        self.model = model
        self.enc_map = enc_map
        self.unk_index = unk_index
        self.input_queue = input_queue
        self.flag_executor = flag_executor
        self.stop_event = threading.Event()
        self.processed = 0
        self.conn = init_db()
        self.db_lock = threading.Lock()

    def stop(self):
        self.stop_event.set()

    def run(self):
        batch_feats, batch_meta = [], []
        while not self.stop_event.is_set():
            try:
                item = self.input_queue.get(timeout=PREDICT_BATCH_TIMEOUT)
                if item is None:
                    break
                tx = item
                trans_num = tx.get('trans_num')
                batch_meta.append((trans_num, tx))
                batch_feats.append(tx_to_features(tx, self.enc_map, self.unk_index))

                while len(batch_feats) < PREDICT_BATCH_SIZE:
                    try:
                        item = self.input_queue.get_nowait()
                        if item is None:
                            break
                        tx = item
                        trans_num = tx.get('trans_num')
                        batch_meta.append((trans_num, tx))
                        batch_feats.append(tx_to_features(tx, self.enc_map, self.unk_index))
                    except queue.Empty:
                        break

                if batch_feats:
                    X = np.array(batch_feats, dtype=np.float32)
                    probs = self.model.predict(X)
                    for (tn, tx), prob in zip(batch_meta, probs):
                        is_fraud = int(prob > FRAUD_THRESHOLD)
                        self.flag_executor.submit(flag_transaction_post, tn, is_fraud)
                        print(f"[{time.strftime('%H:%M:%S')}] trans={tn} prob={prob:.4f} flag={is_fraud}")
                        self.processed += 1

                        # --- Write all fields to SQLite ---
                        with self.db_lock:
                            try:
                                self.conn.execute("""
                                    INSERT OR REPLACE INTO transactions VALUES (
                                        :ssn, :cc_num, :first, :last, :gender, :street, :city, :state, :zip,
                                        :lat, :long, :city_pop, :job, :dob, :acct_num, :trans_num,
                                        :trans_date, :trans_time, :unix_time, :category, :amt, :merchant,
                                        :merch_lat, :merch_long, :distance, :age, :hour, :prob, :is_fraud,
                                        :timestamp
                                    )
                                """, {
                                    "ssn": tx.get("ssn"),
                                    "cc_num": tx.get("cc_num"),
                                    "first": tx.get("first"),
                                    "last": tx.get("last"),
                                    "gender": tx.get("gender"),
                                    "street": tx.get("street"),
                                    "city": tx.get("city"),
                                    "state": tx.get("state"),
                                    "zip": tx.get("zip"),
                                    "lat": safe_get_float(tx, "lat"),
                                    "long": safe_get_float(tx, "long"),
                                    "city_pop": safe_get_float(tx, "city_pop"),
                                    "job": tx.get("job"),
                                    "dob": tx.get("dob"),
                                    "acct_num": tx.get("acct_num"),
                                    "trans_num": tx.get("trans_num"),
                                    "trans_date": tx.get("trans_date"),
                                    "trans_time": tx.get("trans_time"),
                                    "unix_time": safe_get_float(tx, "unix_time"),
                                    "category": tx.get("category"),
                                    "amt": safe_get_float(tx, "amt"),
                                    "merchant": tx.get("merchant"),
                                    "merch_lat": safe_get_float(tx, "merch_lat"),
                                    "merch_long": safe_get_float(tx, "merch_long"),
                                    "distance": tx.get("distance"),
                                    "age": tx.get("age"),
                                    "hour": tx.get("hour"),
                                    "prob": float(prob),
                                    "is_fraud": is_fraud,
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                                })
                                self.conn.commit()
                            except Exception as e:
                                print("DB insert error:", e)

                    batch_feats, batch_meta = [], []
            except queue.Empty:
                continue
            except Exception as e:
                print("BatchPredictor error:", e)
                time.sleep(0.1)

# ---------- Stream Reader ----------
def stream_reader_loop(input_queue):
    try:
        with requests.get(STREAM_URL, headers=headers, stream=True, verify=False, timeout=15) as response:
            response.raise_for_status()
            print("Connected to live stream...")
            for raw in response.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()
                try:
                    tx = json.loads(line)
                except:
                    continue
                try:
                    input_queue.put(tx, timeout=0.5)
                except queue.Full:
                    _ = input_queue.get_nowait()
                    input_queue.put(tx)
    except KeyboardInterrupt:
        print("\nStream reader interrupted.")
    except Exception as e:
        print("Stream reader error:", e)

# ---------- CSV Mode ----------
def process_csv(csv_path="short.csv"):
    print(f"Offline mode — loading CSV: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return

    model, enc_map, unk_index = load_artifacts_and_build_maps()
    conn = init_db()
    db_lock = threading.Lock()

    df = pd.read_csv(csv_path, sep="|")
    feats = []
    for _, tx in df.iterrows():
        tx_dict = tx.to_dict()
        feats.append(tx_to_features(tx_dict, enc_map, unk_index))

    X = np.array(feats, dtype=np.float32)
    probs = model.predict(X)
    preds = (probs > FRAUD_THRESHOLD).astype(int)

    df["fraud_prob"] = probs
    df["predicted_fraud"] = preds

    with db_lock:
        for _, row in df.iterrows():
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO transactions VALUES (
                        :ssn, :cc_num, :first, :last, :gender, :street, :city, :state, :zip,
                        :lat, :long, :city_pop, :job, :dob, :acct_num, :trans_num,
                        :trans_date, :trans_time, :unix_time, :category, :amt, :merchant,
                        :merch_lat, :merch_long, :distance, :age, :hour, :prob, :is_fraud,
                        :timestamp
                    )
                """, {
                    "ssn": row.get("ssn"),
                    "cc_num": row.get("cc_num"),
                    "first": row.get("first"),
                    "last": row.get("last"),
                    "gender": row.get("gender"),
                    "street": row.get("street"),
                    "city": row.get("city"),
                    "state": row.get("state"),
                    "zip": row.get("zip"),
                    "lat": safe_get_float(row, "lat"),
                    "long": safe_get_float(row, "long"),
                    "city_pop": safe_get_float(row, "city_pop"),
                    "job": row.get("job"),
                    "dob": row.get("dob"),
                    "acct_num": row.get("acct_num"),
                    "trans_num": row.get("trans_num"),
                    "trans_date": row.get("trans_date"),
                    "trans_time": row.get("trans_time"),
                    "unix_time": safe_get_float(row, "unix_time"),
                    "category": row.get("category"),
                    "amt": safe_get_float(row, "amt"),
                    "merchant": row.get("merchant"),
                    "merch_lat": safe_get_float(row, "merch_lat"),
                    "merch_long": safe_get_float(row, "merch_long"),
                    "distance": row.get("distance"),
                    "age": row.get("age"),
                    "hour": row.get("hour"),
                    "prob": float(row["fraud_prob"]),
                    "is_fraud": int(row["predicted_fraud"]),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
            except Exception as e:
                print("DB insert error:", e)
        conn.commit()

    if "is_fraud" in df.columns:
        auc = roc_auc_score(df["is_fraud"], probs)
        print(f"AUC = {auc:.4f}")
        print(classification_report(df["is_fraud"], preds))
    else:
        print("No 'is_fraud' column — showing top suspicious transactions:")
        print(df.sort_values("fraud_prob", ascending=False).head(10)[["trans_num", "amt", "fraud_prob"]])

# ---------- MAIN ----------
def main():
    model, enc_map, unk_index = load_artifacts_and_build_maps()
    q = queue.Queue(maxsize=QUEUE_MAXSIZE)
    flag_executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_FLAG)
    predictor = BatchPredictor(model, enc_map, unk_index, q, flag_executor)
    predictor.start()

    try:
        stream_reader_loop(q)
    finally:
        print("Shutting down...")
        q.put_nowait(None)
        predictor.stop()
        predictor.join(timeout=5)
        flag_executor.shutdown(wait=True)
        print(f"Processed ~{predictor.processed} transactions.")

# ---------- Entry ----------
if __name__ == "__main__":
    print("Select mode:")
    print(" [1] Live stream mode (real-time detection)")
    print(" [2] Offline CSV mode (analyze local file)")
    mode = input("> ").strip()

    if mode == "1":
        main()
    else:
        csv_path = input("CSV file path [default: short.csv]: ").strip() or "short.csv"
        process_csv(csv_path)
