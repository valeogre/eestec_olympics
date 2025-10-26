# SIEM Fraud Detection
Sistem de Detectare a Fraudelor in Timp Real pentru Tranzactii POS


## 1. Prezentare Generala

Acest proiect reprezinta un sistem complet de detectare a fraudelor financiare in timp real, bazat pe invatare automata (Machine Learning) si procesarea fluxurilor de date.

**Obiective principale:**
- Detectarea tranzactiilor suspecte pe baza modelelor istorice.
- Analiza comportamentului utilizatorilor si comerciantilor.
- Automatizarea procesului de alertare si raportare a fraudelor.


## 2. Arhitectura Sistemului

Proiectul include doua componente majore:

### 2.1 Backend (Python)
- Se ocupa de antrenarea modelului de invatare automata si de inferenta.
- Proceseaza tranzactiile in timp real, provenite dintr-un stream de date sau dintr-un fisier CSV.
- Comunica cu un API pentru raportarea tranzactiilor suspecte.

### 2.2 Frontend (Streamlit)
- O interfata web interactiva pentru vizualizarea tranzactiilor.
- Permite monitorizarea in timp real si reantrenarea modelului.


## 3. Fluxul de Date

1. **Faza Offline (Antrenarea Modelului)**
   - Se incarca datele istorice etichetate (fisier CSV).
   - Se preproceseaza si se codifica variabilele categoriale.
   - Se antreneaza modelul LightGBM pentru clasificarea tranzactiilor frauduloase.
   - Se salveaza modelul (`fraud_model.pkl`) si encoder-ele (`le_*.pkl`).

2. **Faza Online (Procesare in Timp Real)**
   - Sistemul se conecteaza la un stream de tranzactii prin SSE.
   - Fiecare tranzactie este preprocesata si evaluata de model.
   - Tranzactiile suspecte sunt marcate si trimise catre API pentru raportare.
   - Toate tranzactiile pot fi salvate in baza de date `transactions.db`.


## 4. Structura Proiectului


siem-fraud-detection/
├── training.py            # Script pentru antrenarea modelului
├── detector.py            # Procesarea tranzactiilor in timp real
├── app.py                 # Interfata web (Streamlit)
├── fraud_model.pkl        # Modelul LightGBM antrenat
├── le_*.pkl               # Fisierele LabelEncoder pentru coloanele categoriale
├── transactions.db        # Baza de date SQLite cu tranzactii procesate
├── requirements.txt       # Dependinte Python
└── README.md              # Documentatia proiectului



## 5. Antrenarea Modelului (train_model.py)

### 5.1 Preprocesarea Datelor

python
def preprocess(df, label_encoders=None, training=True):
    df["trans_datetime"] = pd.to_datetime(df["trans_date"] + " " + df["trans_time"], errors="coerce")
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df["age"] = ((df["trans_datetime"] - df["dob"]).dt.days / 365.25).fillna(0)
    df["hour"] = df["trans_datetime"].dt.hour.fillna(0)
    df["distance_km"] = df.apply(
        lambda x: haversine(x["lat"], x["long"], x["merch_lat"], x["merch_long"]), axis=1
    )


Aceasta functie calculeaza varsta clientului, ora tranzactiei si distanta dintre locatia clientului si cea a comerciantului.

### 5.2 Codificarea Coloanelor Categoriale

python
cat_cols = ["category", "merchant", "state", "gender", "job"]
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    joblib.dump(le, f"le_{col}.pkl")


Se foloseste `LabelEncoder` pentru a transforma valorile textuale in coduri numerice.

### 5.3 Antrenarea Modelului

python
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, valid_data],
    valid_names=["train", "valid"],
    num_boost_round=1000,
    callbacks=[early_stopping(stopping_rounds=50), log_evaluation(50)]
)


Modelul este antrenat pentru clasificarea binara (frauda / non-frauda) folosind LightGBM, un algoritm de tip Gradient Boosting.

### 5.4 Salvarea Modelului si Encoderelor

python
joblib.dump(model, "fraud_model.pkl")
for col, le in label_encoders.items():
    joblib.dump(le, f"le_{col}.pkl")

Modelul si encoder-ele sunt salvate local pentru utilizarea lor ulterioara in procesarea in timp real.


## 6. Procesarea in Timp Real (stream_processor.py)

### 6.1 Configurare

python
STREAM_URL = "https://95.217.75.14:8443/stream"
FLAG_URL = "https://95.217.75.14:8443/api/flag"
MODEL_PATH = "fraud_model.pkl"
FRAUD_THRESHOLD = 0.5
CAT_COLS = ['category', 'merchant', 'state', 'gender', 'job']


Defineste endpoint-urile, calea modelului si pragul de decizie pentru frauda.

### 6.2 Incarcarea Modelului si Encoderelor

python
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoders = {}
    for col in CAT_COLS:
        encoders[col] = joblib.load(f"le_{col}.pkl")
    return model, encoders

Se incarca modelul LightGBM si encoder-ele salvate.

### 6.3 Preprocesarea Tranzactiilor

python
def preprocess_tx(tx, encoders):
    amt = float(tx.get('amt', 0.0))
    dt = pd.to_datetime(tx.get('trans_date') + ' ' + tx.get('trans_time'), errors='coerce')
    dob_dt = pd.to_datetime(tx.get('dob'), errors='coerce')
    age = max(0.0, (dt - dob_dt).days / 365.25)
    hour = int(dt.hour)
    distance = haversine(tx['lat'], tx['long'], tx['merch_lat'], tx['merch_long'])
    cat_encoded = [encode_value(encoders.get(col), tx.get(col, '<UNK>')) for col in CAT_COLS]
    return np.array([amt, age, hour, distance] + cat_encoded).reshape(1, -1)

Transforma tranzactiile brute in vectori numerici utilizabili de modelul ML.

### 6.4 Predictie si Raportare

python
def process_event(tx, model, encoders):
    X = preprocess_tx(tx, encoders)
    prob = float(model.predict(X)[0])
    is_fraud = 1 if prob > FRAUD_THRESHOLD else 0
    status, resp = flag_transaction(tx['trans_num'], is_fraud)
    print(f"{tx['trans_num']} -> prob={prob:.4f}, flag={is_fraud}, status={status}")


* Calculeaza probabilitatea ca tranzactia sa fie frauduloasa.
* Trimite rezultatul catre API-ul de raportare.

### 6.5 Procesarea Fluxului de Date

python
with requests.get(STREAM_URL, headers=headers, stream=True, verify=False, timeout=15) as response:
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for line in response.iter_lines(decode_unicode=True):
            if line and not line.startswith(":"):
                tx = json.loads(line[len("data:"):].strip())
                executor.submit(process_event, tx, model, encoders)

Sistemul asculta continuu fluxul de tranzactii, proceseaza in paralel fiecare eveniment si trimite rezultatele catre API.


## 7. Rulare

### 7.1 Instalarea Dependintelor

bash
pip install -r requirements.txt

### 7.2 Antrenarea Modelului

bash
python training.py

### 7.3 Pornirea Procesorului de Stream

bash
python detector.py


### 7.4 Pornirea Interfetei Web

bash
streamlit run app.py


## 8. Baza de Date

Tranzactiile procesate pot fi stocate intr-o baza SQLite (`transactions.db`) pentru analiza ulterioara sau pentru afisarea in interfata Streamlit.

Structura recomandata:

id | trans_num | amount | probability | is_fraud | timestamp

Aceasta baza poate fi folosita pentru:

* Analiza istorica a incidentelor.
* Vizualizarea in dashboard.
* Export CSV pentru reantrenarea modelului.
