# Visitor Intelligence – AI‑Powered Purchase Intent Engine

A premium Streamlit SaaS that predicts purchase intent from completed e‑commerce browsing sessions. The app ships with a pre‑trained XGBoost model and a full UI (public landing page, dashboard, prediction, insights, methodology, and about pages).

---

## ✨ Overview
- **What it does** – Takes 16 aggregated session features (no `PageValues`) and outputs an intent band (Low / Medium / High) with a ranking score.
- **Why it matters** – Enables analysts to prioritize high‑intent visitors for targeted follow‑up.
- **Tech stack** – Python 3.12, Streamlit, Firebase Auth (Google OAuth + Email/Password + Guest demo), XGBoost, scikit‑learn pipelines.

---

## 📦 Installation
```powershell
# Clone the repo (run this yourself)
# git clone https://github.com/GaneshSKumbhar/Visitor-Intelligence.git
# cd Visitor-Intelligence

# Create a virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
> **Note** – All required binary wheels are in `requirements.txt`. No additional system packages are needed.

---

## 🔑 Backend / Firebase Setup
1. Open the Firebase console and create a new project (or reuse an existing one).
2. Enable **Google** and **Email/Password** sign‑in methods under *Authentication → Sign‑in method*.
3. Add `localhost` to *Authentication → Settings → Authorized domains*.
4. Replace the placeholder Firebase config in `app/firebase_auth/index.html` with your project's config snippet (the `<script src="https://.../firebase-app.js"></script>` block).

---

## 🖥️ Front‑end Setup
The UI is built entirely with Streamlit; no separate bundling step is required.

---

## ▶️ Running the Application
```powershell
# Activate the virtual environment if not already active
.\.venv\Scripts\Activate.ps1

# Start Streamlit
streamlit run app/app.py
```
The command prints a local URL (e.g. `http://localhost:8501`). Open it in a browser.

### URL Routes
| Route | Access | Description |
|------|--------|-------------|
| `/` | Public | Modern SaaS landing page with hero, features, workflow, and metrics |
| `/login` | Public | Firebase login (Google, Email/Password, Guest) |
| `/dashboard` | Protected | KPI cards, conversion distribution, cohort analysis |
| `/predict` | Protected | Real‑time session scoring and CSV batch scoring |
| `/insights` | Protected | Model evaluation, SHAP/Permutation importance |
| `/methodology` | Public | ML pipeline architecture and decision‑theoretic cost formula |
| `/about` | Public | Project background, stack, citations |

---

## 🧪 Tests
A Streamlit‑based test suite lives in `test_auth_refresh.py`. Run it with:
```powershell
python -m pytest test_auth_refresh.py
```
All tests should pass, confirming that:
- Public homepage loads without auth
- Guest / Google login stores a signed session token
- Refresh restores the token
- Logout clears the token and Firebase state

---

## 📂 Project Structure
```text
app/
│   app.py               # Main Streamlit app (routing, UI components)
│   firebase_auth/
│       index.html       # Firebase UI component (sign‑in / sign‑out)
│   logic.py             # Feature schema, validation, scoring helpers
src/
│   pipeline.py          # Model preprocessing pipeline (shared)
│   train_project.py     # Optional reproducible analysis script
models/
│   xgb_no_pagevalues.joblib   # Pre‑trained XGBoost pipeline
│   app_meta.json               # Score bands, thresholds, categories
data/processed/
│   shoppers_clean.csv   # Cleaned dataset (12 205 rows)
reports/
│   project_report.pdf
│   project_report.md
│   supplemental_results.json
│   figures/ …
```

---

## 📜 License & Citation
- Academic project – feel free to fork, experiment, and cite.
- Original data: *UCI Online Shoppers Purchasing Intention Dataset* – see the citation in the README.

---

## 🙏 Acknowledgements
- **Google Antigravity** – AI‑assisted development and UI generation.
- **Streamlit** – Rapid prototyping of data‑centric web apps.
- **Firebase** – Authentication backend.

---

*Happy hacking!*