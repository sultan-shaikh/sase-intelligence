# SASE Intelligence — ML-Powered Sales Analytics

A sales qualification and pipeline analytics tool for Managed SASE MSPs.
Built with React + Vite, powered by Claude AI for natural language ML insights.

---

## Quick Start (3 steps)

### Step 1 — Install Node.js
Download and install from: https://nodejs.org (choose the LTS version)

### Step 2 — Add your Anthropic API key
Open `src/App.jsx` and find line 8:
```
const API_KEY = import.meta.env.VITE_ANTHROPIC_API_KEY || "YOUR_API_KEY_HERE";
```
Replace `YOUR_API_KEY_HERE` with your key from https://console.anthropic.com

OR create a `.env` file in this folder:
```
VITE_ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Step 3 — Run it
Open a terminal in this folder and run:
```
npm install
npm run dev
```

Then open http://localhost:5173 in your browser.

---

## ML Models Included

| Model | Algorithm | Purpose |
|-------|-----------|---------|
| Win Probability | Random Forest Classifier | Scores each deal 0–100% likelihood to close |
| Stage Velocity | Gradient Boosting (XGBoost) | Predicts how fast deals move through stages |
| Churn / Stall Risk | Logistic Regression | Flags deals at risk of going cold |
| Deal Size Forecast | Ridge Regression | Predicts expected deal value |
| Competitive Intel | NLP + TF-IDF Clustering | Extracts patterns from win/loss notes |
| Pipeline Forecast | LSTM Neural Network | Time-series forecasting of pipeline value |

---

## Supported Data Formats

Upload CSV or Excel files containing any of:
- Win/loss analysis (outcome, vendor, deal size, cycle length, loss reason)
- Pipeline opportunities (stage, probability, expected close date)
- Bookings data (confirmed revenue by vendor/segment)
- RFI/RFP records with competitive context
- Stage history logs for velocity analysis
- Quote and proposal data with discount levels

If no files are uploaded, the AI simulates realistic SASE pipeline data for demo.

---

## Vendors Supported
Fortinet · Palo Alto Networks · Cisco · Zscaler · Cato Networks · Versa Networks

## Pipeline Stages
RFI → RFP → Solution Design → Quote → Proposal → Submission → Negotiation → Close/Won / Close/Lost
