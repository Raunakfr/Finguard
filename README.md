# 🛡️ FinGuard — Invisible Fraud Detector System

> 🎯 AI Solution Expo | JSS University
> 👥 Team: Tanishq, Kshitij, Shraddha, Anika

---

## 🚀 What is FinGuard?

**FinGuard** is a real-time, behavior-based fraud detection system powered by a Random Forest ML model.

It analyzes UPI/digital transactions and classifies them as:

* ✅ Safe
* ⚠️ Suspicious
* 🚨 Fraud

💡 Each decision comes with a **plain-language explanation** for transparency and trust.

---

## ⚡ Quick Start

### 🪟 Windows

```bash
Double-click START_SERVER.bat
```

### 🍎 Mac / 🐧 Linux

```bash
chmod +x start_server.sh
./start_server.sh
```

🌐 Then open your browser:
👉 http://localhost:5000

---

## ✨ Features

* 🛡️ Random Forest ML model (trained on 284,807 transactions)
* ⚡ Real-time risk scoring (0–100 composite score)
* 🔐 OTP email verification for suspicious transactions
* 📊 Transaction history with SQLite database
* 📈 Statistics dashboard
* 🤖 Explainable AI — plain-language verdicts
* 🚩 Up to 6 risk flags with severity levels (High / Medium / Low)

---

## 📊 Risk Score Thresholds

| Score Range | Verdict       | Action                    |
| ----------- | ------------- | ------------------------- |
| 0–29        | ✅ Safe        | Transaction Allowed       |
| 30–59       | ⚠️ Suspicious | OTP Verification Required |
| 60+         | 🚨 Fraud      | Transaction Blocked       |

---

## 🧰 Tech Stack

* **Backend:** Python, Flask, SQLite
* **ML Model:** Random Forest (scikit-learn)
* **Dataset:** Kaggle Credit Card Fraud Dataset
* **Frontend:** HTML, CSS, JavaScript (Vanilla)
* **Email OTP:** SMTP via Gmail

---

## 🔐 Email OTP Setup (Optional)

Edit `app.py` and add your credentials:

```python
sender_email = "your_email@gmail.com"
app_password = "your_16_char_app_password"
```

🔗 Generate App Password:
https://myaccount.google.com/apppasswords

---

## 📈 Model Performance

* 📊 Trained on: **227,845 transactions**
* ⚠️ False Positive Rate: **~1.7%**
* 🧠 Key Features:

  * Amount ratio
  * Transaction hour risk
  * Location
  * Device
  * Merchant category
  * Recipient type
  * Account age
  * Prior fraud history

---

## 💡 Key Idea

FinGuard uses a **hybrid approach**:

* 🧠 Machine Learning → Pattern detection
* ⚙️ Rule-based logic → Reliability
* 🔐 OTP system → Real-time intervention

👉 This ensures both **accuracy and security** — similar to real-world banking systems.

---

## 🏁 Final Note

FinGuard is not just a model — it's a **complete fraud prevention system** designed to simulate real-world financial security workflows.

---
