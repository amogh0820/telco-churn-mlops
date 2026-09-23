---
title: Telco Churn API
emoji: 📉
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Telco churn prediction API

Scores a telecom subscriber's probability of churning. Interactive docs at
`/docs`. Full source, tests, EDA notebook and training pipeline:
<https://github.com/YOUR_USERNAME/YOUR_REPO>.

Try it:

```bash
curl -X POST https://YOUR_SPACE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender":"Female","SeniorCitizen":"No","Partner":"Yes","Dependents":"No",
    "tenure":2,"PhoneService":"Yes","MultipleLines":"No",
    "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
    "DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes",
    "StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes",
    "PaymentMethod":"Electronic check","MonthlyCharges":95.5,"TotalCharges":190.0
  }'
```
