---
title: Customer Segmentation RFM
emoji: 🎯
colorFrom: purple
colorTo: yellow
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# 🎯 Customer Segmentation — RFM Analysis + K-Means Clustering

An interactive ML web app that segments customers using **RFM Analysis** and **K-Means Clustering**, with actionable marketing recommendations for each segment.

**Live Demo:** [View on Hugging Face Spaces](https://huggingface.co/spaces/taskeen786/customer-segmentation)

---

## 📸 Features

- 🎯 Automatic customer segmentation into 2–5 groups
- 📊 RFM Analysis (Recency, Frequency, Monetary)
- 🗂️ Segment profiles: Champions, Loyal, New, At Risk, Lost
- 📣 Marketing action recommendations per segment
- 📁 Upload your own CSV or use sample data
- ⬇️ Download segmented results as CSV
- 🌙 Dark mode professional UI

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Streamlit | Web app framework |
| Scikit-Learn | K-Means clustering |
| Pandas / NumPy | Data processing |
| Matplotlib | Visualizations |

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/TaskeenHussain/customer-segmentation
cd customer-segmentation
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 How to Use Your Own Data

Upload a CSV with these columns (or map your columns in the sidebar):
- **Recency** — days since last purchase
- **Frequency** — number of purchases
- **MonetaryValue** — total spend

---

## 📁 Project Structure

```
customer-segmentation/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 👤 Author

**Taskeen Hussain** — Data Analyst & Python Developer

- 🐙 GitHub: [github.com/TaskeenHussain](https://github.com/TaskeenHussain)
- 📊 Kaggle: [kaggle.com/taskeenhkbbeechtree](https://kaggle.com/taskeenhkbbeechtree)
- 📧 Email: taskeenuaf@gmail.com
- 🌐 Portfolio: [taskeenhussain.github.io/taskeen-portfolio](https://taskeenhussain.github.io/taskeen-portfolio)
