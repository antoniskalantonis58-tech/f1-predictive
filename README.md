
# 🏎️ Formula 1 Predictive Modeling (2000–2020)

**Goal:** Build a predictive pipeline to determine whether a Formula 1 driver will finish **in the Top 10** based on grid position, constructor, circuit, and year.
We compare three models:

* **Baseline** (naïve grid-based predictor)
* **Logistic Regression**
* **Random Forest Classifier**

---

## 🔍 Research Question

How much predictive value can we extract from simple F1 race metadata (grid position, constructor, circuit, year)?
Do classical ML models outperform a naïve baseline based solely on grid position?

---

## 📂 Project Structure

```
f1-predictive/
│
├── data/
│   └── raw
│       ├── circuits.csv
│       ├── constructors.csv
│       ├── drivers.csv
│       ├── races.csv
│       └── results.csv
│
├── notebooks/
│   └── 01_data_import_cleaning.ipynb
│
├── results/
│   ├── circuits_boxplot.png
│   ├── constructors_boxplot.png
│   ├── grid_vs_finish.png
│   └── rf_feature_importances.png
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── evaluation.py
│   └── models.py
│
├── environment.yml
├── main.py
└── README.md
```



## 🧠 Key Insight

Even with limited features, **Logistic Regression significantly outperforms the baseline**, showing that simple F1 metadata contains strong predictive signals.

-
Test Git


