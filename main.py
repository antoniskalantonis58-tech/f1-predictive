import os

from src.data_loader import load_f1_data
from src.models import (
    baseline_predict,
    train_logistic_regression,
    train_random_forest
)
from src.evaluation import evaluate_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():

    # Debug info
    print("=== WORKING DIR ===", os.getcwd())
    print("=== FILES IN DATA/RAW ===", os.listdir("data/raw"))

    print("\n=== Loading F1 dataset ===")
    df = load_f1_data()
    print("Data loaded:", df.shape)

    # Input features and target
    features = ["grid", "year", "constructorId", "circuitId"]
    
    # ============================================================
    # RÉSULTATS PRINCIPAUX : 2000-2024
    # ============================================================
    print("\n" + "="*60)
    print("RÉSULTATS PRINCIPAUX : Période 2000-2024")
    print("="*60)
    
    X = df[features].values
    y = df["top10"].values

    print(f"\nNombre total d'échantillons : {len(df)}")
    print("\n=== Splitting train/test (80/20) ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ----------------------------
    # 1. Baseline Model
    # ----------------------------
    print("\n--- Baseline Model ---")
    baseline_pred = baseline_predict(X_test[:, 0])
    baseline_acc, baseline_auc = evaluate_model(
        y_test,
        baseline_pred,
        name="Baseline"
    )

    # ----------------------------
    # 2. Logistic Regression
    # ----------------------------
    print("\n--- Logistic Regression ---")
    lr_model = train_logistic_regression(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_acc, lr_auc = evaluate_model(
        y_test,
        lr_pred,
        name="Logistic Regression"
    )

    # ----------------------------
    # 3. Random Forest
    # ----------------------------
    print("\n--- Random Forest ---")
    rf_model = train_random_forest(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc, rf_auc = evaluate_model(
        y_test,
        rf_pred,
        name="Random Forest"
    )

    # ----------------------------
    # Summary Principal
    # ----------------------------
    print("\n" + "="*60)
    print("RÉSUMÉ PRINCIPAL (2000-2024)")
    print("="*60)
    print(f"Baseline           → Acc: {baseline_acc:.3f} ({baseline_acc*100:.1f}%), AUC: {baseline_auc:.3f}")
    print(f"Logistic Regr.     → Acc: {lr_acc:.3f} ({lr_acc*100:.1f}%), AUC: {lr_auc:.3f}")
    print(f"Random Forest      → Acc: {rf_acc:.3f} ({rf_acc*100:.1f}%), AUC: {rf_auc:.3f}")
    print("="*60)

    # ============================================================
    # RÉSULTATS SECONDAIRES : Analyse par périodes de 5 ans
    # ============================================================
    print("\n\n" + "="*60)
    print("RÉSULTATS SECONDAIRES : Analyse par intervalles de 5 ans")
    print("="*60)
    
    periods = [
        (2000, 2004),
        (2005, 2009),
        (2010, 2014),
        (2015, 2019),
        (2020, 2024)
    ]

    for start_year, end_year in periods:
        print(f"\n{'─'*60}")
        print(f"Période {start_year}-{end_year}")
        print(f"{'─'*60}")
        period_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
        if len(period_df) < 100:  # Skip if too small
            print(f"⚠️  Données insuffisantes pour {start_year}-{end_year} ({len(period_df)} échantillons)")
            continue
        
        print(f"Échantillons : {len(period_df)}")
        
        X_period = period_df[features].values
        y_period = period_df["top10"].values
        
        X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
            X_period, y_period, test_size=0.2, random_state=42, stratify=y_period
        )
        
        scaler_p = StandardScaler()
        X_train_p = scaler_p.fit_transform(X_train_p)
        X_test_p = scaler_p.transform(X_test_p)
        
        # Baseline
        baseline_pred_p = baseline_predict(X_test_p[:, 0])
        baseline_acc_p, baseline_auc_p = evaluate_model(y_test_p, baseline_pred_p, name=f"Baseline {start_year}-{end_year}", verbose=False)
        
        # Logistic Regression
        lr_model_p = train_logistic_regression(X_train_p, y_train_p)
        lr_pred_p = lr_model_p.predict(X_test_p)
        lr_acc_p, lr_auc_p = evaluate_model(y_test_p, lr_pred_p, name=f"Logistic {start_year}-{end_year}", verbose=False)
        
        # Random Forest
        rf_model_p = train_random_forest(X_train_p, y_train_p)
        rf_pred_p = rf_model_p.predict(X_test_p)
        rf_acc_p, rf_auc_p = evaluate_model(y_test_p, rf_pred_p, name=f"Random Forest {start_year}-{end_year}", verbose=False)
        
        print(f"  Baseline        → Acc: {baseline_acc_p:.3f} ({baseline_acc_p*100:.1f}%), AUC: {baseline_auc_p:.3f}")
        print(f"  Logistic Regr.  → Acc: {lr_acc_p:.3f} ({lr_acc_p*100:.1f}%), AUC: {lr_auc_p:.3f}")
        print(f"  Random Forest   → Acc: {rf_acc_p:.3f} ({rf_acc_p*100:.1f}%), AUC: {rf_auc_p:.3f}")
    
    print("\n" + "="*60)
    print("Analyse terminée")
    print("="*60)


if __name__ == "__main__":
    main()
