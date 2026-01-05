from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate_model(y_test, y_pred, name="Model", verbose=True):
    """Evaluates a model using accuracy and AUC."""
    acc = accuracy_score(y_test, y_pred)

    try:
        auc = roc_auc_score(y_test, y_pred)
    except:
        auc = 0.0  # fallback if predictions are not probabilistic

    if verbose:
        print(f"{name} → Acc: {acc:.3f}, AUC: {auc:.3f}")
    return acc, auc

