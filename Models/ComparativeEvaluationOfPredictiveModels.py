COMPARATIVE EVALUATION OF PREDICTIVE MODELS

comparison_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=RANDOM_SEED, eval_metric="logloss", n_jobs=-1)
}

comparison_results = {}

for name, model in comparison_models.items():
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    preds = (scores >= 0.50).astype(int)
    comparison_results[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "F1-Score": f1_score(y_test, preds, zero_division=0),
        "AUC-ROC": roc_auc_score(y_test, scores),
        "Average Precision": average_precision_score(y_test, scores)
    }

# Using the previously computed 'test_probabilities' and 'optimal_decision_threshold'
resnet_preds = (test_probabilities >= optimal_decision_threshold).astype(int)
comparison_results["ResNet-Attention"] = {
    "Accuracy": accuracy_score(y_test, resnet_preds),
    "Precision": precision_score(y_test, resnet_preds, zero_division=0),
    "Recall": recall_score(y_test, resnet_preds, zero_division=0),
    "F1-Score": f1_score(y_test, resnet_preds, zero_division=0),
    "AUC-ROC": roc_auc_score(y_test, test_probabilities),
    "Average Precision": average_precision_score(y_test, test_probabilities)
}

df_results = pd.DataFrame(comparison_results).T.round(4)
print(df_results.to_string())

df_results.to_csv(os.path.join(OUTPUT_DIR, "Comparative_Model_Performance.csv"), encoding="utf-8-sig")

log_progress("Phase 13 completed: Comparative evaluation of models executed and exported.")
