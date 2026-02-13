# evaluations.py
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve, roc_auc_score,
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==== Import configs (threshold + font sizes) ====
from configs import (
    LOW_CREDIT_THRESHOLD, HIGH_CREDIT_THRESHOLD,
    LOW_BALANCE_THRESHOLD, HIGH_BALANCE_THRESHOLD,
    NEW_CUSTOMER_TENURE, SENIOR_AGE_THRESHOLD,

    BASE_FONT_SIZE, TITLE_FONT_SIZE, LABEL_FONT_SIZE,
    TICK_FONT_SIZE, LEGEND_FONT_SIZE
)

# ==== Global Plot Font Config ====
plt.rcParams['font.size'] = BASE_FONT_SIZE
plt.rcParams['axes.titlesize'] = TITLE_FONT_SIZE
plt.rcParams['axes.labelsize'] = LABEL_FONT_SIZE
plt.rcParams['xtick.labelsize'] = TICK_FONT_SIZE
plt.rcParams['ytick.labelsize'] = TICK_FONT_SIZE
plt.rcParams['legend.fontsize'] = LEGEND_FONT_SIZE

sns.set_context("notebook", rc={
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.labelsize": LABEL_FONT_SIZE,
    "xtick.labelsize": TICK_FONT_SIZE,
    "ytick.labelsize": TICK_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
})


def remove_top_right_spines(ax):
    """Remove top and right borders for a cleaner look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def get_display_labels(task_name, labels):
    """
    Chuyển nhãn số sang nhãn cụ thể cho từng task.
    Nếu không phải các task chuẩn, giữ nguyên dạng chuỗi của nhãn.
    """
    name = str(task_name).lower()

    if name == "churn":
        mapping = {
            0: "No churn",
            1: "Churn",
        }
    elif "score" in name:
        # CreditScoreClass: 0,1,2
        mapping = {
            0: "Low score",
            1: "Medium score",
            2: "High score",
        }
    elif "balance" in name:
        # HighBalanceFlag: 0,1
        mapping = {
            0: "Low balance",
            1: "High balance",
        }
    else:
        mapping = {}

    display_labels = []
    for lb in labels:
        try:
            lb_int = int(lb)
        except Exception:
            lb_int = lb
        display_labels.append(mapping.get(lb_int, str(lb)))
    return display_labels


# ============================================================
#  PLOT CONFUSION MATRIX + ROC + PR + AUC
# ============================================================
def plot_all(y_true, y_pred, y_proba, task_name, save_dir, model_name=""):
    os.makedirs(save_dir, exist_ok=True)
    postfix = f"_{model_name}" if model_name else ""

    # One-hot encoding → Label indices
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)

    # Normalize proba to shape Nx2 for binary
    if y_proba.ndim == 1:
        y_proba = np.vstack([1 - y_proba, y_proba]).T
    elif y_proba.ndim == 2 and y_proba.shape[1] == 1:
        y_proba = np.hstack([1 - y_proba, y_proba])

    labels = np.unique(np.concatenate([y_true, y_pred]))
    display_labels = get_display_labels(task_name, labels)

    # ---------------- Confusion Matrix ----------------
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=display_labels, columns=display_labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    ax = plt.gca()
    remove_top_right_spines(ax)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{task_name}_confusion_matrix{postfix}.png"), dpi=300)
    plt.close()

    # ---------------- Confusion Matrix Normalized ----------------
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    df_cm_norm = pd.DataFrame(cm_norm, index=display_labels, columns=display_labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm_norm, annot=True, fmt=".2f", cmap="YlGnBu")
    ax = plt.gca()
    remove_top_right_spines(ax)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{task_name}_confusion_matrix_normalized{postfix}.png"), dpi=300)
    plt.close()

    # --------------------------------------------------
    #  Skip ROC/PR when only 1 class
    # --------------------------------------------------
    if len(np.unique(y_true)) < 2:
        print(f"Skip ROC/PR/AUC for '{task_name}' because there is only one class.")
        return

    # ---------------- ROC Curve ----------------
    plt.figure(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    try:
        if y_proba.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[1], lw=2, label=f"AUC = {roc_auc:.2f}")
        else:
            y_true_bin = label_binarize(y_true, classes=labels)
            for i, lb in enumerate(labels):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                auc_score = auc(fpr, tpr)
                class_name = display_labels[i]
                plt.plot(
                    fpr,
                    tpr,
                    color=colors[i],
                    lw=2,
                    label=f"{class_name} (AUC={auc_score:.2f})"
                )

        plt.plot([0, 1], [0, 1], 'k--')
        ax = plt.gca()
        remove_top_right_spines(ax)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{task_name}_roc_curve{postfix}.png"), dpi=300)
        plt.close()

    except Exception as e:
        print(f"ROC Error for task '{task_name}': {e}")

    # ---------------- Precision–Recall Curve ----------------
    plt.figure(figsize=(7, 6))
    colors2 = plt.cm.Dark2(np.linspace(0, 1, len(labels)))

    try:
        if y_proba.shape[1] == 2:
            prec, rec, _ = precision_recall_curve(y_true, y_proba[:, 1])
            pr_auc = auc(rec, prec)
            plt.plot(rec, prec, color=colors2[1], lw=2, label=f"AUC = {pr_auc:.2f}")
        else:
            y_bin = label_binarize(y_true, classes=labels)
            for i, lb in enumerate(labels):
                prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
                pr_auc = auc(rec, prec)
                class_name = display_labels[i]
                plt.plot(
                    rec,
                    prec,
                    color=colors2[i],
                    lw=2,
                    label=f"{class_name} (AUC={pr_auc:.2f})"
                )

        ax = plt.gca()
        remove_top_right_spines(ax)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{task_name}_pr_curve{postfix}.png"), dpi=300)
        plt.close()

    except Exception as e:
        print(f"PR Error for task '{task_name}': {e}")

    # ---------------- AUC Score File ----------------
    try:
        if y_proba.shape[1] == 2:
            auc_score_val = roc_auc_score(y_true, y_proba[:, 1])
        else:
            y_bin = label_binarize(y_true, classes=labels)
            auc_score_val = roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')

        with open(os.path.join(save_dir, f"{task_name}_AUC{postfix}.txt"), "w") as f:
            f.write(f"AUC ({task_name} - {model_name}): {auc_score_val:.4f}\n")
    except Exception as e:
        print(f"AUC file write error for task '{task_name}': {e}")


# ============================================================
#  PLOT LABEL DISTRIBUTION
# ============================================================
def plot_label_distribution(y_dict, save_dir=".", prefix=""):
    os.makedirs(save_dir, exist_ok=True)

    for task_name, y in y_dict.items():
        if y.ndim == 2:
            y = np.argmax(y, axis=1)

        label_counts = pd.Series(y).value_counts().sort_index()
        percentages = label_counts / len(y) * 100

        raw_labels = label_counts.index.values
        display_labels = get_display_labels(task_name, raw_labels)

        plt.figure(figsize=(6, 4))
        ax = sns.barplot(
            x=raw_labels,
            y=label_counts.values,
            palette="Spectral"
        )
        remove_top_right_spines(ax)

        # set nhãn text cho tick
        ax.set_xticklabels(display_labels)

        # show label on bars
        for i, count in enumerate(label_counts.values):
            ax.text(
                i,
                count + 0.5,
                f"{count} ({percentages.iloc[i]:.1f}%)",
                ha='center',
                va='bottom',
                fontsize=TICK_FONT_SIZE
            )

        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()

        tag = f"{prefix}_{task_name}" if prefix else task_name
        out_path = os.path.join(save_dir, f"{tag}_distribution.png")
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"Saved label distribution: {out_path}")


# ============================================================
#  HIGH-RISK CHURN REASONS (RULE-BASED)
# ============================================================
def infer_reason(row):
    reasons = []

    if row["CreditScore"] < LOW_CREDIT_THRESHOLD:
        reasons.append("Low credit score")
    elif row["CreditScore"] > HIGH_CREDIT_THRESHOLD:
        reasons.append("High credit score")

    if row["Balance"] > HIGH_BALANCE_THRESHOLD:
        reasons.append("High balance")
    elif row["Balance"] < LOW_BALANCE_THRESHOLD:
        reasons.append("Very low balance")

    if row["Tenure"] <= NEW_CUSTOMER_TENURE:
        reasons.append("New customer")

    if row["Age"] >= SENIOR_AGE_THRESHOLD:
        reasons.append("Older customer")

    return ", ".join(reasons)


def generate_high_risk_recommendations(csv_in, csv_out):
    df = pd.read_csv(csv_in)
    high_df = df[df["Risk_Level"] == "High"].copy()
    high_df["Churn_Reason_Suggestion"] = high_df.apply(infer_reason, axis=1)
    high_df.to_csv(csv_out, index=False)
    print(f"Saved high-risk recommendations: {csv_out}")


def plot_high_risk_reason_distribution(csv_path, output_path):
    df = pd.read_csv(csv_path)

    if "Churn_Reason_Suggestion" not in df.columns:
        print("Missing Churn_Reason_Suggestion column.")
        return

    reason_list = df["Churn_Reason_Suggestion"].dropna().str.split(", ")
    counter = Counter()
    for reasons in reason_list:
        counter.update(reasons)

    reason_df = pd.DataFrame(counter.items(), columns=["Reason", "Count"])
    reason_df = reason_df.sort_values("Count", ascending=False)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=reason_df,
        x="Count",
        y="Reason",
        palette="viridis"
    )
    remove_top_right_spines(ax)

    plt.xlabel("Frequency")
    plt.ylabel("Churn Reason")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved high-risk churn reason plot: {output_path}")


# ============================================================
#  TRAINING HISTORY
# ============================================================
def plot_training_history(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # ----- Accuracy -----
    plt.figure(figsize=(10, 5))
    for key in history.history:
        if "accuracy" in key:
            plt.plot(history.history[key], lw=2)

    ax = plt.gca()
    remove_top_right_spines(ax)

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(history.history.keys(), loc="center left", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    acc_path = os.path.join(out_dir, "training_validation_accuracy.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()

    # ----- Loss -----
    plt.figure(figsize=(10, 5))
    for key in history.history:
        if "loss" in key:
            plt.plot(history.history[key], lw=2)

    ax = plt.gca()
    remove_top_right_spines(ax)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(history.history.keys(), loc="center left", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    loss_path = os.path.join(out_dir, "training_validation_loss.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()

    return acc_path, loss_path


# ============================================================
#  SIMPLE EXPLANATIONS (MULTITASK PREDICTIONS)
# ============================================================
def generate_simple_explanations(y_churn, y_score, y_balance, metadata_df, max_samples=None):
    out = []
    for i in range(len(y_churn)):
        if y_churn[i] == 1:
            rs = []
            if y_score[i] == 0:
                rs.append("low credit score")
            elif y_score[i] == 2:
                rs.append("high credit score")
            rs.append("high balance" if y_balance[i] == 1 else "low balance")

            text = (
                f"Churn = 1 -> {', '.join(rs)} | "
                f"Age: {metadata_df.iloc[i]['Age']}, "
                f"Country: {metadata_df.iloc[i]['Geography']}"
            )
        else:
            text = "No churn"
        out.append(text)
        if max_samples and len(out) >= max_samples:
            break
    return out


def save_predictions_with_explanations(y_churn, y_score, y_balance, metadata_df, output_path):
    expl = generate_simple_explanations(y_churn, y_score, y_balance, metadata_df)
    df_exp = metadata_df.copy()
    df_exp["Pred_Exited"] = y_churn
    df_exp["Pred_ScoreClass"] = y_score
    df_exp["Pred_HighBalance"] = y_balance
    df_exp["Explanation"] = expl
    df_exp.to_csv(output_path, index=False)
    print(f"Saved explanations: {output_path}")


# ============================================================
#  MODEL COMPARISON PLOT
# ============================================================
def plot_model_comparison(df_compare, metric='F1-score', save_dir="."):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_compare,
        x="Task",
        y=metric,
        hue="Model",
        palette="Accent"
    )
    remove_top_right_spines(ax)

    # add value labels on top of each bar (vertical)
    for c in ax.containers:
        ax.bar_label(
            c,
            fmt="%.2f",
            padding=3,
            fontsize=TICK_FONT_SIZE,
            label_type="edge",
            rotation=90,
            color="black"
        )

    plt.ylabel(metric)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()

    file_path = os.path.join(save_dir, f"compare_{metric.lower()}.png")
    plt.savefig(file_path, dpi=300)
    plt.close()


# ============================================================
#  CHURN REASON STATISTICS FROM EXPLANATION CSV
# ============================================================
def generate_churn_reason_statistics(explanation_csv_path, output_dir="."):
    """
    Đọc file explanation_results.csv (từ save_predictions_with_explanations),
    trích xuất các lý do trong cột 'Explanation' cho các khách hàng churn,
    thống kê tần suất và vẽ barplot.
    """
    if not os.path.exists(explanation_csv_path):
        print(f"Explanation file not found: {explanation_csv_path}")
        return

    df = pd.read_csv(explanation_csv_path)

    if "Pred_Exited" not in df.columns or "Explanation" not in df.columns:
        print("Missing 'Pred_Exited' or 'Explanation' column.")
        return

    df_churn = df[df["Pred_Exited"] == 1].copy()
    if df_churn.empty:
        print("No churn samples found in explanation file.")
        return

    reason_counter = Counter()

    for text in df_churn["Explanation"].dropna():
        # Format: "Churn = 1 -> reason1, reason2 | Age: ..., Country: ..."
        # Lấy phần sau "->" và trước dấu "|"
        if "->" in text:
            reason_part = text.split("->", 1)[1]
        else:
            reason_part = text

        if "|" in reason_part:
            reason_part = reason_part.split("|", 1)[0]

        reasons = [r.strip() for r in reason_part.split(",") if r.strip()]
        reason_counter.update(reasons)

    if not reason_counter:
        print("No reasons extracted from explanations.")
        return

    reason_df = pd.DataFrame(reason_counter.items(), columns=["Reason", "Count"])
    reason_df = reason_df.sort_values("Count", ascending=False)

    os.makedirs(output_dir, exist_ok=True)

    csv_out = os.path.join(output_dir, "churn_reason_statistics.csv")
    fig_out = os.path.join(output_dir, "churn_reason_barplot.png")

    reason_df.to_csv(csv_out, index=False)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=reason_df,
        x="Count",
        y="Reason",
        palette="magma"
    )
    remove_top_right_spines(ax)

    plt.xlabel("Count")
    plt.ylabel("Reason")
    plt.tight_layout()
    plt.savefig(fig_out, dpi=300)
    plt.close()

    print(f"Saved churn reason statistics: {csv_out}")
    print(f"Saved churn reason barplot: {fig_out}")


# ============================================================
#  CHURN RISK ALERT PIPELINE (BASELINE RF)
# ============================================================
def classify_alert(score):
    """
    Quy đổi điểm rủi ro (0–100) sang mức cảnh báo.
    """
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"


def run_churn_risk_alert_pipeline(filepath, result_folder):
    """
    Pipeline baseline cảnh báo churn:
      - Train RandomForest trên toàn bộ features (encode Geog/Gender).
      - Tính xác suất churn trên test set.
      - Chuyển xác suất -> Risk_Score (0–100) -> Risk_Level (Low/Medium/High).
      - Lưu log + classification report + phân bố Risk_Level.
      - Gọi generate_high_risk_recommendations() để xuất file high_risk_customers_with_reasons.csv.
    """
    if not os.path.exists(filepath):
        print(f"Data file not found: {filepath}")
        return

    df = pd.read_csv(filepath)

    # Encode categorical (nếu còn dạng chuỗi)
    df_enc = df.copy()
    if df_enc["Geography"].dtype == object:
        df_enc["Geography"] = pd.factorize(df_enc["Geography"])[0]
    if df_enc["Gender"].dtype == object:
        df_enc["Gender"] = pd.factorize(df_enc["Gender"])[0]

    # Thêm một số feature phụ trợ giống load_clean_data_multitask
    df_enc["CreditScoreClass"] = pd.cut(
        df_enc["CreditScore"],
        bins=[0, 580, 700, 850],
        labels=[0, 1, 2]
    ).astype(int)
    df_enc["HighBalanceFlag"] = (df_enc["Balance"] > HIGH_BALANCE_THRESHOLD).astype(int)

    # X, y
    X = df_enc.drop(["RowNumber", "CustomerId", "Surname", "Exited"], axis=1)
    y = df_enc["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train baseline RF
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    churn_risk_score = np.round(probs * 100).astype(int)
    alert_levels = [classify_alert(score) for score in churn_risk_score]

    # Chuẩn bị thư mục output
    output_dir = os.path.join(result_folder, "churn_risk_alert")
    os.makedirs(output_dir, exist_ok=True)

    # Lưu alert log
    alert_df = X_test.copy()
    alert_df["True_Exited"] = y_test.values
    alert_df["Churn_Prob"] = probs
    alert_df["Churn_Risk_Score"] = churn_risk_score
    alert_df["Risk_Level"] = alert_levels

    alert_log_path = os.path.join(output_dir, "churn_risk_alert_log.csv")
    alert_df.to_csv(alert_log_path, index=False)
    print(f"Alert log saved to {alert_log_path}")

    # Classification report
    preds = model.predict(X_test)
    report = classification_report(y_test, preds)
    report_path = os.path.join(output_dir, "churn_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # Phân bố Risk_Level
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(
        data=alert_df,
        x="Risk_Level",
        order=["Low", "Medium", "High"],
        palette="Set2"
    )
    remove_top_right_spines(ax)

    plt.ylabel("Number of Customers")
    plt.xlabel("Risk Level")
    plt.tight_layout()
    barplot_path = os.path.join(output_dir, "churn_risk_level_distribution.png")
    plt.savefig(barplot_path, dpi=300)
    plt.close()
    print(f"Saved risk level distribution plot: {barplot_path}")

    # Sinh file high_risk_customers_with_reasons.csv
    high_risk_path = os.path.join(output_dir, "high_risk_customers_with_reasons.csv")
    generate_high_risk_recommendations(alert_log_path, high_risk_path)


import pandas as pd
import os

def export_compare_results_to_csv(
        mlt_results, 
        baseline_results, 
        output_path="compare_results.csv"
):
    """
    Gộp kết quả từ Multitask DNN và các Baseline models thành 1 bảng CSV.

    Tham số:
        mlt_results      : list[dict] từ train_multitask_model()
        baseline_results : list[dict] từ train_baseline_models()
        output_path      : file CSV đầu ra
    
    Cột xuất ra:
        Task | Model | Accuracy | Precision | Recall | F1-score
    """

    all_rows = []

    # ==== 1. Thêm kết quả Multitask DNN ====
    for r in mlt_results:
        all_rows.append({
            "Task": r["Task"],
            "Model": "Multitask DNN",
            "Accuracy": r["Accuracy"],
            "Precision": r["Precision"],
            "Recall": r["Recall"],
            "F1-score": r["F1-score"]
        })

    # ==== 2. Thêm kết quả Baselines ====
    for r in baseline_results:
        all_rows.append({
            "Task": r["Task"],
            "Model": r["Model"],
            "Accuracy": r["Accuracy"],
            "Precision": r["Precision"],
            "Recall": r["Recall"],
            "F1-score": r["F1-score"]
        })

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved comparison CSV to: {output_path}")
    return df


# Thêm vào cuối file evaluations.py (hoặc bất kỳ đâu phù hợp)

def reason_to_features(reason_str):
    """
    Chuyển chuỗi lý do từ infer_reason thành set các feature names.
    Ví dụ: "Low credit score, High balance" -> {'CreditScore', 'Balance'}
    """
    if not isinstance(reason_str, str) or not reason_str.strip():
        return set()
    
    mapping = {
        'Low credit score': 'CreditScore',
        'High credit score': 'CreditScore',
        'High balance': 'Balance',
        'Very low balance': 'Balance',
        'New customer': 'Tenure',
        'Older customer': 'Age'
    }
    features = set()
    parts = reason_str.split(',')
    for p in parts:
        p = p.strip()
        if p in mapping:
            features.add(mapping[p])
    return features



# ============================================================
#  CÁC HÀM VẼ CHO SO SÁNH GIẢI THÍCH (SHAP, LIME, RULE)
# ============================================================

def plot_global_feature_importance(mean_shap, rule_freq, lime_freq, feature_names, save_path):
    """Vẽ bar chart so sánh tầm quan trọng feature toàn cục."""
    df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP (|value|)': mean_shap / mean_shap.max(),  # chuẩn hoá
        'Rule frequency': rule_freq / rule_freq.max(),
        'LIME frequency': lime_freq / lime_freq.max()
    })
    df = df.melt(id_vars='Feature', var_name='Method', value_name='Normalized Importance')
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x='Feature', y='Normalized Importance', hue='Method')
    remove_top_right_spines(ax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_jaccard_boxplot(df_comp, save_path):
    """Vẽ boxplot cho 3 cặp Jaccard similarity."""
    plt.figure(figsize=(8, 6))
    data = df_comp[['jaccard_rule_shap', 'jaccard_rule_lime', 'jaccard_shap_lime']]
    ax = sns.boxplot(data=data)
    remove_top_right_spines(ax)
    ax.set_xticklabels(['Rule vs SHAP', 'Rule vs LIME', 'SHAP vs LIME'])
    ax.set_ylabel('Jaccard Similarity')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_shap_summary(shap_values, X, feature_names, save_path):
    """Vẽ SHAP summary plot (beeswarm)."""
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_waterfall(shap_values, expected_value, instance_idx, feature_names, save_path):
    """Vẽ waterfall plot cho một mẫu cụ thể."""
    shap.plots.waterfall(shap.Explanation(values=shap_values[instance_idx],
                                          base_values=expected_value,
                                          data=X_test[instance_idx],
                                          feature_names=feature_names),
                         max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    