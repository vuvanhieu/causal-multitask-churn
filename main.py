# main.py
import os
import pandas as pd
import numpy as np
import shutil
import datetime
from configs import OUTPUT_DIR, get_output_subdir, create_directories
from data_ultils import load_clean_data_multitask, apply_balancing_all_tasks
from models import build_multitask_dnn
from training import (
    train_multitask_model,
    train_baseline_models,
)
from evaluations import (
    plot_label_distribution,
    plot_training_history,
    save_predictions_with_explanations,
    generate_churn_reason_statistics,
    plot_model_comparison,
    run_churn_risk_alert_pipeline,
    plot_high_risk_reason_distribution,
    export_compare_results_to_csv,
    reason_to_features,
    infer_reason,
    remove_top_right_spines,
    plot_global_feature_importance,
    plot_jaccard_boxplot,
    plot_shap_summary,
    plot_waterfall
)
from explainers import (
    explain_multitask_churn_shap,
    explain_multitask_churn_lime,
    explain_multitask_churn_shap_global
)
from collections import Counter


def backup_code(run_id):
    """
    Sao lưu code vào thư mục backup, tên thư mục trùng với tên thư mục kết quả (run_id)
    """
    backup_dir = os.path.join(OUTPUT_DIR, "backup", run_id)
    os.makedirs(backup_dir, exist_ok=True)

    project_root = os.path.dirname(os.path.abspath(__file__))

    for root, _, files in os.walk(project_root):
        # bỏ qua thư mục backup
        if os.path.abspath(root).startswith(os.path.abspath(os.path.join(OUTPUT_DIR, "backup"))):
            continue
        for f in files:
            if f.endswith(".py"):
                src = os.path.join(root, f)
                rel = os.path.relpath(root, project_root)
                dst_dir = os.path.join(backup_dir, rel)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(src, os.path.join(dst_dir, f))

    print(f"[DEBUG] Backup code -> {backup_dir}")


def main():
    # Tạo run_id cho thư mục kết quả và backup
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{now}"
    backup_code(run_id)

    # Thư mục kết quả chính nằm trong OUTPUT, đồng bộ với backup
    result_folder = get_output_subdir(run_id)

    # Đường dẫn file dữ liệu
    filepath = os.path.join(os.getcwd(), 'Bank_Customer_Churn.csv')

    df_raw = pd.read_csv(filepath)
    print(df_raw['Exited'].value_counts(normalize=True))

    # Load & chuẩn hóa dữ liệu
    X, y_dict, df_full = load_clean_data_multitask(filepath)

    # Trước cân bằng
    plot_label_distribution(y_dict, save_dir=result_folder, prefix="before")

    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train_full, X_test, y_churn_train_full, y_churn_test, y_score_train_full, y_score_test, y_balance_train_full, y_balance_test = train_test_split(
        X, y_dict['churn'], y_dict['score'], y_dict['balance'],
        test_size=0.2, random_state=42, stratify=y_dict['churn']
    )

    # Tách val từ train
    X_train, X_val, y_churn_train, y_churn_val, y_score_train, y_score_val, y_balance_train, y_balance_val = train_test_split(
        X_train_full, y_churn_train_full, y_score_train_full, y_balance_train_full,
        test_size=0.2, random_state=42, stratify=y_churn_train_full
    )

    y_train_dict = {
        'churn': y_churn_train,
        'score': y_score_train,
        'balance': y_balance_train
    }
    y_val_dict = {
        'churn': y_churn_val,
        'score': y_score_val,
        'balance': y_balance_val
    }
    y_test_dict = {
        'churn': y_churn_test,
        'score': y_score_test,
        'balance': y_balance_test
    }

    # Cân bằng dữ liệu trên tập train (đa nhiệm)
    X_train, y_train_dict = apply_balancing_all_tasks(X_train, y_train_dict)

    # Sau cân bằng
    plot_label_distribution(
        {
            'churn': y_train_dict['churn'],
            'score': y_train_dict['score'],
            'balance': y_train_dict['balance']
        },
        save_dir=result_folder,
        prefix="after"
    )

    # Metadata tương ứng với X_test (phục vụ giải thích)
    metadata_test = df_full.iloc[len(df_full) - len(X_test):].reset_index(drop=True)

    # ============================
    #  PHẦN 1 – MULTITASK DNN
    # ============================
    multitask_dir = get_output_subdir(run_id, "Multitask_DNN")

    # Grid nhỏ cho ví dụ (có thể mở rộng nếu cần)
    epoch_list = [150]
    batch_size_list = [32]

    all_results = []   # lưu metrics của tất cả cấu hình (ep, bs)
    model_paths = {}   # map (epochs, batch_size) -> file .h5

    for epochs in epoch_list:
        for batch_size in batch_size_list:
            print(f"Training Multitask DNN: epochs={epochs}, batch_size={batch_size}")
            config_folder = get_output_subdir(
                run_id,
                "Multitask_DNN",
                f'ep{epochs}_bs{batch_size}'
            )

            model = build_multitask_dnn(X.shape[1])
            model_path = os.path.join(
                config_folder,
                f"model_e{epochs}_b{batch_size}.h5"
            )

            history, test_metrics = train_multitask_model(
                model=model,
                X_train=X_train,
                y_train_dict=y_train_dict,
                X_val=X_val,
                y_val_dict=y_val_dict,
                X_test=X_test,
                y_test_dict=y_test_dict,
                epochs=epochs,
                batch_size=batch_size,
                save_path=model_path,
                save_dir=config_folder,
                model_name="Multitask DNN"
            )

            # Lưu đường học
            plot_training_history(history, config_folder)

            # Lưu path model + metrics
            model_paths[(epochs, batch_size)] = model_path
            all_results.extend(test_metrics)

            # In nhanh Recall cho churn
            recall_churn = [m for m in test_metrics if m['Task'] == 'churn'][0]['Recall']
            print(f"[Multitask DNN] Recall (churn) Epochs={epochs}, Batch Size={batch_size}: {recall_churn:.4f}")

    # Lưu toàn bộ kết quả grid search của Multitask DNN
    df_all_results = pd.DataFrame(all_results)
    csv_out_path = os.path.join(multitask_dir, "gridsearch_multitask_results.csv")
    df_all_results.to_csv(csv_out_path, index=False)
    print(f"Saved Multitask grid search results to: {csv_out_path}")

    # Chọn mô hình tốt nhất theo F1 + Recall cho churn
    df_churn = df_all_results[df_all_results['Task'] == 'churn'].copy()
    df_churn['Score'] = df_churn['F1-score'] + df_churn['Recall']
    best_row = df_churn.loc[df_churn['Score'].idxmax()]
    best_epochs = int(best_row['Epochs'])
    best_batch = int(best_row['Batch Size'])
    best_model_path = model_paths[(best_epochs, best_batch)]

    print(f"[Multitask DNN] Best model: Epochs={best_epochs}, Batch Size={best_batch}")

    import tensorflow as tf
    best_model = tf.keras.models.load_model(best_model_path)
    best_model.summary()

    # Dự đoán test bằng best Multitask model
    y_pred = best_model.predict(X_test)
    y_churn = (y_pred[0] > 0.5).astype(int).flatten()
    y_score = y_pred[1].argmax(axis=1)
    y_balance = (y_pred[2] > 0.5).astype(int).flatten()

    # Lưu giải thích churn
    explanation_path = os.path.join(multitask_dir, "explanation_results.csv")
    save_predictions_with_explanations(
        y_churn, y_score, y_balance, metadata_test, explanation_path
    )
    generate_churn_reason_statistics(explanation_path, multitask_dir)

    # Lọc metrics của best Multitask DNN cho cả 3 task
    mlt_best_results = [
        r for r in all_results
        if r['Epochs'] == best_epochs and r['Batch Size'] == best_batch
    ]

    # ============================
    #  PHẦN 5 – SO SÁNH GIẢI THÍCH (RULE-BASED vs SHAP vs LIME)
    # ============================
    explain_comparison_dir = get_output_subdir(run_id, "Explanation_Comparison")
    os.makedirs(explain_comparison_dir, exist_ok=True)

    # 5.1. Chuẩn bị dữ liệu nền cho SHAP (lấy 100 mẫu ngẫu nhiên từ X_train)
    background_indices = np.random.choice(len(X_train), size=min(100, len(X_train)), replace=False)
    X_background = X_train[background_indices]

    # 5.2. Danh sách feature names (bỏ các cột target)
    feature_names = df_full.drop(['Exited', 'CreditScoreClass', 'HighBalanceFlag'], axis=1).columns.tolist()
    print("Feature names:", feature_names)

    # 5.3. Chọn mẫu test cần giải thích: 30 khách hàng được dự đoán là churn
    y_pred_churn = y_churn  # từ best_model.predict(X_test)
    churn_pred_indices = np.where(y_pred_churn == 1)[0]

    if len(churn_pred_indices) == 0:
        print("⚠️ Không có mẫu churn nào trong test set, bỏ qua so sánh.")
    else:
        sample_size = min(30, len(churn_pred_indices))
        selected_idx = np.random.choice(churn_pred_indices, size=sample_size, replace=False)
        sample_instances = X_test[selected_idx]
        sample_metadata = metadata_test.iloc[selected_idx]

        comparison_rows = []

        for i, inst in enumerate(sample_instances):
            inst = inst.reshape(1, -1)
            row = sample_metadata.iloc[i]

            # ----- Rule-based -----
            reason_str = infer_reason(row)
            rule_features = reason_to_features(reason_str)

            # ----- SHAP -----
            try:
                shap_res = explain_multitask_churn_shap(
                    best_model, X_background, inst, feature_names
                )
                shap_features = set(shap_res['top_names'])
            except Exception as e:
                print(f"SHAP error on sample {i}: {e}")
                shap_res = None
                shap_features = set()

            # ----- LIME -----
            try:
                lime_res = explain_multitask_churn_lime(
                    best_model, X_train, inst, feature_names
                )
                lime_features = set(lime_res['top_names'])
            except Exception as e:
                print(f"LIME error on sample {i}: {e}")
                lime_res = None
                lime_features = set()

            # ----- Tính Jaccard similarity -----
            jaccard_rule_shap = len(rule_features & shap_features) / len(rule_features | shap_features) if (rule_features or shap_features) else 1.0
            jaccard_rule_lime = len(rule_features & lime_features) / len(rule_features | lime_features) if (rule_features or lime_features) else 1.0
            jaccard_shap_lime = len(shap_features & lime_features) / len(shap_features | lime_features) if (shap_features or lime_features) else 1.0

            comparison_rows.append({
                'sample_id': i,
                'true_churn': row.get('Exited', np.nan),
                'pred_churn': 1,
                'rule_reason': reason_str,
                'rule_features': ', '.join(sorted(rule_features)),
                'shap_top5': ', '.join(shap_res['top_names']) if shap_res else '',
                'lime_top5': ', '.join(lime_res['top_names']) if lime_res else '',
                'jaccard_rule_shap': jaccard_rule_shap,
                'jaccard_rule_lime': jaccard_rule_lime,
                'jaccard_shap_lime': jaccard_shap_lime,
            })

        # Lưu bảng so sánh chi tiết
        df_comp = pd.DataFrame(comparison_rows)
        comp_csv_path = os.path.join(explain_comparison_dir, "explanation_comparison.csv")
        df_comp.to_csv(comp_csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ Saved explanation comparison to: {comp_csv_path}")

        # 5.4. Thống kê tổng hợp
        print("\n=== 📊 THỐNG KÊ SO SÁNH GIẢI THÍCH ===")
        print(f"Jaccard (Rule vs SHAP) - Mean: {df_comp['jaccard_rule_shap'].mean():.3f}, Std: {df_comp['jaccard_rule_shap'].std():.3f}")
        print(f"Jaccard (Rule vs LIME) - Mean: {df_comp['jaccard_rule_lime'].mean():.3f}, Std: {df_comp['jaccard_rule_lime'].std():.3f}")
        print(f"Jaccard (SHAP vs LIME) - Mean: {df_comp['jaccard_shap_lime'].mean():.3f}, Std: {df_comp['jaccard_shap_lime'].std():.3f}")

        # Lưu bảng thống kê
        summary_stats = df_comp[['jaccard_rule_shap', 'jaccard_rule_lime', 'jaccard_shap_lime']].describe()
        summary_stats.to_csv(os.path.join(explain_comparison_dir, "similarity_summary.csv"))

        # 5.5. Vẽ biểu đồ cơ bản
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Heatmap Jaccard cho từng mẫu
        plt.figure(figsize=(10, 8))
        jaccard_matrix = df_comp[['jaccard_rule_shap', 'jaccard_rule_lime', 'jaccard_shap_lime']].T
        sns.heatmap(jaccard_matrix, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Jaccard Similarity'})
        plt.xlabel('Sample index')
        plt.ylabel('Comparison')
        plt.yticks([0.5, 1.5, 2.5], ['Rule vs SHAP', 'Rule vs LIME', 'SHAP vs LIME'], rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(explain_comparison_dir, "jaccard_heatmap.png"), dpi=300)
        plt.close()

        # Bar chart tần suất xuất hiện feature trong top-5 của mỗi phương pháp
        def count_features_from_set_list(feature_sets_list):
            counter = Counter()
            for s in feature_sets_list:
                counter.update(s)
            return counter

        rule_feature_sets = [set(f.split(', ') if f else []) for f in df_comp['rule_features']]
        shap_feature_sets = [set(f.split(', ') if f else []) for f in df_comp['shap_top5']]
        lime_feature_sets = [set(f.split(', ') if f else []) for f in df_comp['lime_top5']]

        rule_counter = count_features_from_set_list(rule_feature_sets)
        shap_counter = count_features_from_set_list(shap_feature_sets)
        lime_counter = count_features_from_set_list(lime_feature_sets)

        feature_counts_df = pd.DataFrame({
            'Rule': rule_counter,
            'SHAP': shap_counter,
            'LIME': lime_counter
        }).fillna(0).astype(int)

        feature_counts_df['Total'] = feature_counts_df.sum(axis=1)
        feature_counts_df = feature_counts_df.sort_values('Total', ascending=False).drop('Total', axis=1)

        ax = feature_counts_df.plot(kind='bar', figsize=(10, 6), color=['#2ecc71', '#3498db', '#e74c3c'])
        remove_top_right_spines(ax)
        plt.xlabel('Feature')
        plt.ylabel('Frequency in Top-5')
        plt.legend(title='Method')
        plt.tight_layout()
        plt.savefig(os.path.join(explain_comparison_dir, "feature_frequency.png"), dpi=300)
        plt.close()

        # --------------------------------------------------------
        #  BỔ SUNG: PHÂN TÍCH TOÀN CỤC VÀ CÁC BIỂU ĐỒ NÂNG CAO
        # --------------------------------------------------------
        # 5.6. Tính SHAP cho toàn bộ tập test (dùng cho global analysis)
        try:
            shap_values, expected_value = explain_multitask_churn_shap_global(
                best_model, X_background, X_test, feature_names
            )
            # Lưu shap_values để dùng lại
            np.save(os.path.join(explain_comparison_dir, "shap_values_test.npy"), shap_values)
            np.save(os.path.join(explain_comparison_dir, "expected_value.npy"), expected_value)

            # 5.7. Global feature importance
            # Tần suất rule từ df_comp
            rule_freq_global = df_comp['rule_features'].str.split(', ').explode().value_counts()
            # Tần suất LIME từ df_comp
            lime_freq_global = df_comp['lime_top5'].str.split(', ').explode().value_counts()
            # Trung bình |SHAP|
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

            plot_global_feature_importance(
                mean_abs_shap, rule_freq_global, lime_freq_global,
                feature_names,
                os.path.join(explain_comparison_dir, "global_feature_importance.png")
            )

            # 5.8. Boxplot Jaccard
            plot_jaccard_boxplot(
                df_comp,
                os.path.join(explain_comparison_dir, "jaccard_boxplot.png")
            )

            # 5.9. SHAP summary plot (beeswarm)
            plot_shap_summary(
                shap_values, X_test, feature_names,
                os.path.join(explain_comparison_dir, "shap_summary.png")
            )

            # 5.10. Waterfall cho 2 mẫu: đồng thuận cao nhất và thấp nhất (dựa trên jaccard_rule_shap)
            if not df_comp.empty:
                # Mẫu có Jaccard Rule-SHAP cao nhất
                high_idx = df_comp.loc[df_comp['jaccard_rule_shap'].idxmax(), 'sample_id']
                plot_waterfall(
                    shap_values, expected_value, high_idx, feature_names,
                    os.path.join(explain_comparison_dir, f"waterfall_sample_{high_idx}_high.png")
                )
                # Mẫu có Jaccard Rule-SHAP thấp nhất
                low_idx = df_comp.loc[df_comp['jaccard_rule_shap'].idxmin(), 'sample_id']
                plot_waterfall(
                    shap_values, expected_value, low_idx, feature_names,
                    os.path.join(explain_comparison_dir, f"waterfall_sample_{low_idx}_low.png")
                )

        except Exception as e:
            print(f"⚠️ Lỗi khi thực hiện phân tích SHAP toàn cục: {e}")
            import traceback
            traceback.print_exc()

    # ============================
    #  KẾT THÚC PHẦN 5
    # ============================

    # ============================
    #  PHẦN 2 – BASELINE MODELS
    # ============================
    baseline_dir = get_output_subdir(run_id, "Baselines")
    baseline_results = []

    for task in ['churn', 'score', 'balance']:
        baseline_results.extend(
            train_baseline_models(
                X_train, X_test,
                y_train=y_train_dict[task],
                y_test=y_test_dict[task],
                task=task,
                save_dir=baseline_dir
            )
        )

    df_baselines = pd.DataFrame(baseline_results)
    baseline_csv_path = os.path.join(baseline_dir, "baseline_comparison.csv")
    df_baselines.to_csv(baseline_csv_path, index=False)
    print(f"Saved baselines comparison to: {baseline_csv_path}")

    # ============================
    #  PHẦN 3 – BẢNG SO SÁNH CSV
    # ============================
    compare_csv_path = os.path.join(result_folder, "compare_models_summary.csv")
    df_compare = export_compare_results_to_csv(
        mlt_results=mlt_best_results,
        baseline_results=baseline_results,
        output_path=compare_csv_path
    )

    # Vẽ biểu đồ so sánh (dùng bảng đã chuẩn hoá df_compare)
    for metric in ['Accuracy', 'F1-score', 'Precision', 'Recall']:
        plot_model_comparison(df_compare, metric=metric, save_dir=result_folder)

    # ============================
    #  PHẦN 4 – PIPELINE CẢNH BÁO CHURN (RF)
    # ============================
    run_churn_risk_alert_pipeline(filepath, result_folder)
    plot_high_risk_reason_distribution(
        csv_path=os.path.join(
            result_folder,
            "churn_risk_alert",
            "high_risk_customers_with_reasons.csv"
        ),
        output_path=os.path.join(
            result_folder,
            "churn_risk_alert",
            "high_risk_reason_barplot.png"
        )
    )


if __name__ == '__main__':
    main()