"""
explainers.py
Các hàm giải thích mô hình Multitask DNN bằng SHAP và LIME.
Chỉ tập trung vào task churn (binary classification).
"""

import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# SHAP Explainer cho Multitask DNN (churn head)
# ------------------------------------------------------------
def explain_multitask_churn_shap(model, X_background, X_instance, feature_names):
    """
    model: Keras multitask model đã train.
    X_background: numpy array, mẫu nền để tạo explainer (khoảng 100 mẫu).
    X_instance: numpy array shape (1, n_features) - mẫu cần giải thích.
    feature_names: list tên các feature.

    Returns:
        shap_values: mảng shap values cho mẫu (shape = (n_features,)).
        expected_value: giá trị baseline (kỳ vọng).
        top_features_idx: indices của top-k feature theo |shap| (k=5).
        top_features_names: tên tương ứng.
    """
    # Hàm predict chỉ lấy output churn (đầu ra thứ 0)
    def model_churn(x):
        return model.predict(x, verbose=0)[0]  # shape (n, 1)

    # Dùng KernelExplainer vì model không phải tree-based
    explainer = shap.KernelExplainer(model_churn, X_background)
    shap_values = explainer.shap_values(X_instance, nsamples=200)

    # shap_values là list, phần tử đầu ứng với output class 1? Cần kiểm tra.
    # Với binary classification, KernelExplainer trả về list 2 phần tử? 
    # Thực tế: model_churn trả về xác suất class 1, nên shap_values là mảng 2D (1, n_features).
    # Đôi khi shap trả về list với 2 phần tử (SHAP cho class 0 và class 1). 
    # Ta sẽ xử lý linh hoạt.
    if isinstance(shap_values, list):
        # Lấy shap cho class 1 (index 1)
        shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
    shap_values = shap_values.flatten()

    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1] if len(expected_value) == 2 else expected_value[0]

    # Top k feature theo trị tuyệt đối
    k = 5
    abs_shap = np.abs(shap_values)
    top_idx = np.argsort(abs_shap)[-k:][::-1]
    top_names = [feature_names[i] for i in top_idx]

    return {
        'shap_values': shap_values,
        'expected_value': expected_value,
        'top_indices': top_idx,
        'top_names': top_names,
        'top_values': shap_values[top_idx]
    }


# ------------------------------------------------------------
# LIME Explainer cho Multitask DNN (churn head)
# ------------------------------------------------------------
def explain_multitask_churn_lime(model, X_train, X_instance, feature_names, class_names=['No Churn', 'Churn']):
    """
    model: Keras multitask model.
    X_train: numpy array, dùng để khởi tạo explainer.
    X_instance: numpy array shape (1, n_features) - mẫu cần giải thích.
    feature_names: list tên feature.
    class_names: list tên lớp (mặc định cho churn).

    Returns:
        exp: đối tượng Explanation từ LIME.
        top_features_names: tên top-k feature theo weight.
        top_weights: weight tương ứng.
    """
    # Hàm predict trả về xác suất cho cả 2 lớp (cần cho LIME)
    def predict_proba(x):
        proba_1 = model.predict(x, verbose=0)[0]  # shape (n,1)
        proba_0 = 1 - proba_1
        return np.hstack([proba_0, proba_1])

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        discretize_continuous=False,
        random_state=42
    )

    exp = explainer.explain_instance(
        X_instance[0],
        predict_proba,
        num_features=5,
        top_labels=1
    )

    # Lấy explanation cho label dự đoán (thường là 1 - churn)
    label = exp.top_labels[0]
    feature_weights = exp.as_list(label=label)

    # feature_weights là list các tuple (feature_name, weight)
    # Lưu ý: LIME có thể tạo feature names dạng "feature_name <= value" hoặc "value < feature_name <= ..."
    # Ta cần chuẩn hóa về tên feature gốc.
    top_features = []
    top_weights = []
    for f, w in feature_weights[:5]:
        # Tách lấy tên feature (phần trước dấu cách hoặc dấu <, <=, >, >=)
        import re
        match = re.match(r'^([a-zA-Z_]+)', f)
        if match:
            fname = match.group(1)
        else:
            fname = f
        top_features.append(fname)
        top_weights.append(w)

    return {
        'explanation': exp,
        'top_names': top_features,
        'top_weights': top_weights,
        'label': label
    }


# ------------------------------------------------------------
# Hàm tiện ích: Lấy top feature từ SHAP/LIME dưới dạng set
# ------------------------------------------------------------
def get_top_feature_set(explanation_result, method):
    if method == 'shap':
        return set(explanation_result['top_names'])
    elif method == 'lime':
        return set(explanation_result['top_names'])
    else:
        return set()
    
    
def explain_multitask_churn_shap_global(model, X_background, X_explain, feature_names):
    """
    Tính SHAP values cho nhiều mẫu (X_explain) cùng lúc.
    Trả về:
        shap_values: mảng 2D (n_samples, n_features)
        expected_value: giá trị baseline
    """
    def model_churn(x):
        return model.predict(x, verbose=0)[0]

    explainer = shap.KernelExplainer(model_churn, X_background)
    shap_values = explainer.shap_values(X_explain, nsamples=200)

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1] if len(expected_value) == 2 else expected_value[0]

    return shap_values, expected_value
