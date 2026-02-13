# training.py
import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint

from evaluations import plot_all

from models import (
    build_logistic_regression_baseline,
    build_naive_bayes_baseline,
    build_svm_baseline,
    build_mlp_baseline,
)


def train_multitask_model(model,
                          X_train, y_train_dict,
                          X_val, y_val_dict,
                          X_test, y_test_dict,
                          epochs=30, batch_size=64,
                          save_path=None, save_dir=".",
                          model_name="Multitask DNN"):
    """
    Huấn luyện mô hình đa nhiệm, lưu best model theo val_churn_accuracy,
    và evaluate trên test set (gồm plot_all cho từng task).
    """
    model.compile(
        optimizer='adam',
        loss={
            'churn': 'binary_crossentropy',
            'score': 'categorical_crossentropy',
            'balance': 'binary_crossentropy'
        },
        loss_weights={
            'churn': 1.0,
            'score': 0.5,
            'balance': 0.5
        },
        metrics={
            'churn': ['accuracy', Precision(name='precision'), Recall(name='recall')],
            'score': 'accuracy',
            'balance': 'accuracy'
        }
    )

    callbacks = []
    if save_path:
        checkpoint = ModelCheckpoint(
            filepath=save_path,
            monitor='val_churn_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)

    history = model.fit(
        X_train,
        y_train_dict,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val_dict),
        verbose=1,
        callbacks=callbacks
    )

    if 'val_churn_recall' in history.history:
        best_epoch = int(np.argmax(history.history['val_churn_recall']) + 1)
    else:
        best_epoch = int(np.argmax(history.history['val_churn_accuracy']) + 1)
    print(f"Best Epoch (based on validation): {best_epoch}")

    if save_path and os.path.exists(save_path):
        best_model = tf.keras.models.load_model(save_path)
    else:
        print("Warning: save_path khong ton tai, su dung model hien tai.")
        best_model = model

    y_pred = best_model.predict(X_test)
    y_true_dict = {
        'churn': y_test_dict['churn'],
        'score': y_test_dict['score'].argmax(axis=1),
        'balance': y_test_dict['balance']
    }
    y_pred_dict = {
        'churn': (y_pred[0] > 0.5).astype(int).flatten(),
        'score': y_pred[1].argmax(axis=1),
        'balance': (y_pred[2] > 0.5).astype(int).flatten()
    }

    task_metrics = []
    os.makedirs(save_dir, exist_ok=True)
    report_txt_path = os.path.join(save_dir, "test_classification_report.txt")

    # Ghi file báo cáo với encoding UTF-8 (an toàn nếu sau này có tiếng Việt)
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Best Epoch (val_churn_recall/accuracy): {best_epoch}\n\n")
        for task in ['churn', 'score', 'balance']:
            y_true = y_true_dict[task]
            y_pred_task = y_pred_dict[task]
            y_proba_task = y_pred[0 if task == 'churn' else 1 if task == 'score' else 2]

            acc = accuracy_score(y_true, y_pred_task)
            prec = precision_score(y_true, y_pred_task, average='weighted', zero_division=0)
            rec = recall_score(y_true, y_pred_task, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred_task, average='weighted', zero_division=0)
            cls_report = classification_report(y_true, y_pred_task)

            f.write(f"=== {task.upper()} ===\n")
            f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1-score: {f1:.4f}\n")
            f.write(cls_report + "\n\n")

            task_metrics.append({
                'Epochs': epochs,
                'Batch Size': batch_size,
                'Task': task,
                'Accuracy': round(acc, 4),
                'F1-score': round(f1, 4),
                'Precision': round(prec, 4),
                'Recall': round(rec, 4),
                'Model': model_name
            })

            plot_all(
                y_true=y_true,
                y_pred=y_pred_task,
                y_proba=y_proba_task,
                task_name=task,
                save_dir=save_dir,
                model_name=model_name
            )

    print(f"Saved test report and metrics to: {report_txt_path}")
    return history, task_metrics


def train_baseline_models(X_train, X_test, y_train, y_test, task, save_dir="."):
    """
    Huấn luyện các baseline (Logistic Regression, Naive Bayes, SVM, MLP)
    cho *một* task đơn (churn / score / balance) và vẽ các hình đánh giá.

    Tham số:
        X_train, X_test : numpy arrays
        y_train, y_test : 1D labels hoặc one-hot (nếu multiclass)
        task            : tên task, ví dụ 'churn', 'score', 'balance'
        save_dir        : thư mục gốc để lưu kết quả các baseline

    Trả về:
        results: list[dict] chứa Accuracy, F1, Precision, Recall cho từng model.
    """
    # Nếu nhãn ở dạng one-hot (ví dụ cho CreditScoreClass) → dùng argmax
    if y_train.ndim == 2:
        y_train = y_train.argmax(axis=1)
        y_test = y_test.argmax(axis=1)

    # Khởi tạo các model baseline từ models.py
    models = {
        "Logistic Regression": build_logistic_regression_baseline(),
        "Naive Bayes": build_naive_bayes_baseline(),
        "SVM": build_svm_baseline(),
        "MLP (Single)": build_mlp_baseline(),
    }

    results = []

    # Tạo thư mục con cho từng model baseline
    os.makedirs(save_dir, exist_ok=True)
    subdirs = {}
    for name in models:
        p = os.path.join(save_dir, name.replace(" ", "_"))
        os.makedirs(p, exist_ok=True)
        subdirs[name] = p

    # Train & evaluate từng model
    for name, clf in models.items():
        print(f"[Baseline] Training {name} for task '{task}'")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Lấy y_proba nếu có, nếu không thì dựng tay
        try:
            y_proba = clf.predict_proba(X_test)
            # nếu chỉ có 1 cột (1 class) thì convert thành [P(0), P(1)]
            if y_proba.shape[1] == 1:
                y_proba = np.hstack([1 - y_proba, y_proba])
        except Exception:
            classes = np.unique(y_train)
            y_proba = np.zeros((len(y_pred), len(classes)))
            for i, c in enumerate(classes):
                y_proba[:, i] = (y_pred == c).astype(int)

        # Gọi plot_all để vẽ Confusion Matrix + ROC + PR + AUC
        plot_all(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            task_name=task,
            save_dir=subdirs[name],
            model_name=name.replace(" ", "_")
        )

        # Tính metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)

        results.append({
            "Task": task,
            "Model": name,
            "Accuracy": round(acc, 3),
            "F1-score": round(f1, 3),
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
        })

    return results
