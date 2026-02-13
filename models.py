# models.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.layers import Concatenate, BatchNormalization
import tensorflow as tf
# ==== Keras Multitask DNN ====
def build_multitask_dnn(input_dim: int) -> tf.keras.Model:
    """
    Mô hình DNN đa nhiệm có nhận thức nhân quả (Causal Multi-Task Learning):
    churn (binary), score (3 lớp), balance (binary), có nhánh mô phỏng can thiệp.
    """
    inputs = Input(shape=(input_dim,))
    # Neural feature extractor
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)

    # Causal intervention simulation branch
    # Intervention 1: Credit score improvement (giả sử đặc trưng 3 là credit score)
    intervention1 = Dense(32, activation='relu')(x)
    intervention1 = Dense(16, activation='relu')(intervention1)
    intervention1 = Dense(1, activation='sigmoid', name='intervention1')(intervention1)

    # Intervention 2: Product recommendation (giả sử đặc trưng 9 là num_products)
    intervention2 = Dense(32, activation='relu')(x)
    intervention2 = Dense(16, activation='relu')(intervention2)
    intervention2 = Dense(1, activation='sigmoid', name='intervention2')(intervention2)

    interventions = Concatenate(name='interventions')([intervention1, intervention2])

    # Factual prediction branch
    factual = Dense(32, activation='relu')(x)

    # Kết hợp factual và hiệu ứng can thiệp cho từng task
    # Churn
    churn_base = Dense(16, activation='relu')(factual)
    churn_out = Dense(1, activation='sigmoid')(churn_base)
    churn_adj = Dense(1, activation='tanh')(interventions)
    out_churn = tf.keras.layers.Add(name='churn')([churn_out, 0.1 * churn_adj])

    # Score
    score_base = Dense(16, activation='relu')(factual)
    out_score = Dense(3, activation='softmax', name='score')(score_base)

    # Balance
    balance_base = Dense(16, activation='relu')(factual)
    balance_out = Dense(1, activation='sigmoid')(balance_base)
    balance_adj = Dense(1, activation='tanh')(interventions)
    out_balance = tf.keras.layers.Add(name='balance')([balance_out, 0.1 * balance_adj])

    model = Model(inputs=inputs, outputs=[out_churn, out_score, out_balance])
    return model


# ==== Baseline models (single-task, scikit-learn) ====
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def build_logistic_regression_baseline():
    """
    Logistic Regression baseline cho từng task đơn.
    Phù hợp cho dữ liệu bảng, hỗ trợ binary / multiclass.
    """
    return LogisticRegression(
        max_iter=1000
    )


def build_naive_bayes_baseline():
    """
    Naive Bayes (GaussianNB) baseline cho từng task đơn.
    Đơn giản, nhanh, làm baseline tham chiếu.
    """
    return GaussianNB()


def build_svm_baseline():
    """
    SVM baseline cho từng task đơn.
    Dùng kernel RBF và probability=True để hỗ trợ ROC/PR.
    """
    return SVC(
        kernel='rbf',
        probability=True,
        random_state=42
    )


def build_mlp_baseline():
    """
    Multilayer Perceptron (MLP) baseline cho từng task đơn.
    """
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42
    )


