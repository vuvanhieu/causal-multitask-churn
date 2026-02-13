
# Causal Multi-Task Learning for Explainable Customer Analytics

## Project Overview
This repository provides CI-MTL, a scientifically rigorous Causal Intervention-Aware Multitask Deep Neural Network for explainable customer analytics on tabular banking data. CI-MTL jointly predicts customer churn, credit-score tier, and high-balance flag, and incorporates an intervention simulation module to generate actionable risk alerts and interpretable outputs.

## Scientific Contributions
- Unified multitask DNN architecture with shared and task-specific representations
- Explicit intervention branch simulating credit-score improvement and personalized recommendations
- Triangulated explainability: rule-based, SHAP, and LIME explanations
- Quantitative analysis of explanation agreement using Jaccard similarity
- Churn risk alert pipeline for operational decision support

## Dataset
Experiments are conducted on the [Bank Customer Churn Dataset](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction), comprising 10,000 customer records and 13 attributes. Target variables include churn, credit-score class, and high-balance flag.

## Installation & Environment
1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/causal-multitask-churn.git
    cd causal-multitask-churn
    ```
2. Create and activate a Python environment (e.g., conda):
    ```bash
    conda create -n dl python=3.9
    conda activate dl
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
Run the main pipeline:
```bash
python main.py
```
Configuration options are available in `configs.py`. All outputs (metrics, figures, explanations, alerts) are saved in the `OUTPUT/` directory.

## Key Files
- `main.py`: Training and evaluation entry point
- `models.py`: Model architecture and routines
- `explainers.py`: Explanation methods (rule-based, SHAP, LIME)
- `training.py`: Training and validation logic
- `evaluations.py`: Performance metrics and analysis
- `data_utils.py`: Data loading and preprocessing
- `decision_tree_thresholds.py`: Threshold logic for credit-score and balance
- `requirements.txt`: Python dependencies

## Results & Reproducibility
- Multitask DNN achieves strong performance (churn F1 = 0.813)
- Churn risk alert pipeline provides actionable customer-level insights
- Explanation agreement quantified via Jaccard similarity
- Figures and tables are available in `OUTPUT/20260213_072446_FINAL/`
- Experiments use stratified train/validation/test splits
- Preprocessing includes label encoding and standardization
- SMOTE and undersampling applied to balance training data

## Hardware
Experiments were performed on Intel i7-10700 CPU, 16 GB RAM, GTX 1060 GPU.

## License
This project is released under the MIT License.

## Citation
If you use this code or results, please cite:
```
@inproceedings{nguyen2026cimtldnn,
  title={A Causal Intervention-Aware Multitask DNN for Explainable Customer Analytics},
  author={Thi-Van Nguyen and Nguyen-Thi Thao and Van-Hieu Vu},
  booktitle={Proceedings of ...},
  year={2026}
}
```

## Contact
- Thi-Van Nguyen: van.nguyenthi1@phenikaa-uni.edu.vn
- Nguyen-Thi Thao: thaont3102@gmail.com
- Van-Hieu Vu: vvhieu@ioit.ac.vn
project/
├── configs.py
├── data_utils.py
├── models.py
├── training.py
├── evaluations.py
├── main.py
└── output/                          # 📁 OUTPUT chính
    └── run_20241215_143022/         # 🕒 Mỗi lần chạy có timestamp riêng
        ├── Multitask_DNN/           # 🤖 Kết quả Multitask DNN
        ├── Baselines/               # 📊 Kết quả baseline models
        ├── Churn_Risk_Alert/        # ⚠️ Kết quả cảnh báo churn
        ├── Plots/                   # 📈 Các biểu đồ
        ├── Models/                  # 💾 Saved models
        ├── Reports/                 # 📄 Báo cáo
        └── Explanations/            # 🔍 Giải thích kết quả