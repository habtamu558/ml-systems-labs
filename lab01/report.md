\# Lab 1 — Baseline Models and System Measurements



\## 1. Goal



The goal of this lab was to set up and verify the machine-learning environment, train two baseline classification models, and measure their accuracy and system-resource characteristics. The models evaluated were Logistic Regression and Random Forest using the scikit-learn breast cancer dataset.



The measurements were used to determine which deployment environments are suitable for each model.



\## 2. Method



The experiment used the `load\_breast\_cancer` dataset from scikit-learn. The dataset contains 569 samples and 30 features.



The data was divided into training and test sets using a 70/30 stratified split with `random\_state=42`. The training set contained 398 samples and the test set contained 171 samples.



Two models were trained:



\* Logistic Regression with `max\_iter=1000` and `random\_state=42`

\* Random Forest with 100 estimators and `random\_state=42`



The random seeds were fixed to 42 for reproducibility.



For system measurements, training time was measured using one warm-up run followed by five timed runs, with the median reported. Single-sample inference latency was measured using 100 repetitions after a warm-up run. Model size was measured from the serialized Joblib files. Memory usage was measured during training and inference.



The deployment suitability of each model was compared against the Cloud, Edge, Mobile, and TinyML resource budgets specified in the lab.



\## 3. Results



\### 3.1 Dataset and baseline accuracy



| Model                    | Test Accuracy |

| ------------------------ | ------------: |

| Logistic Regression      |        0.9474 |

| Random Forest Classifier |        0.9357 |



Logistic Regression achieved the higher test accuracy of 0.9474, while Random Forest achieved 0.9357.



\### 3.2 System measurements



| Model                    | Training Time (s) | Inference Latency (ms) | Model Size (KB) | Training Memory Delta (MB) | Inference Memory Delta (MB) |

| ------------------------ | ----------------: | ---------------------: | --------------: | -------------------------: | --------------------------: |

| Logistic Regression      |            0.5980 |                 0.1152 |          1.0303 |                     0.0273 |                      0.0078 |

| Random Forest Classifier |            0.3227 |                 6.4816 |        284.0713 |                     0.0195 |                      0.0000 |



The Random Forest trained faster than Logistic Regression in the measured experiment. However, Logistic Regression had substantially lower inference latency and a much smaller serialized model size.



\### 3.3 Deployment budget



| Model                    | Cloud | Edge | Mobile | TinyML |

| ------------------------ | ----- | ---- | ------ | ------ |

| Logistic Regression      | Yes   | Yes  | Yes    | Yes    |

| Random Forest Classifier | Yes   | Yes  | Yes    | No     |



Logistic Regression satisfies the measured latency and model-size limits for all four deployment categories. Its model size is approximately 1.03 KB and its inference latency is approximately 0.1152 ms.



Random Forest satisfies the latency and model-size limits for Cloud, Edge, and Mobile deployment. However, its model size of approximately 284.07 KB exceeds the TinyML model-size limit of 100 KB, so it does not meet the TinyML requirements.



\## 4. Conclusions



1\. Logistic Regression provided the best classification accuracy in this experiment, achieving 0.9474 compared with 0.9357 for Random Forest.



2\. Logistic Regression was much more efficient for inference and storage. Its model was approximately 1.03 KB with a median single-sample inference latency of 0.1152 ms, while the Random Forest model was approximately 284.07 KB with a latency of 6.4816 ms.



3\. Logistic Regression is the more suitable model for resource-constrained deployment because it satisfies the measured requirements even for TinyML. Random Forest is still suitable for Cloud, Edge, and Mobile deployment, but its model size is too large for the TinyML limit.



