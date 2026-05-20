# Predicting 30-Day Hospital Readmission for Patients With Diabetes

## Author: Grzegorz Adamiec

**Executive Summary**

This project examines whether machine learning models can predict 30-day hospital readmission among patients with diabetes. The analysis uses the UCI **Diabetes 130-US Hospitals dataset** and draws on the _Strack et al._ study as a methodological reference, while focusing strictly on prediction rather than causal inference.

The initial baseline model used one encounter per patient, converted the original three-level readmitted outcome into a binary 30-day readmission target, and applied logistic regression with a restricted set of admission-related features. After tuning, the model achieved a test ROC-AUC of approximately 0.59, indicating limited but better-than-random ranking performance.

An expanded version of the model incorporated HbA1c (`A1Cresult`; glycated hemoglobin) and serum glucose (`max_glu_serum`) categories. Predictive performance changed very little: the expanded model achieved a ROC-AUC of approximately 0.586, which was effectively the same as the baseline. Overall, the results suggest that the current logistic regression setup and selected variables do not strongly distinguish patients who will be readmitted within 30 days from those who will not.

**Rationale**

Hospital readmission is an important clinical and operational measure because early readmissions may reflect disease severity, gaps in care, or insufficient follow-up support. In patients with diabetes, predicting readmission is particularly relevant because effective disease management often depends on coordination between inpatient treatment and outpatient care.

A reliable prediction model could help identify patients who may benefit from additional discharge planning or closer follow-up after hospitalization. However, this project does not interpret the observed relationships as causal. All findings are treated as predictive associations only.

**Research Question**

Can machine learning models predict whether a patient with diabetes will be readmitted to the hospital within 30 days?

A secondary question is whether HbA1c and glucose-related variables improve predictive performance compared with a more restricted baseline model.

**Data Source**

The dataset is the UCI **Diabetes 130-US Hospitals for Years 1999-2008** dataset, associated with:

Strack, Beata, DeShazo, Jonathan P., Gennings, Chris, Olmo, Juan L., Ventura, Sebastian, Cios, Krzysztof J., and Clore, John N. "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records." BioMed Research International, 2014. https://doi.org/10.1155/2014/781670

The dataset is de-identified. Use it only for educational analysis and do not make patient-care recommendations from this project.

The raw dataset contains 101,766 hospital encounters. To reduce duplication at the patient level and align more closely with the published study, the analysis retained only the first encounter for each patient, leaving 71,518 records before later exclusions and feature-level cleaning. For the expanded analysis, discharge categories related to death or hospice care were removed (similarly to the paper), resulting in 69,973 records before feature-level preprocessing.

The dataset is cached locally in the data/ folder upon first download so that future notebook runs do not require downloading the raw files again.

**Methodology**

The target variable was derived from the original `readmitted` field:

* 1: readmitted within 30 days (<30)
* 0: not readmitted within 30 days (>30 or NO)

The initial feature set was limited to variables available at or near the time of admission:

* age group
* race group
* gender
* admission type
* admission source group
* primary diagnosis group
* prior outpatient, emergency, and inpatient utilization

Several variables were excluded from the baseline because of missingness, high cardinality (to avoid an excessive number of features), or potential data leakage. For example, `weight`, `payer_code`, and `medical_specialty` were excluded because of incomplete data or modelling complexity. Variables that would only be known during or after the hospital stay - including discharge disposition, length of stay, laboratory counts, medication changes, HbA1c results, and glucose measurements - were omitted from the strict admission-time baseline.

Feature engineering steps included (following the approach used in the paper):

* combining age bands into broader age groups,
* grouping race categories,
* grouping admission sources into emergency, physician referral, and other,
* mapping raw ICD-9 primary diagnosis codes into broader clinical categories,
* converting prior utilization counts into categorical ranges (0, 1–2, >2) after an initial trial using numeric counts,
* adding categorical HbA1c and serum glucose variables in the expanded model.

The baseline modeling workflow used logistic regression with balanced class weights. Because the 30-day readmission target is highly imbalanced, accuracy alone was not treated as an appropriate evaluation metric. Model performance was evaluated using confusion matrices, precision, recall, F1-score, ROC-AUC, average precision, and threshold analysis. 

**Results**

After preprocessing for the initial model, the stratified train/test split contained:

* training set: 57,203 records,
* test set: 14,301 records,
* positive class rate: approximately 8.8%.

The tuned logistic regression baseline achieved:

* test ROC-AUC: approximately 0.592,
* test average precision: approximately 0.130,
* readmission-class recall at threshold 0.50: approximately 0.55,
* readmission-class precision at threshold 0.50: approximately 0.11.

The model was able to identify some patients who were readmitted within 30 days, but it also generated a large number of false positives. This outcome is expected when balanced class weights are applied to a strongly imbalanced target variable. The results provide a reasonable baseline for comparison, although the model is not accurate enough for practical deployment in its current form.

The expanded model added HbA1c and glucose-related categories after excluding discharge dispositions associated with death or hospice care. Its ROC-AUC was approximately 0.586, which was effectively unchanged from the baseline model. Within this logistic regression framework, the additional laboratory-result categories did not meaningfully improve predictive performance.

**Next Steps**

The next stage of the project should evaluate models capable of capturing non-linear relationships and feature interactions. Candidate approaches include tree-based methods such as random forest, gradient boosting, XGBoost, and LightGBM. Neural network models could also be explored, although tree-based methods are likely to be a more suitable next step for this type of tabular dataset. 

Additional discharge-time features can also be tested later, including discharge disposition, length of stay, laboratory counts, and medication changes.

**Repository Contents**
* [Initial EDA and baseline modeling notebook](diabetes.ipynb)
