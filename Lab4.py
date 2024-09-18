

import pandas as pd
import numpy as np


column_names = ['Class', 'Age', 'Sex', 'On_thyroxine', 'Query_on_thyroxine', 'On_antithyroid_medication',
                'Sick', 'Pregnant', 'Thyroid_surgery', 'I131_treatment', 'Query_hypothyroid', 'Query_hyperthyroid',
                'Lithium', 'Goiter', 'Tumor', 'Hypopituitary', 'Psych', 'TSH_measured', 'TSH', 'T3_measured',
                'T3', 'TT4_measured', 'TT4', 'T4U_measured', 'T4U', 'FTI_measured', 'FTI', 'TBG_measured',
                'TBG']

df = pd.read_csv('/content/hypothyroid.data', names=column_names)

df.replace('?', np.nan, inplace=True)

df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
df['Class'] = df['Class'].map({'hypothyroid': 1, 'negative': 0})


binary_columns = ['On_thyroxine', 'Query_on_thyroxine', 'On_antithyroid_medication', 'Sick', 'Pregnant',
                  'Thyroid_surgery', 'I131_treatment', 'Query_hypothyroid', 'Query_hyperthyroid', 'Lithium',
                  'Goiter', 'Tumor', 'Hypopituitary', 'Psych', 'TSH_measured', 'T3_measured',
                  'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured']

df[binary_columns] = df[binary_columns].replace({'f': 0, 't': 1, 'y': 1, 'n': 0})

from sklearn.impute import SimpleImputer

# Check which columns still have missing data
missing_columns = df.columns[df.isnull().any()].tolist()
print("Columns with missing data:", missing_columns)

# Apply imputation only on columns that still have missing data
if missing_columns:
    imputer = SimpleImputer(strategy='most_frequent')
    df[missing_columns] = imputer.fit_transform(df[missing_columns])

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


model = BayesianNetwork([('TSH', 'Class'),
                         ('T3', 'Class'),
                         ('Goiter', 'Class')])

# Train the Bayesian Network using Maximum Likelihood Estimation
model.fit(df, estimator=MaximumLikelihoodEstimator)
for cpd in model.get_cpds():
    print(cpd)


# Inference object
inference = VariableElimination(model)

predicted_classes = []
true_classes = df_test['Class'].values

for index, row in df_test.iterrows():

    evidence = {'TSH': row['TSH'], 'T3': row['T3'], 'Goiter': row['Goiter']}


    query_result = inference.map_query(variables=['Class'], evidence=evidence)  # MAP inference for most likely value
    predicted_classes.append(query_result['Class'])


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate evaluation metrics
accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")