'''
This file generates the final visuals for the kdd submission
Heatmap of agent reported feature importance scores
heatmap of correlation of generated features
'''

#%%
from matplotlib import pyplot as plt
import pandas as pd
# df = pd.DataFrame()
df = pd.read_csv('parsed_combined.csv')
# %% for user reported importance scores

[
    'Answer.age_importance',
    'Answer.dependents_importance',
    'Answer.gender_importance',
    'Answer.health_importance',
    'Answer.income_importance',
    'Answer.survdif_importance',
    'Answer.survwith_importance'
]