import pickle
import pandas as pd
import os

#%%
df = pd.read_csv('C:/Users/farha/Downloads/Batch_4171402_batch_results.csv')
# df = df[df['AssignmentStatus']=='Approved']

pickle_dir = '1_airplane_scenario/data/round6.5/'
files = ['pairs_pickle', 'titanic_pickle', 'triples_pickle', 'tripset0_pickle', 'tripset1_pickle', 'tripset2_pickle', 'tripset3_pickle']

pickle_data = []

for filename in files:
    with open(os.path.join(pickle_dir,filename),'rb') as fh:
         data = pickle.load(fh)
         pickle_data.append(data)

# manually inserting the titanic data
pickle_data[1] = [[{'age': '21',
   'health': 'in great health	',
   'gender': 'male',
   'income level': 'low',
   'number of dependents': '0',
   'survival without jacket': '0%',
   'survival with jacket': '32%'},
  {'age': '32',
   'health': 'in great health	',
   'gender': 'male',
   'income level': 'low',
   'number of dependents': '0',
   'survival without jacket': '0%',
   'survival with jacket': '32%'},
  {'age': '52',
   'health': 'in great health	',
   'gender': 'female',
   'income level': 'low',
   'number of dependents': '1',
   'survival without jacket': '0%',
   'survival with jacket': '32%'},
  {'age': '5',
   'health': 'in great health	',
   'gender': 'female',
   'income level': 'low',
   'number of dependents': '0',
   'survival without jacket': '0%',
   'survival with jacket': '32%'}]]

# # open the files, where you stored the pickled data
# file = open('C:/RPI/CompSoc/Ethical AI/EthicalAI/mturk-surveydata/data/round4/manyfeature/random_pickle', 'rb')
# # dump information to that file
# random_data = pickle.load(file)
# # close the file
# file.close()

# # open a file, where you stored the pickled data
# file = open('C:/RPI/CompSoc/Ethical AI/EthicalAI/mturk-surveydata/data/round4/manyfeature/surv_pickle', 'rb')
# # dump information to that file
# surv_data = pickle.load(file)
# # close the file
# file.close()

#%%
columns = df.columns.values

for i in range(17):
    qi_cols = [col for col in columns if ((f'Answer.q{i}option' in col) or (f'Answer.q{i}type' in col))]

# types = [col for col in columns if ('type' in col)]
# tps = set()
# for tp in types:
#     tps = tps.union(set(df[tp].unique()))
# tps = list(tps) 
tps =  ['pair','titanic','triple', 'tripset0', 'tripset1', 'tripset2', 'tripset3', ] # since they're in alphabetical order, order of tps and files (pickles) match totally

data_points = [2,4,3,3,3,3,3] # number of options in each type
#%%
answers = ['WorkerId', 'Answer.age_importance', 'Answer.dependents_importance', \
           'Answer.gender_importance', 'Answer.health_importance', \
           'Answer.income_importance', 'Answer.survdif_importance', \
           'Answer.survwith_importance']
options = "ABCD"
options = ['option'+x for x in options]

alternative_features = pickle_data[0][0][0].keys()

headers = ['agent'] + answers + ['scenario_no'] + list(alternative_features)+['score']

vals = []

"""
parse the data using optionsets from the pickles
final form will have one row per each alternative of each scenario
So if there are n agents, with 4 scenarios per agents and 4 alternatives per scenario
    there would be 16*n rows
The first few columns (agent features and importance values) are the same throughout all values
    for the agent
In addition to agent TurkerID, agent features and alternative features, it will feature the score
    for each alternative.
Each row can be indexed by ['WorkerId,'scenario_no'] or ['agent','scenario_no']
"""

age_test = []

for agent in range(len(df)):
    row = [agent]
    row += list(df.iloc[agent][answers])
    
    for qno in range(17):
        # All question from different pickle based on type
        # No of options will also be determined by that
        q_tp = df.iloc[agent][f'Answer.q{qno}type']
        tp_idx = tps.index(q_tp) # get type index
        # print(q_tp, data_points[tp_idx])
        
        data = pickle_data[tp_idx] # get correct pickle file
        # if(i==2):
        #     data = surv_data
        # else:
        #     data = random_data
            
        optionset = df.iloc[agent][f"Answer.q{qno}optionset"]
        # print(optionset)
        for j,op in enumerate(options):
            # print(f"Answer.{q}{op}")
            if(j == data_points[tp_idx]): # we've reached all the options
                break
            newrow = row.copy()
            newrow += [qno]
            for feat in alternative_features:
                # print(feat, data[optionset][j][feat])
                newrow += [data[optionset][j][feat]]
            score = df.iloc[agent][f"Answer.q{qno}{op}"]
            newrow += [score]
            # print(score)
            vals.append(dict(zip(headers,newrow)))
#%%
df_new = pd.DataFrame(vals)

df_new['survival with jacket'] = df_new['survival with jacket'].apply(lambda s: int(s[:-1]))
df_new['survival without jacket'] = df_new['survival without jacket'].apply(lambda s: int(s[:-1]))
# merge 23 year old and 27 year old to single group
# decided not against doing this because it messes up the distributions
# df_new.loc[df_new['age'] == '27 year old', 'age'] = "20-30 years old"
# df_new.loc[df_new['age'] == '23 year old', 'age'] = "20-30 years old"

# compute survival_delta
# bin survival_chances
df_new['survival delta'] = df_new['survival with jacket'] - df_new['survival without jacket']
# delta_bins = list(range(0,71,10))
# delta_labels = list(range(0,61,10))
# df_new['survival delta'] = pd.cut(df_new['survival delta'], bins = delta_bins, labels = delta_labels)

# bins = list(range(0,101,10))
# labels = list(range(0,91,10))
# df_new['survival with jacket'] = pd.cut(df_new['survival with jacket'], bins = bins, labels = labels)
#%%
df_new.to_csv('parsed_round7.csv',index=False)
