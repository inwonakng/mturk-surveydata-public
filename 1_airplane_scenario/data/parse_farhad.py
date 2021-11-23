#%%
import pickle
import pandas as pd

#%%
# open a file, where you stored the pickled data
file = open('round5/random_pickle', 'rb')

defstory = {'default':
[{'age': 21, 'education level': 'N/A', 'gender': 'male', 'health': 'in great health', 'income level': 'low', 'number of dependents': 0, 'survival with jacket': '32%', 'survival without jacket': '0%'},
{'age': 32, 'education level': 'N/A', 'gender': 'male', 'health': 'in great health', 'income level': 'low', 'number of dependents': 0, 'survival with jacket': '32%', 'survival without jacket': '0%'},
{'age': 52, 'education level': 'N/A', 'gender': 'female', 'health': 'in great health', 'income level': 'high', 'number of dependents': 1, 'survival with jacket': '32%', 'survival without jacket': '0%'},
{'age': 5, 'education level': 'N/A', 'gender': 'female', 'health': 'in great health', 'income level': 'high', 'number of dependents': 0, 'survival with jacket': '32%', 'survival without jacket': '0%'}]}


# dump information to that file
random_data = pickle.load(file)
# close the file
file.close()

df = pd.read_csv('round5/Batch_4165125_batch_results (2).csv')
df2 = pd.read_csv('round5/Batch_4166420_batch_results.csv')

columns = df.columns.values
answers = ['WorkerId', 'Answer.agegroup', 'Answer.education', 'Answer.gender', \
                 'Answer.age_importance', 'Answer.dependents_importance', \
                 'Answer.education_importance', 'Answer.gender_importance', \
                 'Answer.health_importance', 'Answer.income_importance', \
                 'Answer.survdif_importance', 'Answer.survwith_importance']
options = "ABCD"
options = ['option'+x for x in options]

alternative_features = list(random_data[0][0].keys())

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

df_written = pd.DataFrame()

age_test = []

wri_vals = []

for agent in range(len(df)):
    row = [agent]
    row += list(df.iloc[agent][answers])
    for i in range(4):
        # q0 is default question
        if(i==0):
            data = defstory
            # continue
        else:
            data = random_data
            
        q = 'q' + str(i)
        optionset = df.iloc[agent][f"Answer.{q}optionset"]
        # print(optionset)
        for j,op in enumerate(options):
            # print(f"Answer.{q}{op}")
            newrow = row.copy()
            newrow += [i]
            for feat in alternative_features:
                # print(feat, data[optionset][j][feat])
                if not 'survival' in feat:
                    newrow += [data[optionset][j][feat]]
                else:
                    newrow += [int(data[optionset][j][feat][:-1])]
            if(i==2):
                age_test.append(data[optionset][j]['age'])
            score = df.iloc[agent][f"Answer.{q}{op}"]
            newrow += [score]
            # print(score)
            vals.append(dict(zip(headers,newrow)))
    writtenrow = []
    id = df.iloc[agent]['WorkerId']

    df_written = df_written.append(
        pd.Series({'WorkerId':id,
            'text':df.iloc[agent]['Answer.suggestion']},name=0)
    )
    
df_new1 = pd.DataFrame(vals)


vals = []
age_test = []

for agent in range(len(df2)):
    row = [agent]
    row += list(df2.iloc[agent][answers])
    for i in range(4):
        # q0 is default question
        if(i==0):
            data = defstory
        else:
            data = random_data
            
        q = 'q' + str(i)
        optionset = df2.iloc[agent][f"Answer.{q}optionset"]
        # print(optionset)
        for j,op in enumerate(options):
            # print(f"Answer.{q}{op}")
            newrow = row.copy()
            newrow += [i]
            for feat in alternative_features:
                # print(feat, data[optionset][j][feat])
                if not 'survival' in feat:
                    newrow += [data[optionset][j][feat]]
                else:
                    newrow += [int(data[optionset][j][feat][:-1])]
            if(i==2):
                age_test.append(data[optionset][j]['age'])
            score = df2.iloc[agent][f"Answer.{q}{op}"]
            newrow += [score]
            # print(score)
            vals.append(dict(zip(headers,newrow)))
    writtenrow = []
    id = df2.iloc[agent]['WorkerId']

    df_written = df_written.append(
        pd.Series({'WorkerId':id,
            'text':df2.iloc[agent]['Answer.suggestion']},name=0)
    )

df_new2 = pd.DataFrame(vals)

df_all = pd.concat([df_new1,df_new2])


# merge 23 year old and 27 year old to single group
# decided not against doing this because it messes up the distributions
# df_new.loc[df_new['age'] == '27 year old', 'age'] = "20-30 years old"
# df_new.loc[df_new['age'] == '23 year old', 'age'] = "20-30 years old"

# compute survival_delta
# bin survival_chances
df_new2['survival delta'] = df_new2['survival with jacket'] - df_new2['survival without jacket']
# delta_bins = list(range(0,71,10))
# delta_labels = list(range(0,61,10))
# df_new['survival delta'] = pd.cut(df_new['survival delta'], bins = delta_bins, labels = delta_labels)

# bins = list(range(0,101,10))
# labels = list(range(0,91,10))
# df_new['survival with jacket'] = pd.cut(df_new['survival with jacket'], bins = bins, labels = labels)

df_all.to_csv('round5/parsed_data.csv',index=False)
df_written.to_pickle('round5/written_data')

#%%
# open a file, where you stored the pickled data
file = open('C:/RPI/CompSoc/Ethical AI/EthicalAI/mturk-surveydata/data/round4/fewfeature/rand_fewer_pickle', 'rb')
# dump information to that file
random_data = pickle.load(file)
# close the file
file.close()

# open a file, where you stored the pickled data
file = open('C:/RPI/CompSoc/Ethical AI/EthicalAI/mturk-surveydata/data/round4/fewfeature/surv_fewer_pickle', 'rb')
# dump information to that file
surv_data = pickle.load(file)
# close the file
file.close()

df2 = pd.read_csv('C:/Users/farha/Downloads/surveydata_4ops_simple.csv')

columns = df2.columns.values
answers = ['WorkerId', 'Answer.agegroup', 'Answer.education', 'Answer.gender', \
                 'Answer.age_importance', 'Answer.gender_importance', \
                 'Answer.health_importance', 'Answer.income_importance', \
                 'Answer.survdif_importance',  'Answer.survwith_importance']
options = "ABCD"
options = ['option'+x for x in options]

alternative_features = random_data[0][0].keys()

headers = ['agent'] + answers + ['scenario_no'] + list(alternative_features)+['score']

vals = []

for agent in range(len(df2)):
    row = [agent]
    row += list(df2.iloc[agent][answers])
    for i in range(4):
        #q0,q1,q3 from random_data
        #q2 from surv_data
        if(i==2):
            data = surv_data
        else:
            data = random_data
            
        q = 'q' + str(i)
        optionset = df2.iloc[agent][f"Answer.{q}optionset"]
        # print(optionset)
        for j,op in enumerate(options):
            # print(f"Answer.{q}{op}")
            newrow = row.copy()
            newrow += [i]
            for feat in alternative_features:
                # print(feat, data[optionset][j][feat])
                newrow += [data[optionset][j][feat]]
            
            score = df2.iloc[agent][f"Answer.{q}{op}"]
            newrow += [score]
            # print(score)
            vals.append(dict(zip(headers,newrow)))

df_new = pd.DataFrame(vals)

# compute survival_delta
# bin survival_chances
df_new['survival delta'] = df_new['survival with jacket'] - df_new['survival without jacket']
# delta_bins = list(range(0,71,10))
# delta_labels = list(range(0,61,10))
# df_new['survival delta'] = pd.cut(df_new['survival delta'], bins = delta_bins, labels = delta_labels)

# bins = list(range(0,101,10))
# labels = list(range(0,91,10))
# df_new['survival with jacket'] = pd.cut(df_new['survival with jacket'], bins = bins, labels = labels)

df_new.to_csv('parsed_fewer_nonbinned.csv',index=False)