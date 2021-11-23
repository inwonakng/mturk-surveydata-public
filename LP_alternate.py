import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from scipy.optimize import minimize
from satisfaction_calc import *
from scipy.stats import kendalltau
from matplotlib import pyplot as plt
import copy
from time import time
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


pd.options.mode.chained_assignment = None

#%% preprocessing
# read files
# eventually, need to generalize this
# now, read one at a time and update tripsets manually
# also, need to update agents manually
# alt_features = ['age', 'health', 'gender', 'income level', 'education level', \
#        'number of dependents', 'survival with jacket', 'survival delta']

# df = pd.read_csv('C:/RPI/CompSoc/Ethical AI/EthicalAI/mturk-surveydata/parsed_all_ops_nonbinned.csv')
# df = pd.read_csv('C:/RPI/CompSoc/Ethical AI/EthicalAI/mturk-surveydata/parsed_fewer_nonbinned.csv')
df = pd.read_csv('parsed_round7.csv')

workers = '''AA9Y4BEMJYWN1
A1KL8QLFQ4YTOL
A1O6NAOVMJ1XVC
A3J4QZVB7RJQ0U
A39980PKUJE4NS
A3YP2RNFZ1PQP
A2HX13XTRKDD9C
A32F5JI04GX6T6
A3QU964EOJXS56
A2Z0V2A4SP5QE0
A3T3CRF61D218E
AL2KG1ZCSWKBJ
A3ENDV5PRQI2U2
A3K6R3WRN2HISD
A2RWY5NVRMX0DH
A2GR8GTNEG84IK
A24RQYIDMV7OPK
A10YAUJ72AYRMC
'''.split()
df = df[~(df['WorkerId'].isin(workers))]
agent_src = df.agent.unique()
agent_dest = range(len(df.agent.unique()))
di = dict(zip(agent_src, agent_dest))
df['agent'] = df['agent'].map(di)

df2 = pd.read_csv('parsed_round8.csv')
src = ['pair','triple','titanic'] + ['tripset'+str(i) for i in range(4)]
dest = ['pair','triple','titanic'] + ['tripset'+str(i) for i in range(4,8)]
di = dict(zip(src, dest))
df2['type'] = df2['type'].map(di)  
agent_src = df2.agent.unique()
agent_dest = range(len(df.agent.unique()), len(df.agent.unique()) + len(df2.agent.unique()))
di = dict(zip(agent_src, agent_dest))
df2['agent'] = df2['agent'].map(di)

df3 = pd.read_csv('parsed_round9.csv')
dest = ['pair','triple','titanic'] + ['tripset'+str(i) for i in range(8,12)]
di = dict(zip(src, dest))
df3['type'] = df3['type'].map(di)
agent_src = df3.agent.unique()
agent_dest = range(len(df.agent.unique()) + len(df2.agent.unique()), 
                   len(df.agent.unique()) + len(df2.agent.unique()) + len(df3.agent.unique()))
di = dict(zip(agent_src, agent_dest))
df3['agent'] = df3['agent'].map(di)

df4 = pd.read_csv('parsed_round10.csv')
src = ['pair','triple','titanic'] + ['tripset'+str(i) for i in range(16)]
dest = ['pair','triple','titanic'] + ['tripset'+str(i) for i in range(12,12+16)]
di = dict(zip(src, dest))
df4['type'] = df4['type'].map(di)
agent_src = df4.agent.unique()
agent_dest = range(len(df.agent.unique()) + len(df2.agent.unique()) + len(df3.agent.unique()), 
                   len(df.agent.unique()) + len(df2.agent.unique()) + len(df3.agent.unique()) + len(df4.agent.unique()))
di = dict(zip(agent_src, agent_dest))
df4['agent'] = df4['agent'].map(di)

#%%
df = df.append(df2)
df = df.append(df3)
df = df.append(df4)

#%% functions

# Function for sampling and training on samples
# It would help to recompute the voting data here as well

# write now, considering everything as training set
# i.e. not separating according to type

def train_test_split(N, split = 0.8):
    sorted_msk = np.argsort(np.random.rand(N))
    msk = np.zeros(N)
    msk[sorted_msk[:int(N*split)]] = 1
    return msk


def create_data(df, agents, features, split = 0.8):  
    # return both Xtrain, Ytrain and Xtest, Ytest at the same time
    
    X_train_all = []
    Y_train_all = []
    
    X_test_all = []
    Y_test_all = []
    for agent in range(len(agents)):
        # updating this to df['agent'] compared to WorkerId before
        nf = df[df['agent'] == agents[agent]]
        
        nf.fillna(0, inplace=True)
        X = []
        Y = []
        for i in range(17):
            #just break them down into pairs and do the same thing
            # run a nested loop (for i in range(len): for j in range(i+1,len))
            x = np.array(nf[nf['scenario_no']==i][features])
            # y = np.array(nf[nf['scenario_no']==i]['score']).argmax() # just keep top
            y = (-1*np.array(nf[nf['scenario_no']==i]['score'])).argsort() # keep whole ranking
            # include all rank-breaking pairs into training set
            
            for m1 in range(len(y)):
                for m2 in range(m1+1, len(y)):
                    # print(x[[m1,m2]])
                    # print(y[[m1,m2]])
                    # X_train.append(x[[m1,m2]])
                    # now, instead of creating a dict, just have a feature vector
                    #   with the difference of features
                    #X.append([dict(zip(features,x[m1])), dict(zip(features,x[m2]))])
                    X.append(x[m1] - x[m2])
                    Y.append(np.argsort(y[[m1,m2]])[0])
                    
        # create train and test here
        N  = len(X) # no. of pairwise comparisons (probably)
        msk = train_test_split(N, split = split)
        X_train = np.array(X)[msk==1]
        X_test = np.array(X)[msk==0]
        
        y_train = np.array(Y)[msk==1]
        y_test = np.array(Y)[msk==0]
        
        X_train_all.append(X_train)
        Y_train_all.append(y_train)
        X_test_all.append(X_test)
        Y_test_all.append(y_test)
                            
    return X_train_all, Y_train_all, X_test_all, Y_test_all

def match_winners(w1, w2):
    com = set(w1).intersection(set(w2))
    if(len(com) > 0):
        return 1
    else:
        return 0
    
#%%
# age_keys = ['27 year old', 'young child', '18 year old', '23 year old', 'middle aged', 'senior citizen']
# age_vals = [27, 8, 18, 23, 40, 70]
# age_dict = dict(zip(age_keys, age_vals))
# df['age'] = df['age'].map(age_dict)

gender_keys = ['male','female']
gender_vals = [0, 1]
gender_dict = dict(zip(gender_keys, gender_vals))
df['gender'] = df['gender'].map(gender_dict)

health_keys = [np.nan, 'great health', 'in great health', 'in great health\t', 'moderate health problems', \
       'small health problems', 'terminally ill(less than 3 years left)']
health_vals = [0, 0, 0, 0, 2, 1, 3]
health_dict = dict(zip(health_keys, health_vals))
df['health'] = df['health'].map(health_dict)

income_keys = ['high', np.nan, 'low', 'mid']
income_vals = [2,0,0,1]
income_dict = dict(zip(income_keys, income_vals))
df['income level'] = df['income level'].map(income_dict)

if('education level' in df.columns.values):
    education_keys = ['High school graduate', np.nan, 'Middle school graduate', \
           'College graduate', 'Graduate degree']
    education_vals = [1,0,0,2,3]
    education_dict = dict(zip(education_keys, education_vals))
    df['education level'] = df['education level'].map(education_dict)

df['survival delta'] = df['survival with jacket'] - df['survival without jacket']
# features with survival_with
if('education level' in df.columns.values):
    features = ['age', 'health', 'gender', 'income level', 'education level', \
        'number of dependents', 'survival with jacket', 'survival delta']
else:
    features = ['age', 'health', 'gender', 'income level', \
        'number of dependents', 'survival with jacket', 'survival delta']

# features without survival_with
# if('education level' in df.columns.values): # for all features
#     features = ['age', 'health', 'gender', 'income level', 'education level', \
#        'number of dependents', 'survival delta']
# else: # for fewer features
#     features = ['age', 'health', 'gender', 'income level', \
#        'survival delta']
        
#%% preprocessing

X = df[features].values
min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X)
df[features] = X_minmax

#%%
agents = df['agent'].unique()
    
# This to are for older function
# X, Y = create_data(df, agents, features)
# X_test, Y_test = create_data(df, agents, features, train = False)
X, Y, X_test, Y_test = create_data(df, agents, features, split = 0.8)

#%% logistic regression

training_acc = []
testing = True
testing_acc = []

for i,x in enumerate(X):
    if(np.sum(Y[i]) == len(Y[i]) or np.sum(Y[i]) == 0):
        training_acc.append(np.nan)
        if(testing):
            testing_acc.append(np.nan)
        continue
    clf = LogisticRegression(random_state=0).fit(x, Y[i])
    training_acc.append(clf.score(x, Y[i]))
    if(testing):
        testing_acc.append(clf.score(X_test[i], Y_test[i]))

print(np.nanmean(training_acc))
if(testing):
    print(np.nanmean(testing_acc))
    
#%% 1 layer MLP

Acc1 = []
Acc12 = []

for l1 in range(4,21,4):
    print('hidden layer = ', l1)    

    training_acc = []
    testing = True
    testing_acc = []
    
    for i,x in enumerate(X):
        if(np.sum(Y[i]) == len(Y[i]) or np.sum(Y[i]) == 0):
            training_acc.append(np.nan)
            if(testing):
                testing_acc.append(np.nan)
            continue
        clf = MLPClassifier(alpha=1e-5,hidden_layer_sizes=(l1), 
                            random_state=1)
        clf.fit(x, Y[i])
        training_acc.append(clf.score(x, Y[i]))
        if(testing):
            testing_acc.append(clf.score(X_test[i], Y_test[i]))
    
    print(np.nanmean(training_acc))
    Acc1.append(np.nanmean(training_acc))
    if(testing):
        print(np.nanmean(testing_acc))
        Acc12.append(np.nanmean(testing_acc))

#%% 2 layer MLP

Acc = []
Acc2 = []

for l1 in range(4,21,4):
    for l2 in range(4,13,4):
        print('hidden layers = ', l1, l2)    
    
        training_acc = []
        testing = True
        testing_acc = []
        
        for i,x in enumerate(X):
            if(np.sum(Y[i]) == len(Y[i]) or np.sum(Y[i]) == 0):
                training_acc.append(np.nan)
                if(testing):
                    testing_acc.append(np.nan)
                continue
            clf = MLPClassifier(alpha=1e-5,hidden_layer_sizes=(l1, l2), 
                                random_state=1)
            clf.fit(x, Y[i])
            training_acc.append(clf.score(x, Y[i]))
            if(testing):
                testing_acc.append(clf.score(X_test[i], Y_test[i]))
        
        print(np.nanmean(training_acc))
        Acc.append(np.nanmean(training_acc))
        if(testing):
            print(np.nanmean(testing_acc))
            Acc2.append(np.nanmean(testing_acc))
            
#%%

imp_features = ['agent', 'Answer.dependents_importance', 'Answer.gender_importance',
       'Answer.health_importance', 'Answer.income_importance',
       'Answer.survdif_importance', 'Answer.survwith_importance']

df_imp = df[imp_features]
df_imp = df_imp.groupby('agent').min()
df_imp = df_imp.reset_index()

mean_ = df_imp.describe().loc['mean']
std_ = df_imp.describe().loc['std']