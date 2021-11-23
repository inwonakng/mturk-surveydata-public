import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from scipy.optimize import minimize
from satisfaction_calc import *
from scipy.stats import kendalltau
from matplotlib import pyplot as plt
import copy
pd.options.mode.chained_assignment = None

#%% functions

# Function for sampling and training on samples
# It would help to recompute the voting data here as well
def create_data(df, features):  
    agents = df['WorkerId'].unique()
    
    X = []
    Y = []
    for agent in range(len(agents)):
        # agent = np.random.randint(len(agents))
        nf = df[df['WorkerId'] == agents[agent]]
                
        nf.fillna(0, inplace=True)
        X_train = []
        y_train = []
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
                    X_train.append([dict(zip(features,x[m1])), dict(zip(features,x[m2]))])
                    y_train.append(np.argsort(y[[m1,m2]]))
                    
        X.append(X_train)
        Y.append(y_train)
                            
    return X, Y


def match_winners(w1, w2):
    com = set(w1).intersection(set(w2))
    if(len(com) > 0):
        return 1
    else:
        return 0
#%% preprocessing
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

#%%
# tripsets = [f'tripset{i}' for i in range(4)]
# trips = dict(zip(tripsets,[0]*4))
# workers = df['WorkerId'].unique()
# for w in workers:
#     nf = df[df['WorkerId'] == w]
#     for t in tripsets:
#         if(t in nf['type'].unique()):
#             trips[t] += 1
# print(trips)
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
X, Y = create_data(df, features)

#%%
def join_XY(X, Y):
    X_all = []
    Y_all = []
    
    for x in X:
        for pairs in x:
            X_all.append(pairs)
            
    for y in Y:
        for pairs in y:
            Y_all.append(pairs)
    
    return X_all,Y_all

X_all, Y_all = join_XY(X, Y)

#%% First, learn for each person

def LP_learn_helper(x, y, features):
    cnts_feat = []
    for f, feat in enumerate(features):
        cnt = np.zeros(3)
        for ip, pair in enumerate(x):
            ypair = y[ip]
            
            if(pair[0][feat] == pair[1][feat]):
                continue
            else:
                cnt[0] += 1
                # cnt[1] += np.sign((pair[0][f]-pair[1][f])*(ypair[0]-ypair[1]))
                cnt[1] += 1 if (np.sign((pair[0][feat]-pair[1][feat])*(ypair[0]-ypair[1]))>0) else 0
                cnt[2] += 1 if (np.sign((pair[0][feat]-pair[1][feat])*(ypair[0]-ypair[1]))<0) else 0
        
        cnt1 = cnt[1]/cnt[0] if cnt[0]>0 else 0
        cnt2 = cnt[2]/cnt[0] if cnt[0]>0 else 0
        
        cnts_feat.append([cnt1, cnt2])

    return cnts_feat

def LP_learn(x, y, features):
    x_new = copy.deepcopy(x)
    y_new = copy.deepcopy(y)
    
    f_list = []
    
    while(len(x_new)>0):     
        cnts_feat = LP_learn_helper(x_new, y_new, features)
        top = 0
        flag = 0
        top_f = -1
        for f, feat in enumerate(features):
            if(cnts_feat[f][0]>top):
                top = cnts_feat[f][0]
                top_f = f
                flag = 0
            if(cnts_feat[f][1]>top):
                top = cnts_feat[f][1]
                top_f = f
                flag = 1
        f_list.append((features[top_f],flag))
        # print(cnts_feat)
        
        x_update = []
        y_update = []
        for ip, pair in enumerate(x_new):
            if(pair[0][features[top_f]] == pair[1][features[top_f]]):
                x_update.append(pair)
                y_update.append(y_new[ip])
                
        x_new = x_update
        y_new = y_update
        features = [ff for ff in features if not(ff==features[top_f])]
    
    return f_list

learned_imp = []

for ix, x in enumerate(X):
    y = Y[ix]
    # cnts_feat = LP_learn_helper(x,y,features)
    # print(dict(zip(features, cnts_feat)))
    learned_imp.append(LP_learn(x, y, features))

#%%
m = len(features)
borda_score = np.zeros(m)

for lp in learned_imp:
    for l, ll in enumerate(lp):
        j = features.index(ll[0])
        borda_score[j] += m-l-1
        
print(borda_score)
        
# def LP_predict(x1, x2, lp):
#     "lp is a list of features in lexicographic order"
#     for feat in lp:
#         if(x1[feat] == x2[feat]):
#             continue