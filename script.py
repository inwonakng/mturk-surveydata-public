#%% start code
from random import randint
from math import exp
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from mlxtend.plotting import plot_decision_regions
from mlxtend.evaluate import feature_importance_permutation
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.tree import export_graphviz,DecisionTreeClassifier
from graphviz import Source
from os import system

datafile = '1_airplane_scenario/data/final.csv'

# %% get features ready
df = pd.read_csv(datafile)
workers = df['Worker ID'].unique()

feature_names = ['AGE', 'GENDER', 'PURPOSE OF TRIP', 'CAREER', 'HEALTH']

feature_categories = []
for f in feature_names:
    cat = df[f].unique()
    feature_categories.append(cat)

score_dict = {1:'A score', 2:'B score', 3:'C score'}    

age_dict = {
    'middle aged': 40,
    '27 year old': 27,
    'college student':22,
    'young child': 8,
    '5 year old child': 5,
    'senior citizen':70,
    'middle-schooler':12
}

gender_dict = {
    'male':0,
    'man':0,
    'female':1,
    'woman':1,
    'nan':0
}

trip_dict = {
    'on a once in a lifetime chance trip from lottery': 'lottery',
    'on the way to an volunteer trip': 'volunteer',
    'visiting family far away': 'family',
    'on a routine trip': 'routine',
    'on the way to an expensive concert': 'concert',
    'on a vacation': 'vacation',
    'on a volunteer trip': 'volunteer'
}

health_dict = {
    'with asthma': 'asthma',
    'who is terminally ill with 5 years left': 'terminal',
    'who is wheelchair bound': 'wheelchair',
    'who is on a wheelchair': 'wheelchair',
    'in great health': 'healthy'
}

user_age = {
    '10~20': 10,
    '20~29': 20,
    '30~39': 30,
    '40~49': 40,
    '50~59': 50,
    '60~69': 60,
    'other': 0,
    'N/A': 0
}

user_gender = {
    'male': 0,
    'female': 1,
    'other': 0,
    'N/A': 0
}

user_education = {
    'highschool' : 1,
    'college': 2,
    'graduate': 3,
    'other': 0,
    'N/A': 0
}


# %% clean features
# assuming we have 3 options per each task
df_new = pd.DataFrame()
df_usertmp = pd.DataFrame()
task_no = 0
for i in range(0,len(df)-2,3):  
    for j in range(3):
        options = {}
        options['TurkerID'] = [df.iloc[i+j]['Worker ID']]
        options['taskID'] = [task_no]
        #age
        x = df.iloc[i+j]['AGE']
        if(x.isnumeric()):
            age = int(x)
        else:
            age = age_dict[x]
        options['AGE'] = [age]
        #gender
        x = str(df.iloc[i+j]['GENDER'])
        options['GENDER'] = [gender_dict[x]]
        #purpose
        x = str(df.iloc[i+j]['PURPOSE OF TRIP'])
        if(not(str(x)=='nan')):
            options['PURPOSE.'+trip_dict[x]] = [1]
        #career
        x = str(df.iloc[i+j]['CAREER'])
        if(not(str(x)=='nan')):
            options['CAREER.'+x.split()[0]] = [1]
        #health
        x = str(df.iloc[i+j]['HEALTH'])
        if(not(str(x)=='nan')):
            options['HEALTH.'+health_dict[x]] = [1]
        #survival chance
        surv = json.loads(df.iloc[i+j]['SURVIVAL'].replace('\'','\"'))
        options['Survival_with'] = int(surv['with'])
        options['Survival_without'] = int(surv['without'])
        options['Survival_difference'] = int(surv['with']) - int(surv['without'])
        #score
        alt = df.iloc[i+j]['Option #']
        options['Score'] = df.iloc[i+j][score_dict[alt]]
        
        df0 = pd.DataFrame.from_dict(options)
        df_new = df_new.append(df0)
    
    task_no += 1
    
    # user info
    user_options = {}
    user_options['TurkerID'] = [df.iloc[i]['Worker ID']]
    x = str(df.iloc[i]['Age Group'])
    if not x == 'nan':
        user_options['USER.age'] = [user_age[x]]
    x = str(df.iloc[i]['Education'])
    if not x == 'nan':
        user_options['USER.education'] = [user_education[x]]
    x = str(df.iloc[i]['Gender'])
    if not x == 'nan':
        user_options['USER.gender'] = [user_gender[x]]
    df_usertmp = df_usertmp.append(pd.DataFrame.from_dict(user_options))

#%% Clean up user data

df_usertmp = df_usertmp.drop_duplicates()
df_usertmp.fillna(0, inplace=True)
# sometimes people fill their data in firsst time and not second time
# so hopefully they didn't lie the first time
by_id = df_usertmp.groupby('TurkerID')

df_user = pd.DataFrame()

for id,vals in by_id:
    if len(vals) > 1:
        shitval = vals
        mostvalid = 0
        bestrow = vals.iloc[0]
        for i in range(len(vals)):
            validcount = np.count_nonzero(vals.iloc[i])
            if validcount > mostvalid:
                mostvalid = validcount
                bestrow = vals.iloc[i]
        df_user = df_user.append(bestrow)
    else:
        df_user = df_user.append(vals)

df_user.set_index('TurkerID',inplace = True)
#%% Create pairwise data

features = ['AGE', 'GENDER', 'PURPOSE.lottery', \
       'CAREER.pizza', 'HEALTH.asthma', 'Survival_with', \
       'Survival_without', 'Survival_difference', 'PURPOSE.volunteer', \
       'CAREER.professor', 'HEALTH.terminal', 'PURPOSE.family', \
       'HEALTH.wheelchair', 'PURPOSE.routine', 'PURPOSE.concert', \
       'PURPOSE.vacation', 'CAREER.friends', 'CAREER.ex-convict', \
       'CAREER.parent', 'HEALTH.healthy', 'CAREER.homeless', \
       'CAREER.businessperson', 'CAREER.clubmates', 'CAREER.politician', \
       'CAREER.family']
user_features = ['USER.age', 'USER.education', 'USER.gender']

df_turker = df_new.groupby(['TurkerID'])
pairs = [(0,1),(0,2),(1,2)]    

X = []
Y = []
user_d = []

for idx, nf in df_turker:
    turker_x = []
    turker_y = []
    nf.fillna(0, inplace=True)
    user_data = df_user.loc[idx]
    df_task = nf.groupby('taskID')
    for ii, tf in df_task:
        for p in pairs:
            if(nf.iloc[p[0]]['Score'] == nf.iloc[p[1]]['Score']):
                continue #ignore ties for now
            # for now only look at parts with valid user data
            # if user_data.size < 3: continue
            feature_diff = nf.iloc[p[0]][features].values - nf.iloc[p[1]][features].values
            combined = np.append(feature_diff, user_data[user_features].values)
            if(nf.iloc[p[0]]['Score'] > nf.iloc[p[1]]['Score']):
                turker_x.append(feature_diff)
                turker_y.append(1)
            else:
                turker_x.append(feature_diff)
                turker_y.append(0)
    X.append(turker_x)
    Y.append(turker_y)
    user_d.append(user_data)
#%% Running Regression

f_ofinterest = ['GENDER',
                'AGE',
                'CAREER.parent',
                'CAREER.politician',
                'CAREER.ex-convict',
                'HEALTH.terminal',
                'HEALTH.healthy']

max_abs_scaler = preprocessing.MaxAbsScaler()

foreach_feature = dict(zip(f_ofinterest,[[[],[]] for f in f_ofinterest]))

users_f = []
user_ids = []
feats_f = []

for x,y,u in zip(X,Y,user_d):
    zc = np.count_nonzero(y)
    if zc == 0 or zc == len(y): continue
    if np.count_nonzero(u) < 2: continue
    X_maxabs = max_abs_scaler.fit_transform(x)
    lab_enc = preprocessing.LabelEncoder()

    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_maxabs, y)
    # target vector can't be float so rounding up as much
    padded = [int(c * 10) for c in clf.coef_[0]]
    coeffs = dict(zip(features, clf.coef_[0]))
    encoded_co = dict(zip(features, padded))

    userval = u[user_features].values

    users_f.append(userval)
    user_ids.append(u.name)
    feats_f.append(coeffs)

    for f in f_ofinterest:
        y_encoded = coeffs[f]
        foreach_feature[f][0].append(userval)
        foreach_feature[f][1].append(y_encoded)

# %% sorting each user's beta score for the features
df_user_pref = pd.DataFrame()
toprank = {}
botrank = {}
top_n = 3
for f,u_id in zip(feats_f,user_ids):
    vs = sorted(f.items(), key=lambda x: x[1],reverse=True)
    # range is the n value for top n ranks
    for i in range(top_n):
        if vs[i][0] in toprank: toprank[vs[i][0]] += 1
        else: toprank[vs[i][0]] = 1
        
        j = i * -1 -1
        if vs[j][0] in botrank: botrank[vs[j][0]] += 1
        else: botrank[vs[j][0]] = 1

    row = pd.Series(list(f.values()),list(f.keys()),name=u_id)
    df_user_pref = df_user_pref.append(row)

top = sorted(toprank.items(), key=lambda x: x[1],reverse=True)
bot = sorted(botrank.items(), key=lambda x: x[1],reverse=True)

df_user_pref.to_pickle('1_airplane_scenario/data/user_pref_old.pkl')

print(top_n)

# %% using the feature weights to learn user feature
for feat,(user_vector,feat_vector) in foreach_feature.items():

    u_vector = max_abs_scaler.fit_transform(user_vector)
    f_sortbyabs = sorted(feat_vector,key=abs)
    lows = f_sortbyabs[:len(f_sortbyabs)//2]
    highs = f_sortbyabs[len(f_sortbyabs)//2:]

    neg = [f for f in feat_vector if f <= 0]
    pos = [f for f in feat_vector if f > 0]

    hlow_f = []
    negpos_f = []
    for f in feat_vector:
        if f in lows: hlow_f.append(0)
        elif f in highs: hlow_f.append(1)

        if f <= 0: negpos_f.append(0)
        if f > 0: negpos_f.append(1)
    # u_vector = [[u[2]] for u in u_vector]

    regr = LogisticRegression().fit(u_vector,hlow_f)
    print('========================================')
    print('classifying by absolute value (high/low)')
    print('feature of interest\t:'+feat)
    print('score\t\t\t:'+str(regr.score(u_vector,hlow_f)))
    user_coeffs = dict(zip(user_features, regr.coef_[0]))
    for k,v in user_coeffs.items():
        print(k+'\t\t:'+str(v))

    print()

    regr1 = LogisticRegression().fit(u_vector,negpos_f)
    print('classifying by sign (positive/negative)')
    print('feature of interest\t:'+feat)
    print('score\t\t\t:'+str(regr1.score(u_vector,negpos_f)))
    user_coeffs1 = dict(zip(user_features, regr1.coef_[0]))
    for k,v in user_coeffs1.items():
        print(k+'\t\t:'+str(v))

#%% recap of genearl features
all_X = []
all_y = []
for xs in X:
    for x in xs: all_X.append(x)
for ys in Y:
    for y in ys: all_y.append(y)
all_X = max_abs_scaler.fit_transform(all_X)

clf = RandomForestClassifier()
clf.fit(all_X,all_y)
forest = dict(zip(features,clf.feature_importances_))
sorted_forest = sorted(forest.items(), key=lambda x: x[1],reverse=True)

tclf = DecisionTreeClassifier()
tclf.fit(all_X,all_y)
tree = dict(zip(features,tclf.feature_importances_))
sorted_tree = sorted(tree.items(), key=lambda x: x[1],reverse=True)


# results = permutation_importance(clf,all_X,all_y,n_repeats=40)
# rr = dict(zip(features,results.importances_mean))
# sorted_results = sorted(rr.items(), key=lambda x: x[1],reverse=True)
export_graphviz(tclf,out_file='dd.dot',feature_names=features)
system("dot -Tpng dd.dot -o tree1.png")

reg = LogisticRegression()
reg.fit(all_X,all_y)
regre = dict(zip(features,reg.coef_[0]))
sorted_reg = sorted(regre.items(), key=lambda x: abs(x[1]),reverse=True)
# print(regr.score(all_X,all_y))
# coeffs = dict(zip(features, regr.coef_[0]))

#%% making plots
# plt.scatter(feat_vector,[u[0] for u in u_vector],color='black')
# plt.show()
# plt.scatter(feat_vector,[u[1] for u in u_vector],color='blue')
# plt.show()
# plt.scatter(feat_vector,[u[2] for u in u_vector],color='red')
# plt.show()
# plt.scatter([u[1] for u in u_vector],[u[2] for u in u_vector],color='red')
# plt.scatter([u[0] for u in u_vector],[u[2] for u in u_vector],color='green')
# plt.scatter([i for i in range(len(feat_vector))],feat_vector,color='red')
# plt.plot([i for i in range(len(feat_vector))],u_clf.predict(u_vector),color='blue')
# %% log regression try
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
cX = max_abs_scaler.fit_transform(X)
clf = LogisticRegression().fit(cX, y)
clf.score(cX,y)
# clf.predict(X[:2, :])

# %% svm try

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# #############################################################################
# Generate sample data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# #############################################################################
# Add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

# #############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)

# #############################################################################
# Look at the results
lw = 2

svrs = [svr_rbf, svr_lin, svr_poly]
kernel_label = ['RBF', 'Linear', 'Polynomial']
model_color = ['m', 'c', 'g']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].plot(X, svr.fit(X, y).predict(X), color=model_color[ix], lw=lw,
                  label='{} model'.format(kernel_label[ix]))
    axes[ix].scatter(X[svr.support_], y[svr.support_], facecolor="none",
                     edgecolor=model_color[ix], s=50,
                     label='{} support vectors'.format(kernel_label[ix]))
    axes[ix].scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)],
                     y[np.setdiff1d(np.arange(len(X)), svr.support_)],
                     facecolor="none", edgecolor="k", s=50,
                     label='other training data')
    axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                    ncol=1, fancybox=True, shadow=True)

fig.text(0.5, 0.04, 'data', ha='center', va='center')
fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
fig.suptitle("Support Vector Regression", fontsize=14)
plt.show()



# %%
