import analysis as a
import tqdm
from random import randint
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from nltk.tokenize import TreebankWordTokenizer
import json

datafile = 'C:/RPI/CompSoc/Ethical AI/EthicalAI/mturk-surveydata/data/round2_parsed_final.csv'

#%% Clean up data and create dummy features
df = pd.read_csv(datafile)
workers = df['Worker ID'].unique()

feature_names = ['AGE', 'GENDER', 'PURPOSE OF TRIP', 'CAREER', 'HEALTH']

feature_categories = []
for f in feature_names:
    cat = df[f].unique()
    feature_categories.append(cat)

score_dict = {1:'A score', 2:'B score', 3:'C score'}    

age_keys = ['middle aged', '27 year old', 'college student', 'young child', 'senior citizen']
age_vals = [40, 27, 22, 8, 70]
age_dict = dict(zip(age_keys, age_vals))

gender_keys = ['male', 'man','female', 'woman', 'nan']
gender_vals = [0, 0, 1, 1, 0]
gender_dict = dict(zip(gender_keys, gender_vals))

trip_keys = ['on a once in a lifetime chance trip from lottery', \
       'on the way to an volunteer trip', 'visiting family far away', \
       'on a routine trip', 'on the way to an expensive concert', \
       'on a vacation', 'on a volunteer trip']
trip_vals = ['lottery', 'volunteer', 'family', 'routine', 'concert', \
       'vacation', 'volunteer']    
trip_dict = dict(zip(trip_keys, trip_vals))

health_keys = ['with asthma', 'who is terminally ill with 5 years left', \
       'who is wheelchair bound', 'in great health']
health_vals = ['asthma', 'terminal', 'wheelchair', 'healthy']
health_dict = dict(zip(health_keys, health_vals))

group_df = df.groupby(['Worker ID'])

# assuming we have 3 options per each task
df_new = pd.DataFrame()
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
        options['Survival_with'] = surv['with']
        options['Survival_difference'] = surv['with'] - surv['without']
        
        #score
        alt = df.iloc[i+j]['Option #']
        options['Score'] = df.iloc[i+j][score_dict[alt]]
        
        df0 = pd.DataFrame.from_dict(options)
        df_new = df_new.append(df0)
        
    task_no += 1

#%% Create pairwise data

features = ['AGE', 'GENDER', 'PURPOSE.lottery', \
       'CAREER.pizza', 'HEALTH.asthma', 'Survival_with', \
       'Survival_difference', 'PURPOSE.volunteer', \
       'CAREER.professor', 'HEALTH.terminal', 'PURPOSE.family', \
       'HEALTH.wheelchair', 'PURPOSE.routine', 'PURPOSE.concert', \
       'PURPOSE.vacation', 'CAREER.friends', 'CAREER.ex-convict', \
       'CAREER.parent', 'HEALTH.healthy', 'CAREER.homeless', \
       'CAREER.businessperson', 'CAREER.clubmates', 'CAREER.politician', \
       'CAREER.family']

df_group = df_new.groupby(['TurkerID','taskID'])
pairs = [(0,1),(0,2),(1,2)]    

X = []
Y = []

for idx, nf in df_group:
    # print(nf)
    nf.fillna(0, inplace=True)
    for p in pairs:
        if(nf.iloc[p[0]]['Score'] == nf.iloc[p[1]]['Score']):
            continue #ignore ties for now
        feature_diff = nf.iloc[p[0]][features].values - nf.iloc[p[1]][features].values
        if(nf.iloc[p[0]]['Score'] > nf.iloc[p[1]]['Score']):
            X.append(feature_diff)
            Y.append(0)
        else:
            X.append(feature_diff)
            Y.append(1)
            
#%% Logistic Regression
max_abs_scaler = preprocessing.MaxAbsScaler()
X_maxabs = max_abs_scaler.fit_transform(X)

clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_maxabs, Y)
print(clf.score(X_maxabs, Y))
coeffs = dict(zip(features, clf.coef_[0]))

#%% NLP Stuff
# ngram_size = 2
# corpus = []
# # data[workers[0]][0]['A'][-1]

# for i in range(len(workers)):
#     for dat in data[workers[i]]:
#         corpus.append(dat['A'][-1])
        
# #%%

# vect = CountVectorizer(ngram_range=(1,ngram_size), \
#                        tokenizer=TreebankWordTokenizer().tokenize, \
#                        max_df=0.7, stop_words='english')

# X = vect.fit_transform(corpus)
# xx = X.toarray()

# word_list = vect.get_feature_names()

# def find_count(token):
#     idx = word_list.index(token)
#     print(np.sum([xx[i][idx] for i in range(len(xx))]))
#     print(np.mean([xx[i][idx] for i in range(len(xx))]))
    
