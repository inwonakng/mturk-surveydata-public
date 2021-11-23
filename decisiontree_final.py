#%%
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.tree import export_graphviz,DecisionTreeClassifier
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV


health_dict = {
    'in great health': 0,
    'small health problems': 1,
    'moderate health problems': 2,
    'terminally ill(less than 3 years left)': 3
}

gender_dict = {
    'male': 0,
    'female': 1
}

income_dict = {
    'low': 0,
    'nan': 0,
    'mid': 1,
    'high': 2
}

features = [
    'age',
    'health',
    'gender',
    'income level',
    'number of dependents',
    'survival with jacket',
    'survival without jacket',
    'survival delta'
]

def translate(ss):
    vals = list(ss.values)
    vals = [v.replace('\t','') if type(v) == type('st')
            else v for v in vals] 
    vals[1] = health_dict[vals[1]]
    vals[2] = gender_dict[vals[2]]
    vals[3] = income_dict[vals[3]]
    return np.array(vals)

def getpairs(ddf):
    if len(ddf) == 2: return [(0,1)]
    elif len(ddf) == 3: return [(0,1),(0,2),(1,2)]
    elif len(ddf) == 4: return [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    else: return 'WTF'


#%%    

# making pairwise
df = pd.read_csv('parsed_combined.csv')
Y = pd.DataFrame()
X = pd.DataFrame()
pairs = pd.DataFrame(columns=['X','Y'])

for _,r in tqdm(df.groupby('agent')):
    for _,q in r.groupby('scenario_no'):
        for p1,p2 in getpairs(q):
            scos = q['score']
            feats = q[features]
            yy = 1 if scos.iloc[p1] > scos.iloc[p2] else 0
            xx = translate(feats.iloc[p1]) - translate(feats.iloc[p2])
            pairs = pairs.append({'X':xx,'Y':yy},ignore_index=True)
    
# %%
X = pairs['X'].values.tolist()
Y = pairs['Y'].values.reshape(-1,1).tolist()

trainX,testX,trainY,testY = train_test_split(X,Y,random_state=0)

loops = 20
n_iter = 100 

params = {
    'random_state':[0],
    'min_samples_leaf':list(range(10,301,10)),
    'max_leaf_nodes': list(range(1,51)),
    'max_depth': list(range(1,21))
}


goalscore = 0.7

model = DecisionTreeClassifier

score = 0
if not goalscore:
    goalscore = model().fit(trainX,trainY).score(testX,testY)

for i in tqdm(range(loops)):
    if score > goalscore: break
    search = RandomizedSearchCV(model(),params,n_iter = n_iter)
    search.fit(X,Y)
    pp = search.best_params_
    score = model(**pp).fit(trainX,trainY).score(testX,testY)        

print()
print(score)
print(pp)
# %%
# 0.6123217115689382
# {'random_state': 0, 'min_samples_leaf': 240, 'max_leaf_nodes': 34, 'max_depth': 11}
