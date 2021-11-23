#%% start
import sys
%reload_ext autoreload
%autoreload 2
sys.path.append('../../')
from helpers import *

#%%

gdict = {
    'male':0,
    'female':1
}

hdict = {
    'N/A': 0,
    "in great health": 0,
    "small health problems": 1,
    "moderate health problems": 2,
    "terminally ill(less than 3 years left)": 3,
}
edict = {
    'N/A': 0,
    "Middle school graduate":0,
    "High school graduate":1,
    "College graduate":2,
    "Graduate degree":3,
}
idict = {
    'N/A': 0,
    "low": 0,
    "mid": 1,
    "high": 2,
}

pp = pickle.load(open('round5/random_pickle','rb'))
regdf = pd.DataFrame()
excdf = pd.DataFrame()
for ppp in pp:
    for p in ppp:
        regpc = {}
        excpc = {}
        bad = False
        for k,v in p.items():
            if k == 'gender':
                regpc[k] = [gdict[v]]
                excpc[k] = [gdict[v]]
            elif k == 'health':
                regpc[k] = [hdict[v]]
                excpc[k] = [hdict[v]]
            elif k == 'education level':
                regpc[k] = [edict[v]]
                excpc[k] = [edict[v]]
            elif k == 'income level':
                regpc[k] = [idict[v]]
                excpc[k] = [idict[v]]
            elif 'survival' in k:
                regpc[k] = [int(v[:-1])]
                excpc[k] = [int(v[:-1])]
            else:
                if k == 'age' and (v == '12' or v == '8'):
                    bad = True
                excpc[k] = [int(v)]
                regpc[k] = [int(v)]
        regdf = regdf.append(regdf.from_dict(regpc))
        if not bad:
            excdf = excdf.append(excdf.from_dict(excpc))
    
#%% Making Correlation matrix

kcor = regdf.corr()
heatmap(kcor,'Feature Correlation (Kendal)','random','1_airplane_scenario')

pcor = regdf.corr(method='pearson')
heatmap(pcor,'Feature Correlation (Pearson)','random','1_airplane_scenario')

kcor = excdf.corr()
heatmap(kcor,'Feature Correlation (Kendal)','excluded','1_airplane_scenario')

pcor = excdf.corr(method='pearson')
heatmap(pcor,'Feature Correlation (Pearson)','excluded','1_airplane_scenario')
