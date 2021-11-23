import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from scipy.optimize import minimize
from satisfaction_calc import *
from scipy.stats import kendalltau
from matplotlib import pyplot as plt

pd.options.mode.chained_assignment = None

#%% functions

def get_test_data(df, test_type='titanic'):
    print(test_type)
    # generating ground truth ballots, which also act as test data
    agents = df['WorkerId'].unique()
    X_test = []
    y_test = []
    for i, agent in enumerate(agents):
        nf = df[df['WorkerId'] == agent]
        if(not(test_type in nf['type'].unique())):
            continue
        x = np.array(nf[nf['type']==test_type][features])
        # y = np.array(nf[nf['scenario_no']==i]['score']).argmax() # just keep top
        y = (-1*np.array(nf[nf['type']==test_type]['score'])).argsort() # keep whole ranking
        
        X_test.append(x)
        y_test.append(y)
    return X_test,y_test 

# loss and MLE function
def loss(theta, X, y, C):
    nll = 0
    for i,yi in enumerate(y):
        # nll += -(yi *np.log(logit(theta,X[i])) + (1-yi)*np.log((1 - logit(theta,X[i]))))
        # print(yi, X[i])
        gamma = [np.dot(theta, x) for x in X[i]]
        s = np.sum(np.exp(gamma))
        for j,yy in enumerate(yi):
            nll += -(gamma[yy] - np.log(s))
            s -= np.exp(gamma[yy])
            # print(yy, s)
    return nll + np.linalg.norm(theta)*C

def MLE(X_train, y_train):
    theta = np.random.random(len(X_train[0][0]))
    # print(theta)
    res = minimize(fun=loss, x0=theta, args=(X_train, y_train, 0), method='BFGS', \
               options={'gtol': 1e-5, 'maxiter': 100, 'disp': False})
    # print(res.x)
    return res.x

# Function for sampling and training on samples
# It would help to recompute the voting data here as well
def train_all(df, test_type = 'titanic'):  
    print(test_type)
    
    beta_all = []
    agents = df['WorkerId'].unique()
    
    for agent in range(len(agents)):
        # agent = np.random.randint(len(agents))
        nf = df[df['WorkerId'] == agents[agent]]
        
        if(not(test_type in nf['type'].unique())):
            continue
        
        nf.fillna(0, inplace=True)
        X_train = []
        y_train = []
        for i in range(17):
            tp_i = nf[nf['scenario_no']==i]['type'].values[0] # type of current nf
            if(tp_i == test_type):
                continue # do not include test set in training
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
                    X_train.append(x[[m1,m2]])
                    y_train.append(np.argsort(y[[m1,m2]]))
                    
        #     print([list(zip(features,xx)) for xx in x])
        #     print(y)
        
        beta = MLE(X_train, y_train)
        beta_dict = dict(zip(features,beta))
        beta_dict['WorkerId'] = nf.iloc[0]['WorkerId']
        # beta_dict['WorkerId'] = 'Population'
        beta_all.append(beta_dict)
        
    return beta_all


def train_pop(df, samples, test_type = 'titanic'):  
    print(test_type)
    X_cumul = []
    y_cumul = []
    
    agents = df['WorkerId'].unique()
    
    trn_cnt = 0
    for agent in samples:
        # agent = np.random.randint(len(agents))
        nf = df[df['WorkerId'] == agents[agent]]
        
        if(not(test_type in nf['type'].unique())):
            continue
        trn_cnt += 1
        nf.fillna(0, inplace=True)
        
        for i in range(17):
            tp_i = nf[nf['scenario_no']==i]['type'].values[0] # type of current nf
            if(tp_i == test_type):
                continue # do not include test set in training
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
                    X_cumul.append(x[[m1,m2]])
                    y_cumul.append(np.argsort(y[[m1,m2]]))

        #     print([list(zip(features,xx)) for xx in x])
        #     print(y)
        
    beta = MLE(X_cumul, y_cumul)
    beta_dict = dict(zip(features,beta))
    beta_dict['WorkerId'] = 'Population'
    return beta_dict

# predict_votes given beta
def predict_votes(beta_all, features, x, pop_model=False):
    new_ballots = []
    cnt = len(beta_all)
    if(pop_model):
        cnt -= 1
    for beta in beta_all[:cnt]:
        vote = np.argsort(-np.matmul(x, [beta[f] for f in features]))
        new_ballots.append(vote)
    
    if(pop_model):
        beta = beta_all[-1]
        population_score = np.matmul(x, [beta[f] for f in features])
    return np.array(new_ballots), population_score


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
        'survival with jacket', 'survival delta']

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

#%% voting - ground truth

types = ['titanic','tripset0','tripset1','tripset2','tripset3']
# types = ['titanic']

sizes = [[50,100,150],[20,40,60],[20,40,60],[20,40,60],[20,40,60]]
trials = 500

agents = df['WorkerId'].unique()    

for test,test_type in enumerate(types):
    X_test, y_test = get_test_data(df, test_type = test_type)
    #%%
    ballots = np.array(y_test)
    w_pl, s_pl = plurality_winner(ballots)
    w_B, s_B = Borda_winner(ballots)
    w_m, s_m = maximin_winner(ballots)
    rankings = ranking_count(ballots)
    
    print(w_pl, w_B, w_m)
    
    # beta_all = new_train(df, 50)
    
    #%%
    pl_acc = []
    pl_KT = []
    B_acc = []
    B_KT = []
    m_acc = []
    m_KT = []
    
    pop_acc = []
    pop_KT = []
    
    beta_all = train_all(df, test_type = test_type)
    
    training_size = sizes[test]
    for ts in training_size:
        for trial in range(trials):
            print(ts, trial)
            
            samples = np.random.randint(len(agents),size=ts)
            # beta_pop = train_pop(df, samples, test_type = test_type)
            
            beta_sample = []
            for s in samples:
                beta_sample.append(beta_all[s])
            # beta_sample.append(beta_pop)
            
            new_ballots, pop_score = predict_votes(beta_sample, features, X_test[0])
            
            # with open(f"PL_data/{ts}-{trial}-{test_type}-beta.npy", 'wb') as f:
            #     np.save(f,beta_all)
            #     np.save(f,new_ballots)
            #     np.save(f,pop_score)
            
            # compute predictions, then aggregate
            wn_pl, sn_pl = plurality_winner(new_ballots)
            wn_B, sn_B = Borda_winner(new_ballots)
            wn_m, sn_m = maximin_winner(new_ballots)
            
            print(ts, trial, wn_pl, wn_B, wn_m)
            
            pl_acc.append(match_winners(wn_pl, w_pl))
            B_acc.append(match_winners(wn_B, w_B))
            m_acc.append(match_winners(wn_m, w_m))
            
            pl_KT.append(kendalltau(s_pl, sn_pl)[0])
            B_KT.append(kendalltau(s_B, sn_B)[0])
            m_KT.append(kendalltau(s_m, sn_m)[0])
            
            # compare to population prediction
            pop_acc.append([test_type, ts, trial, np.argmax(pop_score) in w_pl, np.argmax(pop_score) in w_B, np.argmax(pop_score) in w_m])
            pop_KT.append([kendalltau(s_pl, pop_score)[0], kendalltau(s_B, pop_score)[0], kendalltau(s_m, pop_score)[0]])
    
        with open(f"PL_data/{test_type}-{ts}-results.npy", 'wb') as f:
            np.save(f, np.array(beta_all))
            np.save(f, np.array(pl_acc))
            np.save(f, np.array(B_acc))
            np.save(f, np.array(m_acc))
            np.save(f, np.array(pl_KT))
            np.save(f, np.array(B_KT))
            np.save(f, np.array(m_KT))
            np.save(f, np.array(pop_acc))
            np.save(f, np.array(pop_KT))

#%%
# x = training_size
# y1 = []
# yerr1 = []
# y2 = []
# yerr2 = []
# y3 = []
# yerr3 = []
# lens = [trials]*len(x)

# acc1 = []
# acc2 = []
# acc3 = []

# init = 0
# for i in range(len(lens)):
#     print(np.sum(pl_acc[init:init+lens[i]]))
#     print(np.sum(B_acc[init:init+lens[i]]))
#     print(np.sum(m_acc[init:init+lens[i]]))
#     acc1.append(np.sum(pl_acc[init:init+lens[i]])/lens[i])
#     acc2.append(np.sum(B_acc[init:init+lens[i]])/lens[i])
#     acc3.append(np.sum(m_acc[init:init+lens[i]])/lens[i])
    
#     y1.append(np.mean(pl_KT[init:init+lens[i]]))
#     yerr1.append(np.std(pl_KT[init:init+lens[i]]))
#     y2.append(np.mean(B_KT[init:init+lens[i]]))
#     yerr2.append(np.std(B_KT[init:init+lens[i]]))
#     y3.append(np.mean(m_KT[init:init+lens[i]]))
#     yerr3.append(np.std(m_KT[init:init+lens[i]]))
    
#     init += lens[i]
    
# plt.errorbar(x, y1, yerr=yerr1, label='Plurality')
# plt.errorbar(x, y2, yerr=yerr2, label='Borda')
# plt.errorbar(x, y3, yerr=yerr3, label='maximin')

# plt.legend(loc='lower right')

#%% predictor
# predicted = []

# def KT_brute(r1, r2):
#     """
#     Parameters
#     ----------
#     r1 : ranking 1
#     r2 : ranking 2
#         both permutations of range(m)

#     Returns
#     -------
#     Kendall Tau distance (not normalized) between the two rankings
#         rather a brute force implementation O(m^2), the best we could do is O(m log m)
#     returns an int between 0 and m(m-1)/2
#     """
#     m = len(r1)
#     KT = 0
#     for i in range(m):
#         for j in range(i+1, m):
#             if((r1.index(i)-r1.index(j)) * (r2.index(i)-r2.index(j)) < 0):
#                 KT += 1
#     return KT

# cnt = 0
# for i,X_ in enumerate(X_train):
#     gamma = [np.dot(beta, x) for x in X_]
#     y_pred = np.argsort(-1* np.array(gamma))
#     predicted.append(y_pred)
#     cnt += KT_brute(list(y_pred), list(y_train[i]))
        