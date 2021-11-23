#TODO: Must merge this with LP_learn
'''
It is of essential importance that I don't run both
'''

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

# leave this part for later
# split = train_test_split(len(df.agent.unique()))

# def create_data_future(df, agents, features, train=True, tests = None):  
#     # this is incomplete and will not work
#     # the idea of splitting the agents into train-test is crappy at best
#     # let's focus on the general thing later on
#     # refer to create_data for now, renaming this as create_data_future
    
#     # train = True creates training set, test = True creates testing set
#     # testset = None chooses random training-testing break
#     # otherwise have to specify
#     #   e.g.tests = ['titanic','tripset0','tripset1','tripset2','tripset3']
#     X = []
#     Y = []
#     for agent in range(len(agents)):
#         # agent = np.random.randint(len(agents))
#         nf = df[df['WorkerId'] == agents[agent]]
        
#         # tests = ['titanic','tripset0','tripset1','tripset2','tripset3']
#         if(train==True):
#             if(tests):
#                 nf = nf[~(nf['type'].isin(tests))]
#         else:
#             if(tests):
#                 nf = nf[(nf['type'].isin(tests))]
#         nf.fillna(0, inplace=True)
#         X_train = []
#         y_train = []
#         for i in range(17):
#             #just break them down into pairs and do the same thing
#             # run a nested loop (for i in range(len): for j in range(i+1,len))
#             x = np.array(nf[nf['scenario_no']==i][features])
#             # y = np.array(nf[nf['scenario_no']==i]['score']).argmax() # just keep top
#             y = (-1*np.array(nf[nf['scenario_no']==i]['score'])).argsort() # keep whole ranking
            
#             # include all rank-breaking pairs into training set
#             for m1 in range(len(y)):
#                 for m2 in range(m1+1, len(y)):
#                     # print(x[[m1,m2]])
#                     # print(y[[m1,m2]])
#                     # X_train.append(x[[m1,m2]])
#                     X_train.append([dict(zip(features,x[m1])), dict(zip(features,x[m2]))])
#                     y_train.append(np.argsort(y[[m1,m2]]))
                    
#         X.append(X_train)
#         Y.append(y_train)
                            
#     return X, Y

def create_aggregation_data(df, agents, features, split = 0.8):  
    # return both Xtrain, Ytrain and Xtest, Ytest at the same time
    # this is for the aggregation bit
    # keep tripset data as test, others as train
    
    X_train_all = []
    Y_train_all = []
    
    X_test_all = []
    Y_test_all = []
    
    types_test_all = []
    
    for agent in range(len(agents)):
    # updating this to df['agent'] compared to WorkerId before
        nf = df[df['agent'] == agents[agent]]
        
        nf.fillna(0, inplace=True)
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        types_test = []
        for i in range(17):
            #just break them down into pairs and do the same thing
            # run a nested loop (for i in range(len): for j in range(i+1,len))
            tp = nf[nf['scenario_no']==i].type.values[0]
            # print(tp)
            x = np.array(nf[nf['scenario_no']==i][features])
            # y = np.array(nf[nf['scenario_no']==i]['score']).argmax() # just keep top
            y = (-1*np.array(nf[nf['scenario_no']==i]['score'])).argsort() # keep whole ranking
            # include all rank-breaking pairs into training set
            
            for m1 in range(len(y)):
                for m2 in range(m1+1, len(y)):
                    # print(x[[m1,m2]])
                    # print(y[[m1,m2]])
                    # X_train.append(x[[m1,m2]])
                    if('tripset' not in tp):
                        X_train.append(x[m1] - x[m2])
                        y_train.append(np.argsort(y[[m1,m2]])[0])
                    else:
                        X_test.append(x[m1] - x[m2])
                        y_test.append(np.argsort(y[[m1,m2]])[0])
                        types_test.append(tp)
            
        X_train_all.append(X_train)
        Y_train_all.append(y_train)
        X_test_all.append(X_test)
        Y_test_all.append(y_test)
        types_test_all.append(types_test)
                            
    return X_train_all, Y_train_all, X_test_all, Y_test_all, types_test_all

# def create_aggregation_data(df, agents, features):  
#     # return both Xtrain, Ytrain and Xtest, Ytest at the same time
#     # this is for the aggregation bit
#     # keep tripset data as test, others as train
    
#     X_train_all = []
#     Y_train_all = []
    
#     X_test_all = []
#     Y_test_all = []
#     types_test_all = []
#     for agent in range(len(agents)):
#         # updating this to df['agent'] compared to WorkerId before
#         nf = df[df['agent'] == agents[agent]]
        
#         nf.fillna(0, inplace=True)
#         X_train = []
#         y_train = []
#         X_test = []
#         y_test = []
#         types_test = []
#         for i in range(17):
#             #just break them down into pairs and do the same thing
#             # run a nested loop (for i in range(len): for j in range(i+1,len))
#             tp = nf[nf['scenario_no']==i].type.values[0]
#             # print(tp)
#             x = np.array(nf[nf['scenario_no']==i][features])
#             # y = np.array(nf[nf['scenario_no']==i]['score']).argmax() # just keep top
#             y = (-1*np.array(nf[nf['scenario_no']==i]['score'])).argsort() # keep whole ranking
#             # include all rank-breaking pairs into training set
            
#             for m1 in range(len(y)):
#                 for m2 in range(m1+1, len(y)):
#                     # print(x[[m1,m2]])
#                     # print(y[[m1,m2]])
#                     # X_train.append(x[[m1,m2]])
#                     if('tripset' not in tp):
#                         X_train.append([dict(zip(features,x[m1])), dict(zip(features,x[m2]))])
#                         y_train.append(np.argsort(y[[m1,m2]]))
#                     else:
#                         X_test.append([dict(zip(features,x[m1])), dict(zip(features,x[m2]))])
#                         y_test.append(np.argsort(y[[m1,m2]]))
#                         types_test.append(tp)
        
#         X_train_all.append(X_train)
#         Y_train_all.append(y_train)
#         X_test_all.append(X_test)
#         Y_test_all.append(y_test)
#         types_test_all.append(types_test)
                            
#     return X_train_all, Y_train_all, X_test_all, Y_test_all, types_test_all

def match_winners(w1, w2):
    com = set(w1).intersection(set(w2))
    if(len(com) > 0):
        return 1
    else:
        return 0

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
                # cnt[1] is no.of pairs where y increases if feat increases
                # cnt[2] is no. of pairs where y increases if feat decreases
                cnt[1] += 1 if (np.sign((pair[0][feat]-pair[1][feat])*(ypair[0]-ypair[1]))>0) else 0
                cnt[2] += 1 if (np.sign((pair[0][feat]-pair[1][feat])*(ypair[0]-ypair[1]))<0) else 0
        
        cnt1 = cnt[1]/cnt[0] if cnt[0]>0 else 0
        cnt2 = cnt[2]/cnt[0] if cnt[0]>0 else 0
        
        # cnt1, cnt2 has ratio of how  many pairs go whichever direction
        # only considers pairs where feature value for f is different
        # 1 -> down, 2 -> up
        cnts_feat.append([cnt1, cnt2])
        # print(feat,cnt[0])
        
    return cnts_feat

def LP_learn(x, y, features):
    x_new = copy.deepcopy(x)
    y_new = copy.deepcopy(y)
    
    feat_new = copy.deepcopy(features)
    
    f_list = []
    
    it = 0
    # currently stopping iterations when data exhausted
    while(len(x_new)>0):     
        # print('\t hello ',it)
        it += 1
        cnts_feat = LP_learn_helper(x_new, y_new, feat_new)
        top = 0
        flag = 0
        top_f = -1
        for f, feat in enumerate(feat_new):
            if(cnts_feat[f][0]>top):
                top = cnts_feat[f][0]
                top_f = f
                flag = 0
            if(cnts_feat[f][1]>top):
                top = cnts_feat[f][1]
                top_f = f
                flag = 1
                
        # top_f has most impactful feature for this iteration
        # flag=0 -> down, flag=1 -> up
        
        f_list.append((feat_new[top_f],flag,cnts_feat[top_f]))
        # print(features[top_f],flag,cnts_feat[top_f])
        # print(cnts_feat[top_f])
        
        # remove features with equal top_f values
        x_update = []
        y_update = []
        for ip, pair in enumerate(x_new):
            if(pair[0][feat_new[top_f]] == pair[1][feat_new[top_f]]):
                x_update.append(pair)
                y_update.append(y_new[ip])
                
        x_new = x_update
        y_new = y_update
        feat_new = [ff for ff in feat_new if not(ff==feat_new[top_f])]
    
    return f_list

#%%

# since we're training on everything, let's test on everything
# we'll call this the training accuracy

# now we're not actually training on everything, we're splitting train-test sets
# we can still compute training and testing accuracy though

def test_pair(x1, x2, features, lp):
    '''
    x1, x2 should be a pair of entries
    pass features since we might need it
    in lp, keep track of flag
    
    if cannot break tie according to lp, break tie randomly
    
    if x1 pref x2, return 0
    else return 1
    '''
    for node in lp:
        xf1 = x1[node[0]]
        xf2 = x2[node[0]]
        flag = node[1]
        
        if(xf1==xf2):
            continue
        else:
            if(flag == 0):
                return xf1 > xf2
            else:
                return xf1 < xf2
    
    # if tie all through, return randomly
    return np.random.choice([True,False])


#%%
def KT_dist(rank, lp, features):
    # rank is a dict
    # lp is a list of tuples
    
    # first, fill the lp to Nearest Neighbors
    lp_rank = [row[0] for row in lp]
    
    for val in rank.values():
        if(val not in lp_rank):
            lp_rank.append(val)
    
    lp_order = []
    for key in rank.keys():
        lp_order.append(lp_rank.index(rank[key]))
        
    return kendalltau(np.arange(7), lp_order)



#%%
def permutation(lst):
    """
    function to create permutations of a given list
        supporting function for ranking_count
    reference: https://www.geeksforgeeks.org/generate-all-the-permutation-of-a-list-in-python/
    """
    if(len(lst) == 0):
        return []
    if(len(lst) == 1):
        return [lst]
    l = []   
    for i in range(len(lst)): 
       m = lst[i] 
       remLst = lst[:i] + lst[i+1:] 
       for p in permutation(remLst): 
           l.append([m] + p) 
    return l
    
#%% create LPs
# create all 7P4 * 2^4 LPs 
# these will be slightly different than the ones we learnt
# because this will not have the third and fourth element. so careful if you
#   ever want to match them

def all_binary(feats):
    if(len(feats) == 1):
        return [[feats[0],0], [feats[0],1]]
    l = []
    for i in range(2):
        f = feats[0]
        remfeats = feats[1:]
        all_bin = all_binary(remfeats)
        # print(all_bin)
        for ab in all_bin:
            # print(ab)
            l.append([f,0] + ab)
            l.append([f,1] + ab)
    return l

def tuple_to_lp(l):
    n = int(len(l)/2)
    return [(l[2*i],l[2*i+1]) for i in range(n)]

def y_to_score(y_test, m=3):
    # this needs to be different from before
    lst = []
    for i1 in range(m):
        for i2 in range(i1+1,m):
            lst.append([i1,i2])
    scores = np.zeros(m)
    for i,yy in enumerate(y_test): # this is where the change is 
    # because, in LP it was [0,1], [1,0]. Now it is 0, 1 respectively
        scores[lst[i][0]] += yy
        scores[lst[i][1]] += 1 - yy
    return scores

if __name__ == '__main__':
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
    
    # However, we need this for alternative learning
    
    X = df[features].values
    min_max_scaler = preprocessing.MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    df[features] = X_minmax
    
    #%%
    agents = df['agent'].unique()
        
    # This to are for older function
    # X, Y = create_data(df, agents, features)
    # X_test, Y_test = create_data(df, agents, features, train = False)
    
    X, Y, X_test, Y_test, tps = create_aggregation_data(df, agents, features)

#%% model choosing
    logreg = True
    mlp1 = False
    mlp2  = False
    
#%% logistic regression
    
    if(logreg):
            
        training_acc = []
        testing = True
        testing_acc = []
        
        Y_out = []
        
        for i,x in enumerate(X):
            if(np.sum(Y[i]) == len(Y[i]) or np.sum(Y[i]) == 0):
                training_acc.append(np.nan)
                if(testing):
                    testing_acc.append(np.nan)
                    Y_out.append([]) # how to treat nan?
                continue
            clf = LogisticRegression(random_state=0).fit(x, Y[i])
            training_acc.append(clf.score(x, Y[i]))
            if(testing):
                testing_acc.append(clf.score(X_test[i], Y_test[i]))
                y_out = clf.predict(X_test[i])
                Y_out.append(y_out)
        
        print(np.nanmean(training_acc))
        if(testing):
            print(np.nanmean(testing_acc))
        
    #%% 1 layer MLP
    if(mlp1):
        
        Acc1 = []
        Acc12 = []
        
        for l1 in range(4,21,4):
            print('hidden layer = ', l1)    
        
            training_acc = []
            testing = True
            testing_acc = []
            
            Y_out = []
        
            for i,x in enumerate(X):
                if(np.sum(Y[i]) == len(Y[i]) or np.sum(Y[i]) == 0):
                    training_acc.append(np.nan)
                    if(testing):
                        testing_acc.append(np.nan)
                        Y_out.append([])
                    continue
                clf = MLPClassifier(alpha=1e-5,hidden_layer_sizes=(l1), 
                                    random_state=1)
                clf.fit(x, Y[i])
                training_acc.append(clf.score(x, Y[i]))
                if(testing):
                    testing_acc.append(clf.score(X_test[i], Y_test[i]))
                    y_out = clf.predict(X_test[i])
                    Y_out.append(y_out)
            
            print(np.nanmean(training_acc))
            Acc1.append(np.nanmean(training_acc))
            if(testing):
                print(np.nanmean(testing_acc))
                Acc12.append(np.nanmean(testing_acc))

    #%% 2 layer MLP
    
    if(mlp2):
        
        Acc = []
        Acc2 = []
        
        for l1 in range(4,21,4):
            for l2 in range(4,13,4):
                print('hidden layers = ', l1, l2)    
            
                training_acc = []
                testing = True
                testing_acc = []
                
                Y_out = []
                
                for i,x in enumerate(X):
                    if(np.sum(Y[i]) == len(Y[i]) or np.sum(Y[i]) == 0):
                        training_acc.append(np.nan)
                        if(testing):
                            testing_acc.append(np.nan)
                            Y_out.append([])
                        continue
                    clf = MLPClassifier(alpha=1e-5,hidden_layer_sizes=(l1, l2), 
                                        random_state=1)
                    clf.fit(x, Y[i])
                    training_acc.append(clf.score(x, Y[i]))
                    if(testing):
                        testing_acc.append(clf.score(X_test[i], Y_test[i]))
                        y_out = clf.predict(X_test[i])
                        Y_out.append(y_out)
                
                print(np.nanmean(training_acc))
                Acc.append(np.nanmean(training_acc))
                if(testing):
                    print(np.nanmean(testing_acc))
                    Acc2.append(np.nanmean(testing_acc))
    
    #%% Compute the winners for plurality and borda
    no_tripsets = 28
    m = 3
    tripsets = [f'tripset{i}' for i in range(no_tripsets)]
    
    scores = np.zeros((no_tripsets, m))
    pl_scores = np.zeros((no_tripsets, m))
    C_scores = np.zeros((no_tripsets, m))
    
    # this is for ground truth
    for i,ytest in enumerate(Y_test):
        # print(ytest)
        s = y_to_score(ytest)
        for j in range(m):
            scores[tripsets.index(tps[i][0])][j] += s[j] # scores have Borda score
                
        pl_scores[tripsets.index(tps[i][0])][np.argmax(s)] += 1 # pl_scores have plurality scores
    
    new_scores = np.zeros((no_tripsets, m))
    new_pl_scores = np.zeros((no_tripsets, m))
#%%    
    # this is for predictions
    for i,yout in enumerate(Y_out):
        if(len(yout)==0):
            continue
        s = y_to_score(yout)
        # print(yout)
        for j in range(m):
            new_scores[tripsets.index(tps[i][0])][j] += s[j]
        new_pl_scores[tripsets.index(tps[i][0])][np.argmax(s)] += 1
            
    w1 = np.argmax(scores, axis=1)
    w2 = np.argmax(new_scores, axis=1)
    
    pl_w1 = np.argmax(pl_scores, axis = 1)
    pl_w2 = np.argmax(new_pl_scores, axis = 1)
    #%%
    
    # lp = LP_learn(X_all, Y_all, features)
    
    # #%%
    
    # accuracy_rec2 = np.zeros(len(X))
    # for ix, x in enumerate(X):
    #     y = Y[ix]
    #     acc = 0
    #     for ip,pairs in enumerate(x):
    #         yi = y[ip]  # you just need y for testing 
    #                     # yi[0] < yi[1] if pairs[0] pref pairs[1]
    #         yret = test_pair(pairs[0], pairs[1], features, lp)
    #                     # yret == True if pairs[0] pref pairs[1]
                    
    #         if(yret):
    #             if(yi[0] > yi[1]):
    #                 acc += 1
    #         else:
    #             if(yi[0] < yi[1]):
    #                 acc += 1
    #     accuracy_rec2[ix] = acc
                
    # print("average for global lp ", np.mean(accuracy_rec2))
    
