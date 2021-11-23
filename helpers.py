from random import randint
from math import exp
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import preprocessing,tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,MaxAbsScaler,normalize
from sklearn.inspection import permutation_importance
from sklearn.tree import export_graphviz,DecisionTreeClassifier
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
from graphviz import Source
from os import system,path
import pickle
from random import shuffle, choice,uniform
from tqdm import tqdm
import matplotlib
import seaborn as sns
from scipy.stats import rankdata,kendalltau
from playsound import playsound
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from nltk import stem
from nltk.corpus import stopwords
from spacy.lang.en import English
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import SGDClassifier
from joblib import dump,load
from voting_rules import *
from lptree import *
from importlib import reload


cmap = matplotlib.cm.get_cmap('viridis')
colors = {
    'age':cmap(0.1),
    'gender':cmap(0.2),
    'health':cmap(0.3),
    'income':cmap(0.4),
    'education':cmap(0.5),
    'dependents':cmap(0.6),
    'surv_with':cmap(0.7),
    'surv_dif':cmap(0.8),
    
    # for airplane here
    'agent affected':cmap(0),
    'population at risk':cmap(0.2),
    'population that gain':cmap(0.4),
    'life decrease':cmap(0.6),
    'decrease chance':cmap(0.8),
    'gain':cmap(1),
}

manycolors = {
    'agent affected': cmap(0.045),
    'population at risk': cmap(0.09),
    'population that gain': cmap(0.135),
    'life decrease': cmap(0.18),
    'decrease chance': cmap(0.225),
    'gain': cmap(0.27),
    'agent affected,population at risk': cmap(0.315),
    'agent affected,population that gain': cmap(0.36),
    'agent affected,life decrease': cmap(0.405),
    'agent affected,decrease chance': cmap(0.45),
    'agent affected,gain': cmap(0.495),
    'population at risk,population that gain': cmap(0.54),
    'population at risk,life decrease': cmap(0.585),
    'population at risk,decrease chance': cmap(0.63),
    'population at risk,gain': cmap(0.675),
    'population that gain,life decrease': cmap(0.72),
    'population that gain,decrease chance': cmap(0.765),
    'population that gain,gain': cmap(0.81),
    'life decrease,decrease chance': cmap(0.855),
    'life decrease,gain': cmap(0.9),
    'decrease chance,gain': cmap(0.945),
}

age_dict = {
    '27 year old': 27,
    'young child': 8,
    '18 year old': 18,
    '23 year old': 23,
    'middle aged': 32,
    'senior citizen': 70
}

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

edu_dict = {
    'nan': 0,
    'Middle school graduate': 0,
    'High school graduate': 1,
    'College graduate': 2,
    'Graduate degree': 3
}

user_age = {
    np.nan: 0,
    '10~19': 10,
    '20~29': 20,
    '30~39': 30,
    '40~49': 40,
    '50~59': 50,
    '60~69': 60,
    'other': 0,
    'N/A': 0
}

user_gender = {
    np.nan: 0,
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
    'N/A': 0,
    np.nan:0
}

trans_feats = {
    'PREF.age': 'age', 
    'PREF.gender': 'gender', 
    'PREF.health': 'health',
    'PREF.income': 'income', 
    'PREF.survdif': 'surv_dif', 
    'PREF.survwith': 'surv_with',
    'PREF.dependents': 'dependents', 
    'PREF.education': 'education'
}

def setdata(is_all, imp_vars):

    if is_all: prefix = 'all'
    else: prefix = 'few'
    
    genX = imp_vars['genX_'+prefix]
    genY = imp_vars['genY_'+prefix]
    X = imp_vars['X_'+prefix]
    Y = imp_vars['Y_'+prefix]
    features = imp_vars['features_'+prefix]
    df_user = imp_vars['df_user_'+prefix]
    df_cleaned = imp_vars['df_cleaned_'+prefix]
    user_d = imp_vars['user_d_'+prefix]

    return is_all,prefix,genX,genY,X,Y,features,df_user,df_cleaned,user_d

def setagent(prefix,imp_vars, setdefault=False):

    if setdefault: 
        prefix = 'all'
        genX = imp_vars['genX']
        genY = imp_vars['genY']
        X = imp_vars['X']
        Y = imp_vars['Y']
        features = imp_vars['features']
    else:
        genX = imp_vars['genX_'+prefix]
        genY = imp_vars['genY_'+prefix]
        X = imp_vars['X_'+prefix]
        Y = imp_vars['Y_'+prefix]
        user_d = imp_vars['user_d_'+prefix]
        features = imp_vars['features'][1:]

    df_user = imp_vars['df_user']
    df_cleaned = imp_vars['df_cleaned']
    user_d = imp_vars['user_d']
    return set,prefix,genX,genY,X,Y,features,df_user,df_cleaned,user_d

def clean_data1(df,is_all,numoptions = 4):
    '''
    takes in the read in df and cleans up the features depending on data type
    returns: 
        df_cleaned (cleaned alternative features by user)
        df_user (cleaned user features)
    '''
    df_cleaned = pd.DataFrame()
    df_usertmp = pd.DataFrame()

    for i in range(0,len(df)-numoptions+1,numoptions):  
        for j in range(numoptions):
            options = {}
            options['TurkerID'] = [df.iloc[i+j]['WorkerId']]
            options['scenario'] = int(df.iloc[i+j]['scenario_no'])
            #age
            x = df.iloc[i+j]['age']
            # if(x.isnumeric()):
            age = int(x)
            # else:
            #     age = age_dict[x]
            options['age'] = [age]
            #gender
            x = str(df.iloc[i+j]['gender'])
            options['gender'] = [gender_dict[x]]
            #health
            x = str(df.iloc[i+j]['health'])
            x = x.replace('\t','')
            options['health'] = [health_dict[x]]
            # income
            x = str(df.iloc[i+j]['income level'])
            options['income'] = [income_dict[x]]
            # education
            if is_all:
                if 'education level' in df:
                    x = str(df.iloc[i+j]['education level'])
                    options['education'] = [edu_dict[x]]
            # number of dependents
                if 'number of dependents' in df:
                    x = df.iloc[i+j]['number of dependents']
                    if not pd.isnull(x):
                        options['dependents'] = int(x)
            #survival chance
            options['surv_with'] = int(df.iloc[i+j]['survival with jacket'])
            # options['surv_without'] = int(df.iloc[i+j]['survival without jacket'])
            options['surv_dif'] = options['surv_with'] - int(df.iloc[i+j]['survival without jacket'])
            #score
            options['score'] = df.iloc[i+j]['score']

            df_cleaned = df_cleaned.append(pd.DataFrame.from_dict(options))
        
        # user info
        user_options = {}
        user_options['TurkerID'] = [df.iloc[i]['WorkerId']]
        if 'Answer.agegroup' in df:
            x = str(df.iloc[i]['Answer.agegroup'])
            if not x == 'nan':
                user_options['USER.age'] = [user_age[x]]
        if 'Answer.education' in df:
            x = str(df.iloc[i]['Answer.education'])
            if not x == 'nan':
                user_options['USER.education'] = [user_education[x]]
        if 'Answer.gender' in df:
            x = str(df.iloc[i]['Answer.gender'])
            if not x == 'nan':
                user_options['USER.gender'] = [user_gender[x]]

        user_options['PREF.age'] = int(df.iloc[i]['Answer.age_importance'])
        user_options['PREF.gender'] = int(df.iloc[i]['Answer.gender_importance'])
        user_options['PREF.health'] = int(df.iloc[i]['Answer.health_importance'])
        user_options['PREF.income'] = int(df.iloc[i]['Answer.income_importance'])
        user_options['PREF.survdif'] = int(df.iloc[i]['Answer.survdif_importance'])
        user_options['PREF.survwith'] = int(df.iloc[i]['Answer.survwith_importance'])
        
        if is_all:
            user_options['PREF.dependents'] = int(df.iloc[i]['Answer.dependents_importance'])
            # user_options['PREF.education'] = int(df.iloc[i]['Answer.education_importance'])

        df_usertmp = df_usertmp.append(pd.DataFrame.from_dict(user_options))

    return df_cleaned,df_usertmp

def clean_userdata(df_usertmp):
    '''
    takes in usertmp 
    returns:
        cleaned user data sorted by user id as index
        and dropped duplicates after collecting as much data as possible
    '''
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
    return df_user

def make_pairwise(df_cleaned,in_user,plant=False,type='four'):
    '''
    returns:
        all_X,all_Y: general features/outcomes
        X,Y: features/outcomes sorted by user
        user_data: coressponding user data for X and Y
    '''
    features = [f for f in list(df_cleaned.columns) 
            if f not in ['TurkerID','scenario','score','question#','proposal#']]
    
    # default is four
    pairs = [(0,1),(0,2),(0,3),
            (1,2),(1,3),(2,3)]
    if type == 'pairs':
        pairs = [(0,1)]
    if type == 'triples':
        pairs = [(0,1),(0,2),(1,2)]
    genX = []
    genY = []
    X = []
    Y = []
    user_d = []

    genX_aff = []
    genY_aff = []
    X_aff = []
    Y_aff = []
    user_d_aff = []

    genX_nei = []
    genY_nei = []
    X_nei = []
    Y_nei = []
    user_d_nei = []

    genX_both = []
    genY_both = []
    X_both = []
    Y_both = []
    user_d_both = []


    sname = 'scenario'
    if plant: sname = 'question#'	

    for idx, oneguy in df_cleaned.groupby('TurkerID'):
        turker_x = []
        turker_y = []
        turker_x_aff = []
        turker_y_aff = []
        turker_x_nei = []
        turker_y_nei = []
        turker_x_both = []
        turker_y_both = []

        user_data = in_user.loc[idx]
        for ii, nf in oneguy.groupby(sname):
            nf.fillna(0, inplace=True)
            for p in pairs:
                if(nf.iloc[p[0]]['score'] == nf.iloc[p[1]]['score']):
                    continue #ignore ties for now
                # for now only look at parts with valid user data
                # if user_data.size < 3: continue
                feature_diff = nf.iloc[p[0]][features].values - nf.iloc[p[1]][features].values
                if(nf.iloc[p[0]]['score'] > nf.iloc[p[1]]['score']):
                    turker_x.append(feature_diff)
                    turker_y.append(1)
                    genX.append(feature_diff)
                    genY.append(1)
                else:
                    turker_x.append(feature_diff)
                    turker_y.append(0)
                    genX.append(feature_diff)
                    genY.append(0)
                
                if plant:
                    # if the agent affected is different value
                    fd = feature_diff[1:]
                    if feature_diff[0] != 0:
                        if(nf.iloc[p[0]]['score'] > nf.iloc[p[1]]['score']):
                            turker_x_aff.append(fd)
                            turker_y_aff.append(1)
                            genX_aff.append(fd)
                            genY_aff.append(1)
                        else:
                            turker_x_aff.append(fd)
                            turker_y_aff.append(0)
                            genX_aff.append(fd)
                            genY_aff.append(0)
                    # if the agent affected in neither
                    elif nf.iloc[p[0]][features].values[0] == 0:
                        if(nf.iloc[p[0]]['score'] > nf.iloc[p[1]]['score']):
                            turker_x_nei.append(fd)
                            turker_y_nei.append(1)
                            genX_nei.append(fd)
                            genY_nei.append(1)
                        else:
                            turker_x_nei.append(fd)
                            turker_y_nei.append(0)
                            genX_nei.append(fd)
                            genY_nei.append(0)
                    elif nf.iloc[p[0]][features].values[0] == 1:
                        if(nf.iloc[p[0]]['score'] > nf.iloc[p[1]]['score']):
                            turker_x_both.append(fd)
                            turker_y_both.append(1)
                            genX_both.append(fd)
                            genY_both.append(1)
                        else:
                            turker_x_both.append(fd)
                            turker_y_both.append(0)
                            genX_both.append(fd)
                            genY_both.append(0)
                    
        # some people gave everything a tie
        if turker_x:
            X.append(turker_x)
            Y.append(turker_y)
            user_d.append(user_data)
        if turker_x_aff:
            X_aff.append(turker_x_aff)
            Y_aff.append(turker_y_aff)
            user_d_aff.append(user_data)
        if turker_x_nei:
            X_nei.append(turker_x_nei)
            Y_nei.append(turker_y_nei)
            user_d_nei.append(user_data)
        if turker_x_both:
            X_both.append(turker_x_both)
            Y_both.append(turker_y_both)
            user_d_both.append(user_data)

    max_scaler = MaxAbsScaler()
    genX = max_scaler.fit_transform(genX)
    if genX_aff: genX_aff = max_scaler.fit_transform(genX_aff)
    if genX_nei: genX_nei = max_scaler.fit_transform(genX_nei)
    if genX_both: genX_both = max_scaler.fit_transform(genX_both)
    
    vv = {  'genX': genX,
            'genY': genY,
            'X': X,
            'Y': Y,
            'user_d': user_d,
            'features': features}
    if plant:
        vv.update({ 'genX_aff': genX_aff,
                    'genY_aff': genY_aff,
                    'X_aff': X_aff,
                    'Y_aff': Y_aff,
                    'user_d_aff': user_d_aff,
                    'genX_nei': genX_nei,
                    'genY_nei': genY_nei,
                    'X_nei': X_nei,
                    'Y_nei': Y_nei,
                    'user_d_nei': user_d_nei,
                    'genX_both': genX_both,
                    'genY_both': genY_both,
                    'X_both': X_both,
                    'Y_both': Y_both,
                    'user_d_both': user_d_both,
                    })
    return vv

def heatmap(data,title,prefix,d_pref):
    annot = False
    if len(data.columns) > 8: size = (20,8)
    else: 
        annot = True
        if len(data.columns) == 8: size = (12,10)
        else: size = (2,12)
    
    plt.figure(figsize=size)

    g=sns.heatmap(data,annot=annot, cmap="PuBuGn")
    plt.yticks(rotation=0)
    g.set_title(title)
    fname = title.lower().replace(' ','_')
    g.get_figure().savefig('/home/inwon/Documents/mturk/'+d_pref+'/visuals/heatmaps/'+fname+'_'+prefix+'.png',bbox_inches = "tight")


def clean_data2(df):
    '''
    Same thing as clean_data1, except i didnt feel like fixing it to be used for this
    one day ill get to it
    '''
    df_cleaned = pd.DataFrame()
    df_usertmp = pd.DataFrame()
    start = True
    for (i,q,p),ff in df.groupby(['WorkerId','Question #','Proposal #']):  
        dd = ff.iloc[0]
        options = {}
        options['TurkerID'] = [str(i)]
        options['question#'] = [int(q)]
        options['proposal#'] = [int(p)]
        # 1 if true 0 if not

        options['agent affected'] = [int(dd['User city'] in dd['Affected city'])]
        
        pc1 = str(dd['C1 Population']).replace(',','')
        pc1 = int(float(pc1))
        pc2 = str(dd['C2 Population']).replace(',','')
        pc2 = int(float(pc2))
        if len(dd['Affected city'])>2:
            pop = pc1+pc2
        elif dd['Affected city'] == 'C1':
            pop = pc1
        elif dd['Affected city'] == 'C2':
            pop = pc2
        options['population at risk'] = [pop]
        options['population that gain'] = [pc1+pc2]
        options['life decrease'] = [int(float(dd['Life decrease']))]
        options['decrease chance'] = [int(float(dd['Decrease chance']))]
        options['gain'] = [int(float(dd['Gain']))]

        options['score'] = [int(dd['Proposal Score'])]
        df_cleaned = df_cleaned.append(pd.DataFrame.from_dict(options))
        
        # user info
        user_options = {}
        user_options['TurkerID'] = str(i)
        x = str(dd['WorkerAge'])
        if not x == 'nan':
            user_options['USER.age'] = [user_age[x]]
        x = str(dd['WorkerEdu'])
        if not x == 'nan':
            user_options['USER.education'] = [user_education[x]]
        x = str(dd['WorkerGender'])
        if not x == 'nan':
            user_options['USER.gender'] = [user_gender[x]]

        if start or str(i) not in df_usertmp['TurkerID'].values:
            df_usertmp = df_usertmp.append(pd.DataFrame.from_dict(user_options))
            start = False

    return df_cleaned,df_usertmp

def smalltable(column_names,column_vals):
    tmpcol = [row[:] for row in column_vals]
    goodcol = []
    for i,v in enumerate(tmpcol):
        ll = []
        for vv in v:
            if type(vv) == float or type(vv) == np.float64:
                ff = str(round(vv,4)).ljust(6,'0')
            else:
                ff = str(vv)
            ll.append(ff)
        goodcol.append(ll)
    assert(len(column_names) == len(goodcol))
    assert(len(set([len(v) for v in goodcol])) == 1)

    allvals = [row[:] for row in goodcol]
    for i,a in enumerate(allvals):
        a.append(column_names[i])
    
    maxlens = [
        max([len(d) for d in v]) for v in allvals
    ]
    
    tt = '|'
    for j,c in enumerate(column_names):
        tt += c.ljust(maxlens[j],' ') +'|'
    tt += '\n|'
    for j,c in enumerate(column_names):
        tt += '-'.ljust(maxlens[j],'-') + '|'
    tt += '\n'
    for i in range(len(goodcol[0])):
        tt += '|'
        for j,v in enumerate(goodcol):
            tt += str(v[i]).ljust(maxlens[j],' ') + '|'
        tt +='\n'
    return tt

def createmodels(d_pref,prefix,transformed = False,justtrans=False):

    srbf_params = {'tol': 0.1, 'random_state': 0, 'probability': True, 'kernel': 'rbf', 'C': 0.8}
    svclin_params = {'random_state':0}
    sgd_params = {'random_state':0}
    if '1' in d_pref:
        rand_param = {'random_state': 0, 'oob_score': True, 'n_estimators': 8, 'min_samples_split': 77, 'min_samples_leaf': 42, 'max_leaf_nodes': 57, 'max_features': 'auto', 'max_depth': 4, 'bootstrap': True }
        tree_param = {'max_depth': 9,'max_features': 'auto','max_leaf_nodes': 50,'min_samples_leaf': 5,'random_state': 0}
        reg_param = {'C': 1,'max_iter': 500,'random_state': 0,'tol': 1}
        s_params = {'C': 0.4,'kernel': 'linear','probability': True,'random_state': 0,'tol': 1e-05}
        
        if 'new' in d_pref:
            rand_param = {'random_state': 0,'oob_score': True,'n_estimators': 20,
                'min_samples_split': 31,
                'min_samples_leaf': 14,
                'max_features': 'auto',
                'max_depth': 5,
                'bootstrap': True}
            tree_param = {'random_state': 0, 'min_samples_leaf': 114, 'max_leaf_nodes': 23, 'max_depth': 7}
            reg_param = {'tol': 1, 'max_iter': 1200, 'C': 2}
            s_params = {'tol': 1e-06,
                    'random_state': 0,
                    'probability': False,
                    'kernel': 'linear',
                    'C': 0.2}
            srbf_params = {'tol': 0.01,
                    'random_state': 0,
                    'probability': True,
                    'kernel': 'rbf',
                    'degree': 5,
                    'C': 3}
        if 'round7' in d_pref:
            rand_param = {'random_state': 0, 'oob_score': True, 'n_estimators': 35, 'min_samples_split': 6, 'min_samples_leaf': 26, 'max_features': 'auto', 'max_depth': 9, 'bootstrap': True}
            tree_param = {'random_state': 0, 'min_samples_leaf': 9, 'max_leaf_nodes': 102, 'max_depth': 6}
            reg_param = {'tol': 1e-05, 'max_iter': 2000, 'C': 3}
            s_params = {'tol': 1e-06, 'random_state': 0, 'probability': False, 'kernel': 'linear', 'C': 0.8}
            srbf_params = {'random_state':0,'kernel':'rbf'}
            svclin_params = {'tol': 0.01,'random_state': 0,'penalty': 'l2','max_iter': 1000,'loss': 'squared_hinge','fit_intercept': True,'dual': False,'C': 1}
            sgd_params = {'tol': 20, 'shuffle': True, 'random_state': 0, 'penalty': 'l2', 'max_iter': 2500, 'loss': 'huber', 'learning_rate': 'adaptive', 'fit_intercept': True, 'eta0': 0.0001, 'early_stopping': True}
        if 'round8' in d_pref:
            reg_param = {'random_state':0}
            svclin_params = {'tol': 0.001,'random_state': 0,'penalty': 'l2','max_iter': 2500,'loss': 'squared_hinge','fit_intercept': True,'dual': False,'C': 1}
    if '2' in d_pref:
        if prefix == 'all':
            # regular
            rand_param = {'random_state': 0, 'oob_score': True, 'n_estimators': 40, 'min_samples_split': 30, 'min_samples_leaf': 15, 'max_features': 'auto', 'max_depth': 7, 'bootstrap': True}
            reg_param = {'tol': 0.01, 'max_iter': 1000, 'C': 0.8}
            s_params = {'tol': 1, 'random_state': 0, 'probability': True, 'kernel': 'linear', 'C': 0.8}
            tree_param = {'max_depth': 5, 'max_leaf_nodes': 16, 'min_samples_leaf': 7, 'random_state': 0}
            if transformed:
                # this was for including single features 
                rand_param = {'random_state': 0, 'oob_score': True, 'n_estimators': 15, 'min_samples_split': 24, 'min_samples_leaf': 11, 'max_features': 'auto', 'max_depth': 9, 'bootstrap': True}
                tree_param = {'random_state': 0, 'min_samples_leaf': 18, 'max_leaf_nodes': 26, 'max_depth': 18}
                reg_param = {'tol': 0.1, 'max_iter': 1500, 'C': 0.4}
                s_params = {'tol': 1, 'random_state': 0, 'probability': False, 'kernel': 'linear', 'C': 0.8}
                srbf_params = {'tol': 0.01, 'random_state': 0, 'probability': True, 'kernel': 'rbf', 'degree': 1, 'C': 1}
            
            if justtrans:
                # and these are for just comparing double features
                tree_param = {'random_state': 0, 'min_samples_leaf': 20, 'max_leaf_nodes': 170, 'max_depth': 14}
                rand_param = {'random_state': 0, 'oob_score': True, 'n_estimators': 10, 'min_samples_split': 65, 'min_samples_leaf': 7, 'max_features': 'auto', 'max_depth': 12, 'bootstrap': True}
                reg_param = {'C': 0.05, 'max_iter': 500, 'tol': 20}
                s_params = {'tol': 0.1, 'random_state': 0, 'probability': True, 'kernel': 'linear', 'C': 0.4}
                srbf_params = {'tol': 1e-06,'random_state': 0,'probability': True,'kernel': 'rbf','degree': 3,'C': 0.6}

        if prefix == 'aff':
            tree_param = {'random_state': 0, 'min_samples_leaf': 15, 'max_leaf_nodes': 12, 'max_depth': 9}
            rand_param = {'random_state': 0, 'oob_score': True, 'n_estimators': 50, 'min_samples_split': 27, 'min_samples_leaf': 9, 'max_features': 'auto', 'max_depth': 6, 'bootstrap': True}
            reg_param = {'C': 0.2, 'max_iter': 1000, 'tol': 0.1}
            s_params = {'C': 0.2, 'kernel': 'linear', 'probability': True, 'random_state': 0, 'tol': 0.1}
        elif prefix =='nei':
            tree_param = {'random_state': 0, 'min_samples_leaf': 4, 'max_leaf_nodes': 28, 'max_depth': 19}
            rand_param = {'random_state': 0, 'oob_score': True, 'n_estimators': 15, 'min_samples_split': 17, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 8, 'bootstrap': True}
            reg_param = {'tol': 1e-06, 'max_iter': 1000, 'C': 0.8}
            s_params = {'tol': 1e-06, 'random_state': 0, 'probability': True, 'kernel': 'linear', 'C': 0.6}
            srbf_params = {'tol': 1, 'random_state': 0, 'probability': False, 'kernel': 'rbf', 'degree': 1, 'C': 0.8}
        elif prefix =='both':
            tree_param = {'random_state': 0, 'min_samples_leaf': 38, 'max_leaf_nodes': 103, 'max_depth': 7}
            rand_param = {'random_state': 0, 'oob_score': True, 'n_estimators': 25, 'min_samples_split': 10, 'min_samples_leaf': 20, 'max_features': 'auto', 'max_depth': 11, 'bootstrap': True}
            reg_param = {'tol': 0.1, 'max_iter': 2000, 'C': 0.6}
            s_params = {'tol': 1, 'random_state': 0, 'probability': True, 'kernel': 'linear', 'C': 0.6}
            srbf_params = {'tol': 0.0001, 'random_state': 0, 'probability': True, 'kernel': 'linear', 'C': 2}
            
    rfore = RandomForestClassifier(**rand_param)
    dtree = DecisionTreeClassifier(**tree_param)
    logreg = LogisticRegression(**reg_param)
    supvec = SVC(**s_params)
    supvec_rbf = SVC(**srbf_params)
    svclinear = LinearSVC(**svclin_params)
    sgd = SGDClassifier(**sgd_params)

    return rfore,dtree,logreg,supvec,supvec_rbf,svclinear,sgd

def random_tune(params,model,defmodel,genX,genY,goalscore = None,loops = 100,iter = 100):
    trainX,testX,trainY,testY = train_test_split(genX,genY,random_state = 0)
    search = RandomizedSearchCV(model,params,n_iter = 100)

    score = 0
    if not goalscore:
        goalscore = defmodel.fit(trainX,trainY).score(testX,testY)

    for i in tqdm(range(loops)):
        if score > goalscore: break
        search = RandomizedSearchCV(model(),params,n_iter = iter)
        search.fit(genX,genY)
        pp = search.best_params_
        score = model(**pp).fit(trainX,trainY).score(testX,testY)        

    return score, pp

def nonlinear(features,genX,justtrans=False):
    newfeats = features.copy()
    if justtrans:
        newfeats = []
    feat_vals = [dict(zip(features,x)) for x in genX]
    newX = []
    for i,f in enumerate(features):
        for f2 in features[i+1:]:
            newfeats.append(f + ','+f2)
    for vv in feat_vals:
        vals = [1 for n in newfeats]
        for i,feats in enumerate(newfeats):
            fl = feats.split(',')
            for f in fl:
                vals[i] *= vv[f]
        newX.append(vals)
    return newfeats,newX

def notify():
    pp = '/home/inwon/Downloads/humanisrael.mp3'
    if path.exists(pp):
        playsound(pp)
    return

def makeguess(genX,genY,ffs,features,thresh):
    corrects = 0
    for xx,yy in zip(genX,genY):
        vv = dict(zip(features,xx))
        pred = 0
        for f,s in ffs:
            if s > 0:
                if vv[f] > thresh[f]:
                    pred = 1
                    break
            if s < 0:
                if vv[f] < thresh[f]:
                    pred = 1
                    break
        if yy == pred: corrects += 1

    return corrects/len(genX)

nlp = English()
table = str.maketrans(dict.fromkeys(string.punctuation))
stemmer = stem.SnowballStemmer('english')
stopwords = set(stopwords.words('english'))
def clean_text(msg):    # converting messages to lowercase    
    msg = msg.lower()    # removing stopwords    
    msg = msg.translate(table)
    msg = [
        word for word in msg.split() 
        if word not in stopwords]    # using a stemmer    
    msg = " ".join([stemmer.stem(word) for word in msg])
    msg = ' '.join([str(t) for t in nlp(msg)])
    # print(nlp(msg))
    return msg

def match_winners(w1, w2):
    com = set(w1).intersection(set(w2))
    if(len(com) > 0):
        return 1
    else:
        return 0

def pair_accuracy(r1,r2):
    allcount = 0
    corcount = 0
    pairs = []
    for i in range(3):
        for j in range(i+1,3):
            pairs.append((i,j))
    r1pairs = [(r1[p[0]],r1[p[1]]) for p in pairs]
    r2pairs = [(r2[p[0]],r2[p[1]]) for p in pairs]
    return float(len(set(r1pairs) & set(r2pairs)))/len(pairs)

def beta_votes(sample,classifier,alt_indivotes,ops,typ):
    ufeats = ['USER.age','USER.education','USER.gender']
    feats = ['age', 'gender', 'health', 'income', 'dependents', 'surv_with', 'surv_dif']

    oneset = []
    user_vals = []
    indiv_corrcount = 0
    indiv_paircount = 0
    indiv_count = 0
    gt_count = 0
    for id, ww in sample.iterrows():
        scores = []
        for t in ops:
            featvals = [1]
            featvals.extend(list(t[feats]))
            uvals = [1]
            uvals.extend(list(ww[ufeats]))

            prod = np.multiply.outer(uvals,featvals).ravel()
            betas = classifier.coef_[0]

            sco = np.dot(betas,prod)
            scores.append(sco)

        ranking = (-1*np.array(scores)).argsort()
        oneset.append(ranking)

        # checking individual accuracy here
        indiv_winner = ranking[0]

        # this is the person we are looknig at. 
        thisuser = [a for a in alt_indivotes[typ] if a['id'] == id]
        if thisuser: 
            gt_count+=1
            thisuser = thisuser[0]
            realwinner = thisuser['winner']
            realranking = thisuser['rankings']

            indiv_corrcount += kendalltau(realranking,ranking).correlation
            indiv_paircount += pair_accuracy(realranking,ranking)
            if match_winners([realwinner], [indiv_winner]):
                indiv_count += 1

            user_vals.append({
                'correlation':indiv_corrcount,
                'paircount': indiv_paircount,
                'regcount': indiv_count,
                'gt_count': gt_count
            })

    return oneset, user_vals

def pairwise(ops):
    pairs = []
    for i,o in enumerate(ops):
        for j in range(i+1,len(ops)):
            pairs.append((i,j))
    return pairs

def tree_votes(sample,classifier,alt_indivotes,ops,typ,method='reg'):
    ufeats = ['USER.age','USER.education','USER.gender']
    feats = ['age', 'gender', 'health', 'income', 'dependents', 'surv_with', 'surv_dif']

    oneset = []
    user_vals = []
    indiv_corrcount = 0
    indiv_paircount = 0
    indiv_count = 0
    gt_count = 0
    for id, ww in sample.iterrows():
        scores = [0 for o in ops]
        pairs = {}
        # print(pairwise(ops))
        for t in pairwise(ops):
            o1 = ops[t[0]]
            o2 = ops[t[1]]
            feats1,feats2 = np.array([1]),np.array([1])
            feats1 = np.append(feats1,list(o1[feats]))
            feats2 = np.append(feats2,list(o2[feats]))

            featdif = feats1-feats2

            uvals = [1]
            uvals.extend(list(ww[ufeats]))

            prod = np.multiply.outer(uvals,featdif).ravel()

            pred = classifier.predict([prod])
            pairs[(t[0],t[1])] = pred

            if method == 'reg':
                if pred == 0: scores[t[1]] += 1
                else: scores[t[0]] += 1

        ranking = (-1*np.array(scores)).argsort()
        oneset.append(ranking)

        # checking individual accuracy here
        indiv_winner = ranking[0]

        # this is the person we are looknig at. 
        thisuser = [a for a in alt_indivotes[typ] if a['id'] == id]
        if thisuser: 
            gt_count+=1
            thisuser = thisuser[0]
            realwinner = thisuser['winner']
            realranking = thisuser['rankings']

            indiv_corrcount += kendalltau(realranking,ranking).correlation
            indiv_paircount += pair_accuracy(realranking,ranking)
            if match_winners([realwinner], [indiv_winner]):
                indiv_count += 1

            user_vals.append({
                'correlation':indiv_corrcount,
                'paircount': indiv_paircount,
                'regcount': indiv_count,
                'gt_count': gt_count
            })

    return oneset, user_vals