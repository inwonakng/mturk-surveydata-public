#%% start code
%reload_ext autoreload
%autoreload 2
from helpers import *
#%% setting variables n stuff
d_pref = '1_airplane_scenario/'

# d_pref = '2_plant_scenario'
if '1' in d_pref:
    imp_vars = pickle.load(open('vars_inwon_round78_multifeat','rb'))
    round_pref = 'round8/'
elif '2' in d_pref:
    imp_vars = pickle.load(open('vars_inwon_plant','rb'))
    round_pref = ''

is_all = None
prefix = None
genX = None
genY = None
X = None
Y = None
features = None
df_user = None
trainX = None
trainY = None
testX = None
testY = None
df_cleaned = None
user_d = None
transformed = False
justtrans = False

rfore,dtree,logreg,supvec,supvec_rbf,svclin,sgd = None,None,None,None,None,None,None

def setall():
    reload(setdata,{'is_all':True,'imp_vars':imp_vars})

def setfew():
    reload(setdata,{'is_all':False,'imp_vars':imp_vars})

def setaff():
    reload(setagent,
    {'prefix':'aff','imp_vars':imp_vars})

def setnei():
    reload(setagent,
    {'prefix': 'nei','imp_vars':imp_vars})

def setboth():
    reload(setagent,
    {'prefix': 'both','imp_vars':imp_vars})

def setdef():
    reload(setagent,
    {'prefix':'','imp_vars':imp_vars,'setdefault':True})

def nonlin(jt = False):
    global features,genX,transformed,trainX,testX,trainY,testY,justtrans
    features,genX = nonlinear(features,genX,jt)
    trainX,testX,trainY,testY = train_test_split(genX,genY,random_state = 0)
    transformed = True
    justtrans = jt
    loadmodels()

def loadmodels():
    global rfore,dtree,logreg,supvec,supvec_rbf
    rfore,dtree,logreg,supvec,supvec_rbf,svclin,sgd = createmodels(d_pref+round_pref,prefix,transformed,justtrans)

def reload(fn,argus):
    global is_all,prefix,genX,genY,X,Y,features,df_user,df_cleaned,user_d,trainX,testX,trainY,testY
    is_all,prefix,genX,genY,X,Y,features,df_user,df_cleaned,user_d = fn(**argus)
    trainX,testX,trainY,testY = train_test_split(genX,genY,random_state = 0)
    loadmodels()

if '1' in d_pref: setall()
else: setdef()

genX = imp_vars['genX_old']

'''defaults for comparison'''
defsvc = SVC(kernel='linear',random_state=0)
deffor = RandomForestClassifier(random_state=0)
deftree = DecisionTreeClassifier(random_state=0)
defreg = LogisticRegression(random_state=0)
deflinsvc = LinearSVC(random_state = 0)
defsdg = SGDClassifier(random_state=0)

defsvc_rbf = SVC(kernel='rbf',random_state=0)
#%% correlation check on generated data

pp = pickle.load(open('1_airplane_scenario/data/round8/gen_data/pairs_pickle','rb'))

for p in pp:
    

#%% checking duplicate values in user response
gc = [c for c in user_d[0].keys() if 'PREF.' in c]
count = 0
for u in user_d:
    uu = u[gc]
    # simple duplicate count
    dupc = len([c for c in uu.duplicated() if c == True])
    vals = {}
    if dupc > 1:
        for i,us in enumerate(uu):
            if us in vals: continue
            vals[us] = 0
            for uss in uu[i:]:
                if us == uss: vals[us] += 1
    if vals and max(vals.values())> 3:
        print(vals)
        count += 1
print(count/len(user_d))

#%% making a simple correlation matrix and heatmaps
all_users = []

# let's try adding all users
for u,vals in df_cleaned.groupby('TurkerID'):

    all_ranks = []
    all_mats = []
    # for now just looking at one user
    
    for t,r in vals.groupby('scenario'):
        # 3rd table always only compares survival, so pretty useless for this purpose
        if t == 2: continue
        r_data = r.drop(columns=['TurkerID','scenario'])
        kendal_corrmat = r_data.corr(method='kendall')
        kendal_corrmat.fillna(0,inplace=True)
        all_mats.append(kendal_corrmat)
        # newmat = normalize_mat(kendal_corrmat)

    all_users.append(all_mats)

agg = pd.DataFrame()

for i,a in enumerate(all_users):
    allmat = sum(a)/len(a)
    pp = allmat['score'].drop('score')
    pp.name = 'User '+str(i)
    agg = agg.append(pp)

heatmap(agg.T,'Kendall Tau Correlation',prefix,d_pref)

allmat = sum([sum(a) for a in all_users])/sum([len(a) for a in all_users])
dd = pd.DataFrame(allmat['score'].drop('score'))

heatmap(dd,'Kendall Tau Correlation',prefix,d_pref)

'''######################################'''
'''Code below here will take super long!!'''
'''######################################'''
# %% hyperparam tuning for decisionTrees...
mdepths = [i for i in range(1,20)]
msampleleafs = [i for i in range(1,200)]
mleafnodes = [i for i in range(1,200)]

params = {
    'max_depth': mdepths,
    'min_samples_leaf': msampleleafs,
    'max_leaf_nodes': mleafnodes,
    'random_state':[0]
}

it,bestsco = 0,0.6666666666666666
bestpp = None
while it < 5:
    sco,pp = random_tune(
        params,DecisionTreeClassifier,deftree,genX,genY,
        bestsco,
        loops = 100,
        iter = 10
        )
    if sco > bestsco:
        it = 0
        bestsco = sco
        bestpp = pp
    else:
        it += 1

notify()

print('==================================')
print(deftree.fit(trainX,trainY).score(testX,testY))
print(bestsco)
print(bestpp)

#%% hyperparam tuning for randomForest...

forests = []

n_est = [i*5 for i in range(1,10)]
ma_depth = [i for i in range(1,15)]
mi_ssplit = [i for i in range(0,81)]
mi_sleaf = [i for i in range(0,61)]
# ma_lnodes = [i for i in range(0,80)]

params = {
    'n_estimators': n_est,
    'max_depth': ma_depth,
    'min_samples_split': mi_ssplit,
    'min_samples_leaf': mi_sleaf,
    # 'max_leaf_nodes': ma_lnodes,
    'oob_score': [True],
    'bootstrap': [True],
    'max_features':['auto'],
    'random_state':[0]
}

bestsco = 0.6837606837606838
it = 0
bestpp = None
while it < 3:
    sco,pp = random_tune(
        params,RandomForestClassifier,deftree,genX,genY,
        bestsco,
        loops = 20,
        iter = 10
        )
    if sco > bestsco:
        it = 0
        bestsco = sco
        bestpp = pp
    else: it += 1

notify()

print('==================================')
print(deffor.fit(trainX,trainY).score(testX,testY))
print(bestsco)
print(bestpp)
#%% tuning hparams for logreg
p = {
    'tol':[20,15,10,1,0.1,0.01,0.001,0.0001,0.00001,0.000001],
    'C': [0.05,0.1,0.2,0.4,0.6,0.8,1,2,3],
    'max_iter': [500,800,1000,1200,1500,2000],
}

bestsco = 0
it = 0
bestpp = None
while it < 5:
    sco,pp = random_tune(
        p,LogisticRegression,deftree,genX,genY,
        bestsco,
        loops = 20,
        iter = 10
        )
    if sco > bestsco:
        it = 0
        bestsco = sco
        bestpp = pp
    else: it += 1
notify()

print(defreg.fit(trainX,trainY).score(testX,testY))
print('=================================')
print(bestsco)
print(bestpp)
#%%SVM tuning time !
p = {
    'tol':[2,1,0.1,0.01,0.001,0.0001,0.00001,0.000001],
    'C': [0.2,0.4,0.6,0.8,1,2,3],
    'probability': [True,False],
    'kernel': ['linear'],
    'random_state': [0]
}

bestsco = 0.6794871794871795
it = 0
bestpp = None
while it < 3:
    sco,pp = random_tune(
        p,SVC,deftree,genX,genY,
        bestsco,
        loops = 10,
        iter = 10
        )
    if sco > bestsco:
        it = 0
        bestsco = sco
        bestpp = pp
    else: it += 1

notify()

{'tol': 1e-05,
 'random_state': 0,
 'probability': True,
 'kernel': 'linear',
 'C': 0.2}

print(SVC(kernel='linear',random_state=0).fit(trainX,trainY).score(testX,testY))
print('=================================')
print(bestsco)
print(bestpp)

#%%SVM tuning time !
p = {
    'tol':[20,15,10,1,0.1,0.01,0.001,0.0001,0.00001,0.000001],
    'C': [0.2,0.4,0.6,0.8,1,2,3],
    # 'probability': [True,False],
    # 'kernel': ['rbf'],
    'random_state':[0],
    # 'degree': [1,2,3,4,5],
    'penalty':['l1','l2'],
    'loss':['hinge','squared_hinge'],
    'dual':[True,False],
    'fit_intercept':[True,False],
    'max_iter': [i*100 for i in range(10,30,5)],
    # 'shuffle': [True,False],
    # 'learning_rate': ['constant','optimal','invscaling','adaptive'],
    # 'early_stopping': [True,False],
    # 'eta0': [0.5, 0.02, 0.001, 0.0001, 0.00001],
}

bestsco = 0.66548463356974
it = 0
bestpp = None
while it < 3:
    sco,pp = random_tune(
        p,LinearSVC,deflinsvc,genX,genY,
        bestsco,
        loops = 10,
        iter = 5
        )
    if sco > bestsco:
        it = 0
        bestsco = sco
        bestpp = pp
    else: it += 1

notify()

print(deflinsvc.fit(trainX,trainY).score(testX,testY))

print(bestsco)
print(bestpp)

'''###################################'''
'''Code below here makes charts/tables'''
'''###################################'''
# %% looking at user given ground truth 
scored_feats = ['PREF.age', 'PREF.gender', 'PREF.health',
         'PREF.income', 'PREF.survdif']

if is_all: scored_feats.extend(['PREF.dependents'])

scores = dict(zip(scored_feats,
        preprocessing.normalize([df_user[scored_feats].mean()])[0]))
scores_sorted = sorted(scores.items(), key=lambda x: x[1])
clean_scores = [(trans_feats[s[0]],round(s[1],4)) for s in scores_sorted]

feat_imp = pd.Series([f[1] for f in clean_scores],
            index=[f[0]for f in clean_scores],name='ground truth')
feat_imp.plot(kind='barh',title='Agent assigned scores',
    color = [colors[c] for c in feat_imp.index])

plt.savefig(d_pref+'/visuals/feat_importances/'+prefix+'_features/ground_truth.png',bbox_inches = "tight")
plt.show()

heatmap(pd.DataFrame().append(feat_imp).T,'Average ground truth',prefix,d_pref)
plt.show()

agg = df_user[scored_feats]
agg.columns = [trans_feats[f] for f in scored_feats]

heatmap(agg.T,'Agent ground truth',prefix,d_pref)

#%% Using tuned hparams make new classifiers
rfore,dtree,logreg,supvec,supvec_rbf,svclin,sgd = createmodels(d_pref+round_pref,prefix)
trainX,testX,trainY,testY = train_test_split(genX,genY,random_state = 0)

#%% testing prediction accuracy of the new ones

models = [
    deffor,
    rfore,
    deftree,
    dtree,
    defreg,
    logreg,
    deflinsvc,
    svclin,
    defsdg,
    sgd,
    # defsvc,
    # supvec,
    # defsvc_rbf,
    # supvec_rbf,
]

sum_scores = [0 for m in models]
ntest = 1

# for i in tqdm(range(ntest)):
    # making random training sets here
    # trainX,testX,trainY,testY = train_test_split(genX,genY)

for i,m in tqdm(enumerate(models)):
    m.fit(trainX,trainY)
    sum_scores[i] += m.score(testX,testY)

scores = [s/ntest for s in sum_scores]

if is_all: print('All features')
else: print('Fewer features')

cols = [
    'Random Forest (default)',
    'Random Forest (tuned)',
    'Decision Tree (default)',
    'Decision Tree (tuned)',
    'Logistic Regression (default)',
    'Logistic Regression (tuned)',
    'Linear SVM (default)',
    'Linear SVM (tuned)',
    'SDG Classifier(default)',
    'SDG Classifier(tuned)',
    # 'SVM linear (default) ',
    # 'SVM linear (tuned) ',
    # 'SVM rbf (default) ',
    # 'SVM rbf (tuned) ',
]

print(smalltable(['Algorithm','Score'],[cols,scores]))

#%% Using tuned models to get feature importances again

ff = features.copy()
ff.remove('surv_with')

deffor.fit(genX,genY)
forest = dict(zip(ff, normalize([deffor.feature_importances_])[0]))
sorted_forest = sorted(forest.items(), key=lambda x: x[1],reverse=True)
clean_forest = [(s[0],round(s[1],4)) for s in sorted_forest]

deftree.fit(genX,genY)
detree = dict(zip(ff, normalize([deftree.feature_importances_])[0]))
sorted_tree = sorted(detree.items(), key=lambda x: x[1],reverse=True)
clean_tree = [(s[0],round(s[1],4)) for s in sorted_tree]

logreg.fit(genX,genY)
regre = dict(zip(ff, normalize([logreg.coef_[0]])[0]))
sorted_reg = sorted(regre.items(), key=lambda x: x[1],reverse=True)
clean_reg = [(s[0],round(s[1],4)) for s in sorted_reg]

svclin.fit(genX,genY)
svals = dict(zip(ff, normalize([svclin.coef_[0]])[0]))
sorted_svals = sorted(svals.items(), key=lambda x: x[1],reverse=True)
clean_svals = [(s[0],round(s[1],4)) for s in sorted_svals]

#%% Random Forest general diagram
# making markdown table so i can copy paste

if is_all: print('All features')
else: print('Fewer features')
print('random forest')

col_names = ['Feature','Score']
col_vals = [
    [d[0] for d in clean_forest],
    [str(d[1]).ljust(6,'0') for d in clean_forest]
]

feat_imp = pd.Series([f[1] for f in clean_forest],
            index=[f[0]for f in clean_forest], name='coeff')

cc = colors
tfeat = ''
if transformed: 
    cc = manycolors
    tfeat = '_tform'
    if justtrans: tfeat = '_onlytform'

feat_imp.plot(kind='barh',title='Random Forest',
    color = [cc[c] for c in feat_imp.index])
plt.gca().invert_yaxis()

plt.savefig(d_pref+'/visuals/feat_importances/'+prefix+tfeat+'_features/r_forest.png',bbox_inches = "tight")

plt.show()

heatmap(pd.DataFrame().append(feat_imp).T,'Random Forest',prefix,d_pref)
plt.show()

print(smalltable(col_names,col_vals))


#%% Decision tree general diagram
print('d tree')

col_names = ['Feature','Score']
col_vals = [
    [d[0] for d in clean_tree],
    [str(d[1]).ljust(6,'0') for d in clean_tree]
]

feat_imp = pd.Series([f[1] for f in clean_tree],
            index=[f[0]for f in clean_tree], name='coeff')

cc = colors
tfeat = ''
if transformed: 
    cc = manycolors
    tfeat = '_tform'
    if justtrans: tfeat = '_onlytform'

feat_imp.plot(kind='barh',title='Decision Tree',
    color = [cc[c] for c in feat_imp.index])
plt.gca().invert_yaxis()

plt.savefig(d_pref+'/visuals/feat_importances/'+prefix+tfeat+'_features/d_tree.png',bbox_inches = "tight")
plt.show()

heatmap(pd.DataFrame().append(feat_imp).T,'Decision Tree',prefix,d_pref)
plt.show()

print(smalltable(col_names,col_vals))

#%% Logistic regression general diagram
print('logreg')

col_names = ['Feature','Score']
col_vals = [
    [d[0] for d in clean_reg],
    [str(d[1]).ljust(6,'0') for d in clean_reg]
]

feat_imp = pd.Series([f[1] for f in clean_reg],
            index=[f[0]for f in clean_reg], name='coeff')

cc = colors
tfeat = ''
if transformed: 
    cc = manycolors
    tfeat = '_tform'
    if justtrans: tfeat = '_onlytform'

feat_imp.plot(kind='barh',title='Logistic Regression',
    color = [cc[c] for c in feat_imp.index])
plt.gca().invert_yaxis()

plt.savefig(d_pref+'/visuals/feat_importances/'+prefix+tfeat+'_features/logreg.png',bbox_inches = "tight")
plt.show()

heatmap(pd.DataFrame().append(feat_imp).T,'Logistic Regression',prefix,d_pref)
plt.show()

print(smalltable(col_names,col_vals))


#%% SVC general diagram
print('svm linear')

col_names = ['Feature','Score']
col_vals = [
    [d[0] for d in clean_svals],
    [str(d[1]).ljust(6,'0') for d in clean_svals]
]

feat_imp = pd.Series([f[1] for f in clean_svals],
            index=[f[0]for f in clean_svals], name='coeff')

cc = colors
tfeat = ''
if transformed: 
    cc = manycolors
    tfeat = '_tform'
    if justtrans: tfeat = '_onlytform'

feat_imp.plot(kind='barh',title='Support Vector Classification',
    color = [cc[c] for c in feat_imp.index])
plt.gca().invert_yaxis()

plt.savefig(d_pref+'/visuals/feat_importances/'+prefix+tfeat+'_features/svc.png',bbox_inches = "tight")
plt.show()

heatmap(pd.DataFrame().append(feat_imp).T,'SVC',prefix,d_pref)
plt.show()

print(smalltable(col_names,col_vals))

#%% making a quick dtree diagram

print('default score:',DecisionTreeClassifier().fit(trainX,trainY).score(testX,testY))

if prefix=='all':
    p0 = {'max_depth': 5, 'max_leaf_nodes': 16, 'min_samples_leaf': 7, 'random_state': 0}
    p1 = {'max_depth': 2, 'max_leaf_nodes': 3, 'min_samples_leaf': 1, 'random_state': 0}
    trees = [p0,p1]
    if transformed:
        p0 = {'random_state': 0, 'min_samples_leaf': 39, 'max_leaf_nodes': 164, 'max_depth': 3}
        trees = [p0]
if prefix=='aff':
    p0 = {'random_state': 0, 'min_samples_leaf': 13, 'max_leaf_nodes': 161, 'max_depth': 4}
    trees = [p0]
if prefix=='notme':
    p0 = {'random_state': 0, 'oob_score': True, 'n_estimators': 5, 'min_samples_split': 6, 'min_samples_leaf': 5, 'max_features': 'auto', 'max_depth': 5, 'bootstrap': True}
    p1 = {'random_state': 0, 'oob_score': True, 'n_estimators': 10, 'min_samples_split': 44, 'min_samples_leaf': 6, 'max_features': 'auto', 'max_depth': 3, 'bootstrap': True}
    trees = [p0,p1]
if prefix=='yesme':
    p0 = {'random_state': 0, 'min_samples_leaf': 24, 'max_leaf_nodes': 54, 'max_depth': 4}
    trees = [p0]

print(smalltable(
        ['depth','score'],
        [[p['max_depth'] for p in trees], 
        [DecisionTreeClassifier(**p).fit(trainX,trainY).score(testX,testY)
         for p in trees]]
        ))

for p in trees:
    dt = DecisionTreeClassifier(**p).fit(trainX,trainY)

    dt.fit(genX,genY)
    fig = plt.figure(figsize=(40,12))
    tree.plot_tree(dt,feature_names=features,filled=True,fontsize=16,
        class_names=['No','Yes'])
    fig.savefig('2_plant_scenario/visuals/'+prefix+'_dtreemap_d5.png')

#%% Running Regression

max_scaler = preprocessing.MaxAbsScaler()

users_f = []
user_ids = []
logreg_beta = []
tree_beta = []
rand_beta = []
svc_beta = []
linsvc_beta = []
sgd_beta = []

# need to reconstruct forest and tree when doing for each user, because
# not enough data points when doing per agent

# i'm setting these to default right now, but could be changed to something else 
rfore_param = rfore.get_params()
rfore_param.pop('min_samples_split')
rfore_param.pop('min_samples_leaf')

dtree_param = dtree.get_params()
dtree_param.pop('min_samples_leaf')

newforest = RandomForestClassifier(**rfore_param)
newtree = DecisionTreeClassifier(**dtree_param)

uf = ['1','user.age','user.gender','user.education']
ff = ['1']
ff.extend(features)

combos = []
for u in uf:
    for f in ff:
        combos.append((u,f))
for x,y,(_,u) in tqdm(zip(X,Y,df_user.iterrows())):
    zc = np.count_nonzero(y)
    if zc == 0 or zc == len(y): continue
    y = np.array(y).astype(np.int8)
    # if np.count_nonzero(u) < 2: continue
    X_maxabs = max_scaler.fit_transform(x)
    
    newforest.fit(X_maxabs,y)
    rand_coeffs = dict(zip(combos, newforest.feature_importances_))

    newtree.fit(X_maxabs,y)
    tree_coeffs = dict(zip(combos, newtree.feature_importances_))

    logreg.fit(X_maxabs, y)
    log_coeffs = dict(zip(combos, logreg.coef_[0]))

    # supvec.fit(X_maxabs,y)
    # svc_coeffs = dict(zip(features, supvec.coef_[0]))

    svclin.fit(X_maxabs, y)
    linsvc_coeffs = dict(zip(combos, svclin.coef_[0]))

    # sgd.fit(X_maxabs,y)
    # sgd_coeffs = dict(zip(features, sgd.coef_[0]))

    # userval = u[user_features].values
    # users_f.append(userval)
    user_ids.append(u.name)
    rand_beta.append(rand_coeffs)
    tree_beta.append(tree_coeffs)
    logreg_beta.append(log_coeffs)
    # svc_beta.append(svc_coeffs)
    linsvc_beta.append(linsvc_coeffs)
    # sgd_beta.append(sgd_coeffs)
# %% saving user weights

df_userpref_logreg = pd.DataFrame()
df_userpref_dtree = pd.DataFrame()
df_userpref_rforest = pd.DataFrame()
df_userpref_svc  = pd.DataFrame()
df_userpref_linsvc = pd.DataFrame()
# df_userpref_sgd = pd.DataFrame()

if is_all: prefix = 'all'
else: prefix = 'few'

mult = '_mult'

for f,u_id in zip(logreg_beta,user_ids):
    vs = sorted(f.items(), key=lambda x: x[1],reverse=True)
    # range is the n value for top n ranks
    f['WorkerId'] = u_id
    df_userpref_logreg = df_userpref_logreg.append(
        pd.DataFrame(f,index=[0]))
if '1' in d_pref:
    filename = d_pref+'data/'+round_pref+'user_prefs/'+prefix+'_logreg'+mult+'.csv'
if '2' in d_pref:
    filename = d_pref+'/data/user_prefs/'+prefix+'_logreg.csv'
df_userpref_logreg.to_csv(filename,index = False)

for f,u_id in zip(tree_beta,user_ids):
    vs = sorted(f.items(), key=lambda x: x[1],reverse=True)
    # range is the n value for top n ranks
    f['WorkerId'] = u_id
    df_userpref_dtree = df_userpref_dtree.append(
        pd.DataFrame(f,index=[0]))

if '1' in d_pref:
    filename = d_pref+'data/'+round_pref+'user_prefs/'+prefix+'_dtree'+mult+'.csv'
if '2' in d_pref:
    filename = d_pref+'/data/user_prefs/'+prefix+'_dtree.csv'
df_userpref_dtree.to_csv(filename,index = False)

for f,u_id in zip(rand_beta,user_ids):
    vs = sorted(f.items(), key=lambda x: x[1],reverse=True)
    # range is the n value for top n ranks
    f['WorkerId'] = u_id
    df_userpref_rforest = df_userpref_rforest.append(
        pd.DataFrame(f,index=[0]))

if '1' in d_pref:
    filename = d_pref+'data/'+round_pref+'user_prefs/'+prefix+'_rforest'+mult+'.csv'
if '2' in d_pref:
    filename = d_pref+'/data/user_prefs/'+prefix+'_rforest.csv'
df_userpref_rforest.to_csv(filename,index = False)

# for f,u_id in zip(svc_beta,user_ids):
#     vs = sorted(f.items(), key=lambda x: x[1],reverse=True)
#     # range is the n value for top n ranks
#     f['WorkerId'] = u_id
#     df_userpref_svc = df_userpref_svc.append(
#         pd.DataFrame(f,index=[0]))

# if '1' in d_pref:
#     filename = d_pref+'data/'+round_pref+'user_prefs/'+prefix+'_svc.csv'
# if '2' in d_pref:
#     filename = d_pref+'/data/user_prefs/'+prefix+'_svc.csv'
# df_userpref_svc.to_csv(filename,index = False)

for f,u_id in zip(linsvc_beta,user_ids):
    vs = sorted(f.items(), key=lambda x: x[1],reverse=True)
    # range is the n value for top n ranks
    f['WorkerId'] = u_id
    df_userpref_linsvc = df_userpref_linsvc.append(
        pd.DataFrame(f,index=[0]))

if '1' in d_pref:
    filename = d_pref+'data/'+round_pref+'user_prefs/'+prefix+'_linsvc'+mult+'.csv'
if '2' in d_pref:
    filename = d_pref+'/data/user_prefs/'+prefix+'_linsvc.csv'
df_userpref_linsvc.to_csv(filename,index = False)

# for f,u_id in zip(sgd_beta,user_ids):
#     vs = sorted(f.items(), key=lambda x: x[1],reverse=True)
#     # range is the n value for top n ranks
#     f['WorkerId'] = u_id
#     df_userpref_sgd = df_userpref_sgd.append(
#         pd.DataFrame(f,index=[0]))

# if '1' in d_pref:
#     filename = d_pref+'data/'+round_pref+'user_prefs/'+prefix+'_sgd.csv'
# if '2' in d_pref:
#     filename = d_pref+'/data/user_prefs/'+prefix+'_sgd.csv'
# df_userpref_sgd.to_csv(filename,index = False)

# %% make heatmaps

heatmap(df_userpref_rforest.T,'Random forest for each agent',prefix,d_pref)
heatmap(df_userpref_dtree.T,'Decision tree for each agent',prefix,d_pref)
heatmap(df_userpref_logreg.T,'Logistic regression for each agent',prefix,d_pref)
heatmap(df_userpref_svc.T,'SVC for each agent',prefix,d_pref)

# %% guessing pairwise
sorted_forest
sorted_tree
sorted_reg
sorted_svals
clean_scores

thresh = {
    'age':0.3,
    'gender':0.3,
    'health':0.3,
    'income':0.3,
    'education':0.3,
    'dependents':0.3,
    'surv_dif':0.1,
    }

bestsco = 0.5610756608933455

localbest = 0
bestthresh = None
it = 0

while it < 10:
    for i in tqdm(range(1000)):
        tt = dict(zip(features,[uniform(-1,1) for i in features]))
        ss = makeguess(genX,genY,sorted_svals,features,tt)
        if ss > localbest:
            localbest = ss
            bestthresh = tt.copy()
    if bestsco < localbest:
        it = 0
        bestsco = localbest
    else: it += 1

print()
print(bestsco)
print(bestthresh)

#%%
# forest
# 0.5610756608933455
{'age': 0.5358786358914871,
 'gender': 0.3592956007100363,
 'health': 0.44845797479983207,
 'income': 0.12858840231926139,
 'education': 0.9862823703890418,
 'dependents': 0.8016127178392249,
 'surv_dif': 0.11215236071814272

#  tree
# 0.5615314494074749
{'age': 0.9866112086780956, 'gender': 0.8455389287408195, 'health': 0.9628719256104892, 'income': 0.452979656219479, 'education': 0.7141082837626578, 'dependents': 0.6207057372848814, 'surv_dif': 0.13226419420790925}

# reg
# 0.5610756608933455
{'age': 0.03821461084802835, 'gender': 0.8479392180959657, 'health': 0.5444927804497601, 'income': 0.8280825490438306, 'education': 0.06051488786382109, 'dependents': 0.30872682415670516, 'surv_dif': 0.10301068086356935}

# svc
# 0.5615314494074749
{'age': 0.7802079208234618, 'gender': 0.740360712273435, 'health': 0.8439571048674326, 'income': 0.14976102942945957, 'education': 0.7551718647639323, 'dependents': 0.6617835694008563, 'surv_dif': 0.14591726460957388}