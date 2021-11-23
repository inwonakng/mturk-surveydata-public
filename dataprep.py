#%% start code
%reload_ext autoreload
%autoreload 2
from helpers import *

manyfile = '1_airplane_scenario/data/round4/parsed_all_ops_nonbinned.csv'
fewfile = '1_airplane_scenario/data/round4/parsed_fewer_nonbinned.csv'

newfile = '1_airplane_scenario/data/round5/parsed_data.csv'

plantfile = '2_plant_scenario/data/cumultive.csv'

# up here for consistent coloring thru features
cmap = matplotlib.cm.get_cmap('viridis')
colors = {
    'age':cmap(0.1),
    'gender':cmap(0.2),
    'health':cmap(0.3),
    'income':cmap(0.4),
    'education':cmap(0.5),
    'dependents':cmap(0.6),
    # 'surv_with':cmap(0.7),
    'surv_dif':cmap(0.7)  
}

# %% get features ready and clean data

df_all = pd.read_csv(manyfile)
df_few = pd.read_csv(fewfile)

df_cleaned_all, df_usertmp_all = clean_data1(df_all,True)
df_cleaned_few, df_usertmp_few = clean_data1(df_few,False)
df_user_all = clean_userdata(df_usertmp_all)
df_user_few = clean_userdata(df_usertmp_few)
#  creating pairwise data..
# return all_X,all_Y,X,Y,user_d,features

vall = make_pairwise(df_cleaned_all,df_user_all)
vfew = make_pairwise(df_cleaned_few,df_user_few)

# stuff is getting nasty so put it all in a dict
imp_vars = {
    'df_cleaned_all': df_cleaned_all,
    'df_user_all': df_user_all,
    'genX_all': vall['genX'], 
    'genY_all': vall['genY'],
    'X_all': vall['X'],
    'Y_all': vall['Y'],
    'user_d_all': vall['user_d'],
    'features_all': vall['features'],
    'df_cleaned_few': df_cleaned_few,
    'df_user_few': df_user_few,
    'genX_few': vfew['genX'],
    'genY_few': vfew['genY'],
    'X_few': vfew['X'],
    'Y_few': vfew['Y'],
    'user_d_few': vfew['user_d'],
    'features_few': vfew['features'],
}
pickle.dump(imp_vars,open('vars_inwon','wb'))

# %% prepareing plant data!
df = pd.read_csv(plantfile)
# doing some data cleaning...

d,u = clean_data2(df)
uu = clean_userdata(u)
vals = make_pairwise(d,uu,True)
# save pickle

imp_vars = {
    'df_cleaned': d,
    'df_user': u
}

imp_vars.update(vals)
pickle.dump(imp_vars,open('vars_inwon_plant','wb'))

# %% for our newest data

df_new = pd.read_csv(newfile)

rejects = pd.read_csv('1_airplane_scenario/data/round5/rejectlist.csv')
df_new = df_new.replace('great health','in great health')
df_cleaned_all, df_usertmp_all = clean_data1(df_new,True)

df_new = df_new.loc[~df_new.WorkerId.isin(list(rejects['WorkerId:']))]

df_user_all = clean_userdata(df_usertmp_all)
vall = make_pairwise(df_cleaned_all,df_user_all)
imp_vars = {
    'df_cleaned_all': df_cleaned_all,
    'df_user_all': df_user_all,
    'genX_all': vall['genX'], 
    'genY_all': vall['genY'],
    'X_all': vall['X'],
    'Y_all': vall['Y'],
    'user_d_all': vall['user_d'],
    'features_all': vall['features']
}
# pickle.dump(imp_vars,open('vars_inwon_new','wb'))
# %% round7
newfile = '1_airplane_scenario/data/round6.5/round7_parsed_withfeatures.csv'
newfile2 = '1_airplane_scenario/data/round8/parsed_round8.csv'
dd1 = pd.read_csv(newfile)
dd2 = pd.read_csv(newfile2)

# making sure the tripsets dont get messed up
dd2 = dd2.replace(
    ['tripset0','tripset1','tripset2','tripset3'],
    ['tripset4','tripset5','tripset6','tripset7'])
dd = pd.concat([dd1,dd2])
pairdata = pd.DataFrame()
tripledata = pd.DataFrame()

ground_truth = {
    'titanic': pd.DataFrame(),
    'tripset0': pd.DataFrame(),
    'tripset1': pd.DataFrame(),
    'tripset2': pd.DataFrame(),
    'tripset3': pd.DataFrame(),
    'tripset4': pd.DataFrame(),
    'tripset5': pd.DataFrame(),
    'tripset6': pd.DataFrame(),
    'tripset7': pd.DataFrame(),
}
#%%

# taking out pairwise data only
for i,rr in dd.iterrows():
    if rr['type'] == 'pair':
        pairdata = pairdata.append(rr)
    elif rr.type == 'triple':
        tripledata = tripledata.append(rr)
    else:
        ground_truth[rr.type] = ground_truth[rr.type].append(rr)

#%% doing the stuff now
df_cleaned,df_user = clean_data1(pairdata,True,2)
df_user_clean = clean_userdata(df_user)
vals = make_pairwise(df_cleaned,df_user_clean,type='pairs')

tt_df_cleaned,tt_df_user = clean_data1(tripledata,True,3)
tt_df_user_clean = clean_userdata(tt_df_user)
tvals = make_pairwise(tt_df_cleaned,tt_df_user_clean,type='triples')

gvals = {}

for typ,dt in ground_truth.items():
    if typ=='titanic': numop = 4
    else: numop = 3
    
    cleaned,users = clean_data1(dt,True,numop)
    u_cleaned = clean_userdata(users)
    gvals[typ] = {
        'rankdata': cleaned,
        'userdata': u_cleaned
    }
#%%
# merge pairwise and triple

xx = []
yy = []

for x1,x2 in zip(vals['X'],tvals['X']):
    xx.append(np.concatenate([x1,x2]))

for y1,y2 in zip(vals['Y'],tvals['Y']):
    yy.append(np.concatenate([y1,y2]))

mergevals = {
    'genX': np.concatenate([vals['genX'],tvals['genX']]),
    'genY': np.concatenate([vals['genY'],tvals['genY']]),
    'X': xx,
    'Y': yy,
    'user_d': vals['user_d'],
    # 'features': np.concatenate([vals['features'],tvals['features']]),
}

userdata = df_user_clean

#%% Merging agent features with alternative features

newgenX = []
newX = []
oldgenX = []
oldX = []

ufeats = ['USER.age', 'USER.education', 'USER.gender']
for x,(_,u) in zip(mergevals['X'],userdata.iterrows()):
    oneuser = []
    for xx in x:
        oldgenX.append(xx)
        uval,xval = [1],[1]
        uval.extend(u[ufeats].tolist())
        xval.extend(xx)

        prod = np.multiply.outer(uval,xval).ravel()
        oneuser.append(prod)
    newX.append(oneuser)
    oldX.append(xx)    

for x in newX: newgenX.extend(x)
for x in oldX: oldgenX.extend(x)

#%%
imp_vars = {
    'df_cleaned_all': df_cleaned,
    'df_user_all': userdata,
    'genX_old': oldgenX,
    'genX_all': newgenX, 
    'genY_all': mergevals['genY'],
    'X_all': newX,
    'Y_all': mergevals['Y'],
    'user_d_all': mergevals['user_d'],
    'features_all': vals['features'],
}

for typ,dat in gvals.items():
    if typ == 'titanic': pp = 'ti'
    else: pp = typ[0]+typ[-1]
    imp_vars['df_cleaned_all_'+pp] = dat['rankdata']
    imp_vars['df_user_all_'+pp] = dat['userdata']

pickle.dump(imp_vars,open('vars_inwon_round78_multifeat','wb'))

# %%
