#%% start code
%reload_ext autoreload
%autoreload 2
from helpers import *
# %%
imp_vars = load('vars_inwon_round78_multifeat')
tmp_userx = imp_vars['X_all']
uservals = imp_vars['user_d_all']
userinfo = imp_vars['df_user_all']
userx = []
usery = imp_vars['Y_all']

def get_altfeats(X):
    assert(len(X) == 32)
    return X[1:8]

# fix the multiplied features back to regular
for X in tmp_userx:
    xx = []
    for tx in X:
        xx.append(get_altfeats(tx))
    userx.append(xx)
# setup should be done here
#%%

bins = {}

bins['USER.age'] = userinfo['USER.age'].unique()
bins['USER.education'] = userinfo['USER.education'].unique()
bins['USER.gender'] = userinfo['USER.gender'].unique()

criteria = []
for k,v in bins.items():
    for vv in v:
        criteria.append((k,vv))
#%%

keys = lambda x: list(x.keys())
valss = lambda x: list(x.values())

binned = pd.DataFrame()
# binned.columns = ['id','age','gender','education','x','y']
ufeats = ['USER.age','USER.education','USER.gender']
ffeats = ['age','gender','education']

def sample(binned):
    binned_train = dict([((c[0][5:],c[1]),[[],[]]) for c in criteria])
    binned_test = pd.DataFrame()
    for _,row in binned.iterrows():
        xx,yy = row['x'],row['y']
        trax,tesx,tray,tesy = train_test_split(xx,yy)
        testrow = row.copy()
        testrow['x'] = tesx
        testrow['y'] = tesy
        binned_test = binned_test.append(testrow)
        for ff in ffeats:
            group = (ff,row[ff])
            binned_train[group][0].extend(trax)
            binned_train[group][1] = np.append(binned_train[group][1],tray)
    return binned_train,binned_test

# for k,v in bins.items():
#     for vv in v:
#         binned[(k,vv)] = {'x':[],'y':[]}

# binning data into dictionaries 
for u,xx,yy in zip(uservals,userx,usery):
    binned = binned.append(binned.from_dict({
        'id': [u.name],
        'age': [u['USER.age']],
        'gender':[u['USER.gender']],
        'education': [u['USER.education']],
        'x':[xx],
        'y':[yy],
    }))
        

binned_train,binned_test = sample(binned)

sometrees = {}
for group,vals in binned_train.items():
    xx = vals[0]
    yy = vals[1]
    # for ss in sep_x:

    clf = DecisionTreeClassifier().fit(xx,yy)
    sometrees[group] = (clf)

correct,totalcount = 0,0

for _,row in binned_test.iterrows():
    trees = []
    for ff in ffeats:
        gg = (ff,row[ff])
        trees.append(sometrees[gg])
    for xx,yy in zip(row['x'],row['y']):
        pred = 0
        for t in trees:
            pred += t.predict([xx])[0]
        if pred/len(trees) > 0.5:
            my_y = 1
        else: my_y = 0
        if yy == my_y: correct += 1
        totalcount += 1

print(correct/totalcount)

'''doing graph generation here'''
# %%
features = imp_vars['features_all']

records = pd.DataFrame()

for c,v in criteria:
    allx,ally = [],[]
    allx_low,ally_low = [],[]
    allx_high,ally_high = [],[]
    
    equal = [(uu,yy) for uu,vv,yy in zip(userx,uservals,usery) if vv[c] == v]
    for iii in equal: 
        allx.extend(iii[0])
        ally.extend(iii[1])
    allvals = {
        '=':(allx,ally),
    }

    outcomes = {}   
    for i,f in enumerate(features):
        ccount = 0
        # print(f)
        bb= []
        for op,(xx,yy) in allvals.items():
            thresh = 0
            highxwin = (len([x for x,y in zip(xx,yy) if x[i] >= thresh and y == 1])+ 
                len([x for x,y in zip(xx,yy) if x[i] < thresh and y == 0]))
            lowxwin = (len([x for x,y in zip(xx,yy) if x[i] >= thresh and y == 0])+ 
                len([x for x,y in zip(xx,yy) if x[i] < thresh and y == 1]))
            results = [
                (highxwin/len(xx), 'highxwin',thresh,op,highxwin),
                (lowxwin/len(xx), 'lowxwin',thresh,op,lowxwin)
            ]
            
        best = max(results)
        outcomes[f] = best[1]
        outcomes[f+'_acc'] = best[0]
        outcomes[f+'_thresh'] = best[2]
        outcomes[f+'_size'] = best[4]

    records = records.append(
        pd.Series(
            outcomes,name=c[5:] +' '+ best[3] +' ' + str(v) 
        )
    )

#%% got the data now

factors = dict()

# tis chooses only the best
# for f in features:
    # if 'lowxwin' in records[f].unique():
    #     lowmax = records.loc[records[f] == 'lowxwin'][f + '_acc'].idxmax()
    #     factors[f + '_preflow'] = [lowmax]
    #     factors[f + '_preflow'].extend(records.loc[lowmax][[f + '_acc',f+'_thresh',f+'_size']])
    # if 'highxwin' in records[f].unique():
    #     highmax = records.loc[records[f] == 'highxwin'][f + '_acc'].idxmax()
    #     factors[f + '_prefhigh'] = [highmax]
    #     factors[f + '_prefhigh'].extend(records.loc[highmax][[f + '_acc',f+'_thresh',f+'_size']])

for i,row in records.iterrows():
    vals = [(k,v) for k,v in dict(row).items()]
    # break
    factors[i] = [dict(vals[i:i+4]) for i in range(0,len(vals),4)]


for k in factors.copy():
    if k == 'age = 0.0': factors.pop(k)
    if k == 'education = 0.0': factors.pop(k)

#%%  making graph

user_age = {
    10.0: '10~19',
    20.0: '20~29',
    30.0: '30~39',
    40.0: '40~49',
    50.0: '50~59',
    60.0: '60~69',
}

user_gender = {
    0.0: 'male',
    1.0: 'female',
}

user_education = {
    1.0: 'highschool',
    2.0: 'college',
    3.0: 'graduate',
}

userf_key = {
    'age': user_age,
    'gender': user_gender,
    'education': user_education,
}

cmap = matplotlib.cm.get_cmap('coolwarm')

x,y,size,colors,labels,titles = [],[],[],[],[],[]

feats = [list(vv.keys())[0] for vv in list(factors.values())[0] ]

for k,v in factors.items():
    x.append([i for i,vv in enumerate(v)])
    y.append([v[jj][f+'_acc'] for jj,f in enumerate(feats)])
    size.append([(v[jj][f+'_size'] ) for jj,f in enumerate(feats)])
    directions = [10 if v[jj][f] =='lowxwin' else 230
                for jj,f in enumerate(feats)]
    # directions = [(i/len(v)) for i in range(-3,4)]
    print(directions)
    colors.append([cmap(d) for d in directions])
    labels.append([f for f in feats])

    typ = k[:k.index(' = ')]
    val = float(k[k.index(' = ')+3:])
    titles.append('User '+typ+': '+userf_key[typ][val])

z = [[ss**2 for ss in sss] for sss in size]

z = (normalize(z)**2)*20000

def makechart(subplot,x,y,z,colors,labels,size,title):
# Change color with c and alpha. I map the color to the X axis value.
    # plt.figure(figsize=(10,16))
    # plt.axes(yticks=[i for i in range(20)])
    pp = subplot.scatter(
        x, 
        y, 
        s=z, 
        color=colors, 
        cmap="coolwarm", 
        alpha=0.4, 
        edgecolors="grey", 
        linewidth=2,
        )
    samplesize = 'min sample size: {}\nmax simple size: {}'.format(min(size),max(size))
    # plt.xticks([i for i in range(-3,11)])
    # plt.legend(handles=[pp], title='title', bbox_to_anchor=(1.2, 1), loc='upper left')
    for i,l in enumerate(labels):
        subplot.annotate(l,xy=(x[i],y[i]),ha='center')

    ax = subplot.get_xlim()
    subplot.set_title(title,fontsize=10)

    # xpoint = textx
    # ypoint = texty
    subplot.text(0.5,-0.158,
    'min sample size: {}\nmax simple size: {}'.format(min(size),max(size)),
    ha='center',fontsize=10, transform=subplot.transAxes)

    # axes.title("User preference on alternative features")
    # subplot.tight_layout()
    # plt.show
    # plt.savefig()


fig,axes = plt.subplots(3,4,figsize=(10,12))
fig.subplots_adjust(bottom=0.2)
# plt.xlabel("User group")
# plt.ylabel("Accuracy")
row_x = [min(ee.get_xlim()) for ee in axes[0]]
# justxx = [ee.get_xlim() for ee in axes[0]]
# print(justxx)
avg_x = 0

maxidx = len(x)


# ax = axes[0][0]
# idx = 0
# makechart(ax,x[idx],y[idx],z[idx],colors[idx],labels[idx],size[idx],titles[idx])

for i,axss in enumerate(axes):
    # row_y = [max(ee.get_ylim()) for ee in axss]
    # avg_y = sum(row_y)/len(row_y)
    for j,ax in enumerate(axss):
        idx = i * 4 + j
        if(idx == maxidx):break
        makechart(ax,x[idx],y[idx],z[idx],colors[idx],labels[idx],size[idx],titles[idx])
# for aa,xx,yy,zz,col,ll,ss in zip(axes,x,y,z,colors,labels,size):
#     makechart(aa,xx,yy,zz,col,ll,ss)
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
plt.tight_layout()
# cbar = fig.colorbar(matplotlib.cm.ScalarMappable(cmap='coolwarm'),fraction=0.05,pad=0.07,ticks = [0,1],orientation='horizontal')
# cbar.ax.set_xticklabels(['Prefer lower value','Prefer higher value'])

# %%
