#%% start code
%reload_ext autoreload
%autoreload 2
from helpers import *

#%% loading pickles
imp_vars = pickle.load(open('vars_inwon_round78_multifeat','rb'))
# imp_vars = pickle.load(open('vars_inwon_round7','rb'))

userx = imp_vars['X_all']
usery = imp_vars['Y_all']
userd = imp_vars['user_d_all']

datas = {
    'titanic': imp_vars['df_cleaned_all_ti'],
    'trip0': imp_vars['df_cleaned_all_t0'],
    'trip1': imp_vars['df_cleaned_all_t1'],
    'trip2': imp_vars['df_cleaned_all_t2'],
    'trip3': imp_vars['df_cleaned_all_t3'],
    'trip4': imp_vars['df_cleaned_all_t4'],
    'trip5': imp_vars['df_cleaned_all_t5'],
    'trip6': imp_vars['df_cleaned_all_t6'],
    'trip7': imp_vars['df_cleaned_all_t7'],
}

userdatas = imp_vars['df_user_all']

# userdatas = {
#     'titanic': imp_vars['df_user_all'],
#     'trip0': imp_vars['df_user_all_t0'],
#     'trip1': imp_vars['df_user_all_t1'],
#     'trip2': imp_vars['df_user_all_t2'],
#     'trip3': imp_vars['df_user_all_t3'],
#     'trip4': imp_vars['df_user_all_t4'],
#     'trip5': imp_vars['df_user_all_t5'],
#     'trip6': imp_vars['df_user_all_t6'],
#     'trip7': imp_vars['df_user_all_t7'],
# }


#%% checking ground truth

p_winners = {}
p_votes = {}
b_winners = {}
b_votes = {}
m_winners = {}
m_votes = {}

for typ,dd in datas.items():
    votes = []
    for i,oneuser in dd.groupby('TurkerID'):
        for _,group in oneuser.groupby('scenario'):
            votes.append((-1*np.array(group['score'])).argsort())

    p_winners[typ],p_votes[typ] = plurality_winner(np.array(votes))
    b_winners[typ],b_votes[typ] = Borda_winner(np.array(votes))
    m_winners[typ],m_votes[typ] = maximin_winner(np.array(votes))

print(p_winners,p_votes)
print(b_winners,b_votes)
print(m_winners,m_votes)

ground_truth = {
    'plurality': {'winners': p_winners, 'votes': p_votes},
    'borda': {'winners': b_winners, 'votes': b_votes},
    'maximin': {'winners': m_winners, 'votes': m_votes}
}

#models:
rfore,dtree,logreg,supvec,supvec_rbf,svclin,sgd = createmodels('1_airplane_scenario/round8','all')

#%% lets try a voting rule

feats = imp_vars['features_all']
ufeats = ['USER.age','USER.education','USER.gender']


alternatives = {
                'titanic': [datas['titanic'].iloc[i][feats] for i in range(4)],
                'trip0': [datas['trip0'].iloc[i][feats] for i in range(3)],
                'trip1': [datas['trip1'].iloc[i][feats] for i in range(3)],
                'trip2': [datas['trip2'].iloc[i][feats] for i in range(3)],
                'trip3': [datas['trip3'].iloc[i][feats] for i in range(3)],
                'trip4': [datas['trip4'].iloc[i][feats] for i in range(3)],
                'trip5': [datas['trip5'].iloc[i][feats] for i in range(3)],
                'trip6': [datas['trip6'].iloc[i][feats] for i in range(3)],
                'trip7': [datas['trip7'].iloc[i][feats] for i in range(3)],
                }

alt_indivotes = {}


# saving individual votes here
for name,alts in alternatives.items():
    if name == 'titanic': numops = 4
    else: numops = 3
    alt_indivotes[name] = []
    
    for id,userrow in datas[name].groupby('TurkerID'):
        # some dude answered the titanic twice, so gotta separate once more
        for _,rows in userrow.groupby('scenario'):
            ranks = (-1*np.array(rows['score'])).argsort()
            alt_indivotes[name].append(
                {'id':id,'rankings':ranks,
                'winner':ranks[0]})
#%%

models = {
    'dt': dtree,
    # 'lr': logreg,
    # 'sv': svclin,
    'rf': rfore,
}


all_votes = {}
iter = 10
# samples = [40,80,120,len(userx)]
samples = [20]

votefuncs = {
    # 'plurality': plurality_winner,
    'borda': Borda_winner,
    'maximin': maximin_winner
}

# Change voting type here!!!
# votetype = 'plurality'
# votetype = 'borda'
# votetype = 'maximin'

# _,testx,__,testy = train_test_split(imp_vars['genX_all'],imp_vars['genY_all'],random_state=0)

if 'saved_results' in locals().keys(): allresults = saved_results
else: allresults = {}
for typ,ops in alternatives.items():
    results = []
    # for s_size in [50,100,150]:
    for s_size in samples:
        for name,classifier in models.items():           
            cor_count = 0
            indiv_count = 0
            indiv_corrcount = 0
            indiv_paircount = 0
            gt_count = 0
            train_acc = 0
            itersets = []
            for itera in tqdm(range(iter)):
                ss = np.random.randint(len(userx),size = s_size)
                trainx = [u for n in ss for u in userx[n]]
                trainy = [u for n in ss for u in usery[n]]
                sample = pd.DataFrame()
                for i in ss:
                    sample = sample.append(userdatas.iloc[i])
                try: classifier.fit(trainx,trainy)
                except: 
                    # sometimes it won't like it because not enough y varitey
                    itera -= 1
                    continue
                train_acc += classifier.score(trainx,trainy)
                
                if not name in ['rf', 'dt']: 
                    oneset,user_vals = beta_votes(sample,classifier,alt_indivotes,ops,typ)
                else:
                    oneset,user_vals = tree_votes(sample,classifier,alt_indivotes,ops,typ)

                idniv_count = sum([u['regcount'] for u in user_vals])
                indiv_corrcount = sum([u['correlation'] for u in user_vals])
                idniv_paircount = sum([u['paircount'] for u in user_vals])
                gt_count = sum([u['gt_count'] for u in user_vals])
                itersets.append(oneset)
                # change this to change voting rule!!
                # im taking this outside the loop so we get all the data first
                # win,votes = votefuncs[votetype](np.array(oneset))
                # g_winner = ground_truth[votetype]['winners'][typ]
                # g_votes = ground_truth[votetype]['votes'][typ]

                # grank = (-1*np.array(g_votes)).argsort()
                # arank = (-1*np.array(votes)).argsort()

                # aggr_corrcount += kendalltau(grank,arank).correlation

                # if match_winners(g_winner,win): cor_count += 1
                
                # vote_log.append((win,votes))                
            recorded = {
                'classifier':name,
                'samplesize':s_size,
                'indiv_correct': float(indiv_count)/(gt_count),
                'indiv_corr_avg': float(indiv_corrcount)/(gt_count),
                'indiv_pair_match': float(indiv_paircount)/(gt_count),
                'training_accuracy': train_acc/iter,
                'cls_ranks': itersets,
                }
            print('\n data:',typ,'\n type:',name,'accuracy:',float(cor_count)/iter,'\t',s_size,'samples')
            
            results.append(recorded)
            # all_votes[(name,s_size)] = vote_log
    allresults[typ] = results

dump(allresults,'all_results_voting')
#%% now we got the data, lets run voting rules

allresults = load('all_results_voting')

for votetype in votefuncs.keys():
    names = {
        'lr':'Logistic Regression',
        'sv':'Linear SVM',
        'dt': 'Decision Tree',
        'rf': 'Random Forest'
    }

    scores = pd.DataFrame()

    for typ,mm in allresults.items():
        for results in mm:
            aggr_corrcount = 0
            vote_log = []
            for sset in results['cls_ranks']:
                win,votes = votefuncs[votetype](np.array(sset))
                g_winner = ground_truth[votetype]['winners'][typ]
                g_votes = ground_truth[votetype]['votes'][typ]

                grank = (-1*np.array(g_votes)).argsort()
                arank = (-1*np.array(votes)).argsort()

                aggr_corrcount += kendalltau(grank,arank).correlation

                if match_winners(g_winner,win): cor_count += 1
                
                vote_log.append((win,votes)) 
            results.update({
                'accuracy':float(cor_count)/iter,
                'correct count':cor_count,
                'votes':vote_log,
                'aggr_votes_correlation': float(aggr_corrcount)/(iter),
            })

            votes = results['votes']
            wins = [v[0][0] for v in votes]
            predicted = np.argmax(np.bincount(np.array(wins)))
            
            votes = [np.array(v[1]) for v in results['votes']]
            sum(votes)

            inp = {'Dataset': [type],
                    'classifier': [names[results['classifier']]],
                    'samplesize':[results['samplesize']],
                    'Aggregated accuracy':[results['accuracy']],
                    'Correct aggregated count':[results['correct count']],
                    'Individual winner accuracy':[results['indiv_correct']],
                    'Individual ranking correlation': [results['indiv_corr_avg']],
                    'Individual ranking pairwise match': [results['indiv_pair_match']],
                    'Aggregated ranking correlation': [results['aggr_votes_correlation']],
                    'Training accuracy':[results['training_accuracy']],
                    'model predicted':[predicted],
                    'ground truth': [p_winners[type]]
                    }

            scores = scores.append(scores.from_dict(inp))

    #run this block to save the results!
    # 
    scores.to_csv(votetype+'_results.csv',index = False)
    dump(allresults,votetype+'_results')

# scores.to_pickle('1_airplane_scenario/data/round7/voting/plurality/plurality_results')

#%% read in csvs and compare results now
rules = [
    'plurality',
    'borda',
    'maximin',
]
pr = '1_airplane_scenario/data/voting'

results = {}

for r in rules:
    inc = pd.read_csv(pr + '/multifeatures/'+r+'_results.csv')
    results[r] = inc.dropna(thresh=2)

# %% graph for titanic in all rules:

dpref = {
    'titanic': 'Titanic',
    'trip0': 'Triple #0',
    'trip1': 'Triple #1',
    'trip2': 'Triple #2',
    'trip3': 'Triple #3',
    'trip4': 'Triple #4',
    'trip5': 'Triple #5',
    'trip6': 'Triple #6',
    'trip7': 'Triple #7',
}

mode = 'Linear SVM'
# mode = 'Logistic Regression'
dset = 'titanic'

# criteria = 'Aggregated accuracy'
criterias = ['Individual winner accuracy'
,'Individual ranking correlation'
,'Individual ranking pairwise match']


labels = [  '40',
            '80',
            '120',
            '170']
fig,ax = plt.subplots()
width = 0.2
x = np.arange(len(labels))

vals = {}

res = results['borda']
for c in criterias:
    ll = res[(res.Dataset == dset) & (res.classifier ==mode)][c] * 100
    vals[c] = list(ll)

# for (v,res),c in zip(results.items(),criterias):
#     ll = res[(res.Dataset == dset) & (res.classifier ==mode)][c] * 100
#     vals[c] = list(ll)

rects1 = ax.bar(x - width, vals[criterias[0]], 
                width, label='Winner accuracy')
rects1 = ax.bar(x, vals[criterias[1]], 
                width, label='Ranking correlation')
rects1 = ax.bar(x + width, vals[criterias[2]], 
                width, label='Ranking pairwise match')

ax.set_ylabel(criteria + ' scores')
ax.set_xlabel('Sample size')

ax.set_title(dpref[dset] +' dataset' )
# ax.set_title(dpref[dset] +' dataset with '+mode )

ax.set_yticks(np.arange(10)*10)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc="upper center", bbox_to_anchor=(1.15, 1))
# plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol= 2)

#%%
for r,re in results.items():
    for i,dd in re.groupby('Dataset'):
        for criteria in ['Individual winner accuracy','Individual ranking correlation','Individual ranking pairwise match']:
            fig,ax = plt.subplots()
            width = 0.20
            labels = [  'sample size = 40',
                        'sample size = 80',
                        'sample size = 120']
            x = np.arange(len(labels))
            vals = {}
            for m,nn in dd.groupby('classifier'):
                vals[m] = list(nn[criteria])
            rects1 = ax.bar(x - width/2, vals['Linear SVM'], 
                    width, label='Linear SVM')
            rects2 = ax.bar(x + width/2, vals['Logistic Regression'], 
                    width, label='Logistic Regression')

            ax.set_ylabel('Scores')
            ax.set_title(criteria+'\n Dataset: '+i)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()

            pp = pr+'/multifeatures/visuals/'+ i.lower()+'_' + criteria.lower().replace(' ','_')+'.png'

            fig.savefig(pp)