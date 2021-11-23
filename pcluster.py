#%% start
import sys
%reload_ext autoreload
%autoreload 2
from helpers import *

#%%
train = pd.read_csv('titanic_data/train.csv')
test = pd.read_csv('titanic_data/test.csv')

for i,t in test.iterrows():
    train = train.append(t)
#%%
sdic = {'male':0,'female':1}

cc = pd.DataFrame()
for i,t in train.iterrows():
    # vals = t[['Parch','Age','Sex',]]
    vals = t[['Parch','Age','Sex','Pclass','PassengerId']]
    # if str(vals['Parch']) == 'nan': continue
    if str(vals['Age']) == 'nan': continue
    if str(vals['Sex']) == 'nan': continue
    vals['Sex'] = sdic[vals['Sex']]
    cc = cc.append(vals)

cc.fillna(0)
cc = cc.reset_index()
cc = cc.drop(columns=['index'])
cnew = cc.drop(columns=['PassengerId'])
#%%

inertias = []
for i in range(1,10):
    k= KMeans(n_clusters=i).fit(cnew)
    inertias.append(k.inertia_)

plt.plot(range(1,10), inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()
#%%
kk = KMeans(n_clusters=3).fit(cnew)
# closest, _ = pairwise_distances_argmin_min(kk.cluster_centers_, cnew)

ll = pd.concat((cnew,pd.DataFrame(kk.labels_)),axis=1)
ll = ll.rename({0:'labels'},axis=1)
g = sns.pairplot(ll,hue='labels')

# # g.get_figure().savefig('/home/inwon/Documents/mturk/visuals/titanic_dependents.png',bbox_inches = "tight")

# print(ll.columns)
# print(kk.cluster_centers_)

