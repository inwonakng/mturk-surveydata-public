#%%
import numpy as np
from sklearn.cluster import SpectralClustering as sc
from sklearn.mixture import GaussianMixture
import pandas as pd

# %%
scores_pairs = pd.read_pickle('../airplane_final_data/scores_pairs')
agent_features = pd.read_pickle('../airplane_final_data/agent_features')
agent_imp = pd.read_pickle('../airplane_final_data/agent_imp')

#%%

# Similarity matrix method
# https://reader.elsevier.com/reader/sd/pii/S0167865519300108?token=081FC1B40BEBC1DEC8A2F94A8420410C7DEF0BE3C4441DCFB9C691DD40845DB012843E68B8FDFC8721BBFBCCBB6C9059
sample = scores_pairs.sample(200)
train_feat = sample.feature_dif.to_list()
gmm = GaussianMixture(n_components=3,random_state=0,covariance_type='diag').fit(train_feat)
centers = gmm.means_
# similiarity matrix

S = []
for vec in train_feat:
    [p] = gmm.predict([vec])
    S.append(
        np.exp(
            abs(centers[p] - vec)**2
            /gmm.covariances_[p]
            *-1 )
    )


    # break

# %%

gmm.covariances_[0].view()
array([[ 0.15596063, -0.07123086, -0.00638823, -0.05638781,  0.04828876, -0.01286561, -0.00759402],
       [-0.07123086,  0.17676474, -0.03848003, -0.00130994, -0.05471005, -0.02865844, -0.0723488 ],
       [-0.00638823, -0.03848003,  0.21250689, -0.04614785,  0.01220009, -0.01868899, -0.01123327],
       [-0.05638781, -0.00130994, -0.04614785,  0.18094024, -0.03013355, 0.03940404,  0.03114289],
       [ 0.04828876, -0.05471005,  0.01220009, -0.03013355,  0.20883189, -0.02481747, -0.04827404],
       [-0.01286561, -0.02865844, -0.01868899,  0.03940404, -0.02481747, 0.10203363,  0.1107339 ],
       [-0.00759402, -0.0723488 , -0.01123327,  0.03114289, -0.04827404, 0.1107339 ,  0.18900878]])

#%%

#%% Clustering agents using their reported importance scores
'''sse = []
imax = 11
sillou = []
for i in range(2,imax):
    km = KMeans(n_clusters=i,init='k-means++',random_state=0)
    km.fit(agent_imp.imp.to_list())
    sse.append(km.inertia_)
    sillou.append(silhouette_score(agent_imp.imp.to_list(),km.labels_))
    # km.cluster_centers_

plt.plot(list(range(2,imax)),sillou)'''
data = np.array(agent_imp.imp.to_list())
agentf = 'agegroup'
af = agent_features[agentf].to_numpy()[...,np.newaxis]
with_agent = np.concatenate((af,data),axis=1)

km = KMeans(n_clusters=3,init='k-means++',random_state=0)
km.fit(with_agent)

# plt.plot(list(range(2,imax)),sse)
centers = pd.DataFrame(km.cluster_centers_,columns = ['Agent:{}'.format(agentf)]+features)
print(centers.to_markdown())
print(km.inertia_)
print(np.unique(km.labels_,return_counts=True))
# %% Now we have the data in pairwise feature differences and scores

''' Simple 2-D NN to use for each agent for clustering
'''
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a1 = nn.Linear(7,5)
        self.a2 = nn.Linear(5,1)
    def forward(self,x):
        o1 = self.a1(x)
        return self.a2(o1)

models = []
for agent,data in tqdm(pairs.groupby('agent')):
    # model = LogisticRegression(random_state=0).fit(X=data.feature_dif.to_list(),y=data.score_dif.to_list())
    # for i,feat in enumerate(features):
    x = torch.tensor(data.feature_dif.to_list()).float()
    y = torch.tensor(data.score_dif.to_list()).float()

    model = Net()
    criterion = nn.MSELoss(reduction='sum') 
    optimizer = torch.optim.SGD(model.parameters(),lr=0.0001)

    for t in range(500):
        y_pred = model(x)

        loss = criterion(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    models.append(model)
    # break
