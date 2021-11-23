#%%
%reload_ext autoreload
%autoreload 2
from lptree import *
from helpers import *

#%% loading pickles
imp_vars = pickle.load(open('vars_inwon_round78_multifeat','rb'))
# imp_vars = pickle.load(open('vars_inwon_round7','rb'))

xx = imp_vars['genX_all']
yy = imp_vars['genY_all']

xtrain,xtest,ytrain,ytest = train_test_split(xx,yy)
#%%
tree = lptree()
tree.learn(xtrain,ytrain)
# %%
