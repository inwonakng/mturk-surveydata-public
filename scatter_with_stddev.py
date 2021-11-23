#%% 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
meanvals = [('age',         7.160475483),
            ('gender',      3.592868),
            ('health',      7.057949),
            ('income',      3.044577),
            ('dependents',  5.829123),
            ('survival difference',     6.441308),
            ('survival with',    6.512630)]

stddev = [  ('age',         2.459393027),
            ('gender',      3.048332),
            ('health',      2.430198),
            ('income',      2.905419),
            ('dependents',  2.853399),
            ('survival differece',     2.517774),
            ('survival with',    2.555964)]


cmap = matplotlib.cm.get_cmap('viridis')
barcolors = {
    'age':cmap(0.1),
    'gender':cmap(0.2),
    'health':cmap(0.3),
    'income':cmap(0.4),
    'education':cmap(0.5),
    'dependents':cmap(0.6),
    'survival with':cmap(0.7),
    'survival difference':cmap(0.8),
}            
# %%

x = [m[0] for m in meanvals]
y = [m[1] for m in meanvals]
yerr = [m[1] for m in stddev]
colors = [barcolors[xx] for xx in x]

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)

fig,ax = plt.subplots(figsize=(8,5))

trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
# trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData

for xx,yy,cc,err in zip(x,y,colors,yerr):
    (_, cap, _) = ax.errorbar([xx], [yy], yerr=[err], marker="o",capsize=5,solid_capstyle='butt', linestyle="None", transform=trans1,c = cc)
    # cap[0].set_markeredgewidth(10)

ax.annotate(round(y[0],3),(0,y[0]))

for xx,yy in enumerate(y[1:]):
    ax.annotate(round(yy,3),(xx+0.3,yy))
# fig.tight_layout()
plt.ylabel("Average agent reported scores")
# plt.xlabel("Features")?
fig.autofmt_xdate()
plt.tight_layout()
# %%

# %%
