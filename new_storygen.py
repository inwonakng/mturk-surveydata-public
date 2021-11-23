'''
New version of storygen to make every possible option
'''

#%%
import json
import itertools

'''copied functions from storygen'''
def parse_categories(data):
    cate = data.pop('categories')
    good_categ = {}
    for c,l in cate.items():
        good_categ[c] = list(l.values())
    return good_categ

def parse_bad_combo(data, categories):
    bc = data.pop('bad combo')
    bad_combos = {}
    for b,c in bc.items():
        bad_combos[b] = {}
        bad_combos[b]['categories'] = c['categories']
        bad_combos[b]['classes'] = []
        if c['classes']:
            for cat,ids in c['classes'].items():
                for i in ids:
                    bad_combos[b]['classes'].append(categories[cat][int(i)])
    return bad_combos

# returns false if the given combo is invalid
def check_combo(bad_combos, combo):
    # not_allowed = set()

    if not combo['age'] in bad_combos: 
        return not(combo['survival with jacket'] <= combo['survival without jacket'])

    not_allowed = bad_combos[combo['age']]
    return not(combo['income level'] in not_allowed['classes']
                or combo['number of dependents'] in not_allowed['classes']
                or combo['survival with jacket'] <= combo['survival without jacket'])

def recursive_build(in_categ,current):
    ccopy = in_categ.copy()
    cname = list(ccopy)[0]
    now = ccopy.pop(cname)
    print(cname)
    print(now)

    added = []
    if current:
        for n in now:
            for c in current:
                cc = c.copy()
                cc[cname] = n
                added.append(cc)
    else:
        added = [{cname:n} for n in now]

    if not ccopy:
        return added

    return recursive_build(ccopy,added)
#%%

scenario = '1_airplane_scenario/'

with open(scenario + 'newclass.json') as file:
    data = json.load(file)

categories = parse_categories(data)
badcombo = parse_bad_combo(data,categories)

# extra = ['survival with jacket','survival without jacket']

categories['survival without jacket'] = list(range(5,41,5))
categories['survival with jacket'] = list(range(5,91,5))

all_combs = recursive_build(categories,[])

filtered = [a for a in all_combs if check_combo(badcombo,a)]

formatted = []
for f in filtered:
    fnew = f.copy()
    v = f['survival without jacket']
    fnew['survival without jacket'] = '{}%'.format(v)
    v = f['survival with jacket']
    fnew['survival with jacket'] = '{}%'.format(v)
    formatted.append(fnew)


json.dump(formatted,open('all_combos.json','w'))

# for c in categories:
#     all_combs.append()


# %%
