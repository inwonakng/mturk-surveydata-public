import csv
import scipy.stats
import pickle

def main():
    whole,first = read_file('2_plant_scenario/data/round2/plant_ranking_50.csv')

    pp = loadpickle('2_plant_scenario/data/round2/plant_pickle')

    columnnames = [
        'WorkerId',
        'WorkerAge',
        'WorkerEdu',
        'WorkerGender',
        'Question #',
        'Proposal #',
        'Proposal Score',
        'C1 Population',
        'C2 Population',
        'Gain',
        'Life decrease',
        'Decrease chance',
        'Number city',
        'Affected city',
        'User city',
        'Survey score',
        'written response'
    ]

    relevants = {
        'approved': first.index('AssignmentStatus'),
        'worker':   first.index('WorkerId'),
        'age':      first.index('Answer.agegroup'),
        'edu':      first.index('Answer.education'),
        'gender':   first.index('Answer.gender'),       
        # first.index('Answer.q'+str(i)+'_no.no'),
        # first.index('Answer.q'+str(i)+'_yes.yes'),
        'usercity': first.index('Answer.usercity'),
        'written':  first.index('Answer.written'),
        'hardness': first.index('Answer.q3survey'),
        'op0':      first.index('Answer.optionset0'),
        'q0r0':     first.index('Answer.q0response0'),
        'q0r1':     first.index('Answer.q0response1'),
        'q0r2':     first.index('Answer.q0response2'),
        'q0r3':     first.index('Answer.q0response3'),
        'op1':      first.index('Answer.optionset1'),
        'q1r0':     first.index('Answer.q1response0'),
        'q1r1':     first.index('Answer.q1response1'),
        'q1r2':     first.index('Answer.q1response2'),
        'q1r3':     first.index('Answer.q1response3'),
        'op2':      first.index('Answer.optionset2'),
        'q2r0':     first.index('Answer.q2response0'),
        'q2r1':     first.index('Answer.q2response1'),
        'q2r2':     first.index('Answer.q2response2'),
        'q2r3':     first.index('Answer.q2response3')
    }

    values = []

    c1count = 0
    c2count = 0

    u1p1 = 0
    u1p2 = 0
    u2p1 = 0
    u2p2 = 0

    for j in whole:
        if j[relevants['approved']] != 'Approved': continue

        # checking city distribution
        c = j[relevants['usercity']]
        if c == 'C1': c1count += 1
        elif c == 'C2': c2count += 1

        q1affected = [p['affected'][0] for p in pp[int(j[relevants['op1']])]]
        q2affected = [p['affected'][0] for p in pp[int(j[relevants['op2']])]]

        for p in q1affected:
            if c == 'C1' and p == 'C1': u1p1 += 1
            elif c == 'C1' and p == 'C2': u1p2 += 1
            elif c == 'C2' and p == 'C1': u2p1 += 1
            elif c == 'C2' and p == 'C2': u2p2 += 1

        for p in q2affected:
            if c == 'C1' and p == 'C1': u1p1 += 1
            elif c == 'C1' and p == 'C2': u1p2 += 1
            elif c == 'C2' and p == 'C1': u2p1 += 1
            elif c == 'C2' and p == 'C2': u2p2 += 1

        for i in range(3):
            if i == 0: 
                numcity = 1
                affected = 'N/A'
                usercity = 'N/A'
            else: 
                numcity = 2
                affected = ''
                usercity = ''
        
            for k in range(4):
                opval = pp[int(j[relevants['op'+str(i)]])][k]

                if affected != 'N/A': affected = opval['affected'][0]
                if usercity != 'N/A': usercity = c

                valuerow = [
                    j[relevants['worker']],
                    j[relevants['age']],
                    j[relevants['edu']],
                    j[relevants['gender']],
                    str(i),
                    str(k),
                    j[relevants['q'+str(i)+'r'+str(k)]],
                    opval['population']['C1'],
                    opval['population']['C2'],
                    opval['eco_gain'],
                    opval['life_loss'],
                    opval['loss_chance'],
                    str(numcity),
                    affected,
                    usercity,
                    j[relevants['hardness']],
                    j[relevants['written']]
                ]
                values.append(valuerow)

    print('all:',len(whole))    
    print('c1:',c1count)
    print('c2:',c2count)
    
    print('User-C1, Plant-C1:',u1p1)
    print('User-C1, Plant-C2:',u1p2)
    print('User-C2, Plant-C1:',u2p1)
    print('User-C2, Plant-C2:',u2p2)

    assert(len(values[0])==len(columnnames))

    write_parse('2_plant_scenario/data/round2/cleaned_ranking.csv',columnnames,values)

def loadpickle(filename):
    f = open(filename,'rb')
    pp = pickle.load(f)
    f.close()
    return pp

def idx(ls,el):
    idxs = []
    for i in range(len(ls)):
        if ls[i] == el: idxs.append(i)
    return idxs

def rank(a_score,b_score,c_score):
    literal = ['A','B','C']
    ranked = list(scipy.stats.rankdata([a_score,b_score,c_score]))

    best = max(ranked)
    idxs = idx(ranked,best)
    # ties exist
    if len(idxs) > 1:
        if len(idxs)==3: return 'A=B=C'
        else:
            last = (set([0,1,2])-set(idxs)).pop()
            return literal[idxs[0]]+'='+literal[idxs[1]]+'>'+literal[last]
    # at least no tie with first rank
    else:
        cp = ranked.copy()
        cp.remove(best)
        second = max(cp)
        second_idxs = idx(ranked,second)
        # tie for second
        if len(second_idxs) == 2:
            return literal[idxs[0]]+'>'+literal[second_idxs[0]]+'='+literal[second_idxs[1]]
        else:
            last = (set([0,1,2])-set(idxs)-set(second_idxs)).pop()
            return literal[idxs[0]]+'>'+literal[second_idxs[0]]+'>'+literal[last]

def read_file(path_results):
    whole = []
    with open(path_results,newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='\"')
        for row in spamreader:
            whole.append(row)
    first = whole.pop(0)

    return whole,first

def find_indexs(first):
    options = ['A','B','C']
    num_qs = 4

    useful_indexes = []
    for i in range(num_qs):
        indexs = {}
        indexs['essay'] = first.index('Answer.q'+str(i)+'essay')
        indexs['optionset'] = first.index('Answer.q'+str(i)+'optionset')
        for o in options:
            indexs['option'+o] = first.index('Answer.q'+str(i)+'option'+o)
        useful_indexes.append(indexs)

    demographic = { 'agegroup': first.index('Answer.agegroup'),
                    'education': first.index('Answer.education'),
                    'gender': first.index('Answer.gender'),
                    'workerid': first.index('WorkerId')}

    return useful_indexes,demographic

def parse_options(optionset):
    base = {'age':'N/A','gender':'N/A','purpose of trip':'N/A','career':'N/A','health':'N/A','survival':'N/A','focus':'N/A'}
    for o,v in optionset.items():
        if o in base: base[o] = v
        if o == 'purpose': base['purpose of trip'] = v
        if o == 'with': base['survival'] = {'with':v,'without':optionset['without']}
        if o == 'relationship': base['career'] = v
    return base

def write_parse(name,columnnames,values):
    with open(name, 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
        w.writerow(columnnames)
        idx = 0
        for v in values:
            w.writerow (v)

if __name__ == '__main__':
    main()
