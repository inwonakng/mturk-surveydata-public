from helpers import *

class lptree:
    def __init__(self,type='up-ui'):
        self.type = 'up-ui'
        self.root = self.node()
        self.nodes = {}
        self.weights = np.array([])
        self.threshold = 20

    def generate_labels(self):
        return 0

    def choose_attribute(self,node,X,Y):
        attributes = set([e for e,x in enumerate(X[0])])
        labels = set(list(self.nodes.keys()))
        allowed = attributes - labels
        allowed = list(allowed)
        shuffle(allowed)
        chosen = None
        for attr in allowed:
            ones = [x[attr] for i,x in enumerate(X) if Y[i] == 1]
            zeros = [x[attr] for i,x in enumerate(X) if Y[i] == 0]
            # print(ones)
            # print(zeros)
            oneup = min(ones) > max(zeros)
            zeroup = min(zeros) > max(ones)
            if oneup and not zeroup:
                thres = (min(ones) - max(zeros))/2 + max(zeros)
                chosen = {'attr': attr, 'lambda':(
                    lambda xs: (xs + self.threshold >= thres)or(xs - self.threshold >= thres))}
            if zeroup and not oneup:
                chosen = {'attr': attr, 'lambda':(
                    lambda xs: (xs + self.threshold < thres)or(xs - self.threshold < thres))}
            else:
                continue
            if chosen: continue
        if not chosen:
            print('COULD NOT FINISH!') 
            return 'COULD NOT FINISH!'
        
        return chosen['attr'],chosen['lambda']
    
    def learn(self,X,Y):
        assert(len(X) == len(Y))
        self.root = self.node()
        node = self.root
        for i in X[0][1:]:
            node.child = self.node()
            node = node.child
        node = self.root
        while node.child:
            r = self.choose_attribute(node,X,Y)
            if 'C' in r: break
            label,table = r[0],r[1]
            node.label = label
            node.table = table
            node = node.child
            self.nodes[label] = table
        

    def runtree(self,X):
        node = self.root()
        tt = self.threshold
        while node.child:
            high = tt + X[node.label] 
            low = tt - X[node.label]
            outcome = (node.table(high) is node.table(low))
            if outcome: return node.table(high)
        return False
    
    # def predict(self,X):
        # for x in X:
            
    class node:
        def __init__(self, label = '', table = None):
            self.parent = None
            self.child = None
            self.label = label
            self.table = table
        
        def __str__(self):
            txt =  'label:  ' + str(self.label) + '\n'
            if self.parent != None:
                txt += 'parent: ' + str(self.parent.label) + '\n'
            if self.child != None:
                txt += 'child:  ' + str(self.child.label) + '\n'
            if self.table != None:
                txt += 'table:  ' + str(self.table) + '\n'
            return txt

        def set_parent(self,othernode):
            '''set the othernode as the parent node of this node'''
            assert(type(othernode) == type(self))
            self.parent = othernode
            othernode.child = self

        def set_child(self,othernode):
            '''set the othernode as the child node of this node'''
            assert(type(othernode) == type(self))
            self.child = othernode
            othernode.parent = self

        def get_parent(self,othernode):
            assert(type(othernode) == type(self))
            assert(self.parent != None)
            return self.parent

        def get_child(self,othernode):
            assert(type(othernode) == type(self))
            assert(self.child != None)
            return self.child
           

