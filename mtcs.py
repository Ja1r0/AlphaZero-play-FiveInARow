import numpy as np

'''
input
=====
network parameters
------------------
input an state s,can output P and v.
'''
def Legal_movements(s):
    '''
    Input
    =====
    {ndarray} of size (7,7)
    0---there is no chessman
    1---there has a chessman
        the current state of the board
    Output
    ======
    {ndarray} of size (7,7)
    0---it is not a legal movement
    1---it is a legal movement
        the legal movements of the current player.
    in other words,the no chessman places.
    '''
    zeros=np.zeros_like(s)
    legal_movements=np.power(zeros,s)
    return legal_movements

def Dihedral_reflect(s):
    '''
    :param s: {ndarray} of board
    :return: rotate and reflect to the board,8 conditions in total
    '''
    s_r1=np.zeros_like(s)
    s_r2=np.zeros_like(s)
    s_r3 = np.zeros_like(s)
    s_rr = np.zeros_like(s)
    s_rr1 = np.zeros_like(s)
    s_rr2 = np.zeros_like(s)
    s_rr3 = np.zeros_like(s)
    r=s.shape[0]
    c=s.shape[1]
    for i in range(r):
        for j in range(c):
            s_r1[j][c-1-i]=s[i][j]
            s_r2[c-1-i][c-1-j]=s[i][j]
            s_r3[c-1-j][c-1-(c-1-i)]=s[i][j]
            s_rr[i][c-1-j]=s[i][j]
            s_rr1[c-1-j][c - 1 - i] = s[i][j]
            s_rr2[c - 1 - i][c - 1 - (c-1-j)] = s[i][j]
            s_rr3[c - 1 - (c-1-j)][c - 1 - (c - 1 - i)] = s[i][j]
    s_rotates=[s,s_r1,s_r2,s_r3,s_rr,s_rr1,s_rr2,s_rr3]
    return s_rotates


class Node:
    def __init__(self,s):
        legal_movs=Legal_movements(s)
        row_idx,col_idx=np.nonzero(legal_movs)
        self.legal_idx=list(zip(row_idx,col_idx))
        self.stat=[{'N':0,'W':0.0,'Q':0.0,'P':0.0} for _ in range(len(self.legal_idx))]
        self.edges=dict(zip(self.legal_idx,self.stat))
        self.childs_num=len(self.edges)
    def update(self,action,**kwargs):
        '''
        :param action: {tuple} e.g. (0,0) means location w.r.t the legal action
        :param kwargs: e.g. (action=(0,0),Q=1.1,N=2)
        '''
        edge=self.edges[action]
        for name,value in kwargs.items():
            edge[name]=value
    def select(self,c):
        sum_N=0
        for action,stat in self.edges.items():
            sum_N+=stat['N']
        edges_U=[]
        for action,stat in self.edges.items():
            edge_U=stat['Q']+c*stat['P']*sum_N/(1+stat['N'])
            edges_U.append([edge_U,action])
        edges_U.sort()
        action=edges_U[0][1]
        return action
'''
if __name__ == '__main__':
    s=np.array([[0,1,0],[0,0,1],[0,0,0]])
    node=Node(s)
    print(node.childs_num)
'''

class Montecarlo:
    def __init__(self,s_root):
        self.nodes=[]
        self.nodes.append(Node(s_root))
    def update(self,state):
        pass
    def get_play(self):
        pass
    def run_simulation(self):
        pass

def Mtcs(s_root):
    nodes=[]
    nodes.append(Node(s_root))
    current_s=nodes[0]
    