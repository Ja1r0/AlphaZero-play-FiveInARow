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

class Node:
    def __init__(self,s):
        legal_movs=Legal_movements(s)
        row_idx,col_idx=np.nonzero(legal_movs)
        self.legal_idx=list(zip(row_idx,col_idx))
        pass

if __name__ == '__main__':
    s=np.array([[0,1,0],[1,1,0],[0,0,0]])
    node=Node(s)
    print(node.legal_idx)

class Montecarlo:
    def __init__(self,board,):
        pass
    def update(self,state):
        pass
    def get_play(self):
        pass
    def run_simulation(self):
        pass