
class Network:
    def __init__(self):
    def predict(self,s,history):
        return P,v
    def update(self,samples):

class Mtcs_search:
    def __init__(self,network):
    def move_prob(self,root_s):
        return pi
    def move(self):
        return s
class Data_pool:
    def __init__(self):
    def store(self,sample):
    def sampling(self,batch_size):
        return samples

class Game:
    def __init__(self):
    def winner(self,s):
        return r_T
    
def pipeline():
    net=Network()
    game=Game()
    data_pool=Data_pool()
    for i in iterations_num:
        mtcs=Mtcs_search(net)
        s = s_0
        for t in timesteps_num:
            pi=mtcs.move_prob(s)
            s=mtcs.move()
            if search_value < threshold or t > maximum_length:
                r_T=game.winner(s)
                data_pool.store(sample)
                break
        samples=data_pool.sampling(batch_size=B)
        net.update(samples)