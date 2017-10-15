# import torch
import numpy as np
from collections import namedtuple, deque
import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

NimState = list
NimAction = namedtuple('NimAction', ['pile', 'nstones'])
bin_dtype = np.int8

def int2bin(n, width):
    out = np.zeros(width, bin_dtype)
    current = n
    for i in reversed(range(width)):
        out[i] = current % 2
        current //= 2
    return out

def bin2int(bitvec):
    out = 0
    for bit in bitvec:
        out += bit
        out *= 2
    out //= 2
    return out

def calc_nimsum(nimstate, width):
    out = np.zeros(width, bin_dtype)
    for pilesize in nimstate:
        out += int2bin(pilesize, width)
    out %= 2
    return out

assert np.allclose(calc_nimsum([1,2,3], 3),  np.array([0,0,0], bin_dtype))

def calc_width(pilesize):
    return int(np.log2(pilesize)) + 1

def calc_act_val(nimstate):
    width = max(map(calc_width, nimstate))
    nimsum = calc_nimsum(nimstate, width)
    if (nimsum==0).all():
        val = -1
        nimaction = NimAction(pile=0, nstones=1)
    else:
        val = 1
        bigpileidx = np.argmax(nimstate)
        bigpilesize = nimstate[bigpileidx]
        binbigpile = int2bin(bigpilesize, width)
        newbinbigpile = (binbigpile + nimsum) % 2
        newbigpilesize = bin2int(newbinbigpile)
        if newbigpilesize > bigpilesize: newbigpilesize -= 2**int(np.log2(newbigpilesize))
        nimaction = NimAction(pile=bigpileidx, nstones=bigpilesize-newbigpilesize)
    return nimaction, val

def update_nimstate(nimstate, nimaction):
    if nimstate[nimaction.pile] > nimaction.nstones:
        newnimstate = nimstate.copy()
        newnimstate[nimaction.pile] -= nimaction.nstones
    else:
        newnimstate = nimstate[:nimaction.pile] + nimstate[nimaction.pile+1:]
    return newnimstate

def gen_nimstate(width, maxpiles):
    npiles = np.random.randint(1, maxpiles+1)
    return np.random.randint(0, 2**width, size=npiles)

class Mlp1(nn.Module):
    def __init__(self, width, hidsize):
        nn.Module.__init__(self)
        self.l0 = nn.Linear(2*width, hidsize)
        self.l1 = nn.Linear(hidsize, width)
    def forward(self, x0, x1):
        x = torch.cat((x0, x1), 1)
        x = self.l0(x)
        x = F.relu(x)
        x = self.l1(x)
        return x

def train_elwise_xor():
    """
    Lets just check that we can approximate an elementwise XOR operation
    """
    width = 8
    hidsize = 32
    batchsize = 64
    nhidlayers = 1
    lossbuf = deque([], maxlen=100)
    # cell = nn.GRUCell(width, width)

    cell = Mlp1(width, hidsize)
    optimizer = torch.optim.Adam(cell.parameters(), lr=1e-4)
    while True:
        a = torch.ones(batchsize, width)*.5
        x0 = Variable(torch.bernoulli(a))
        x1 = Variable(torch.bernoulli(a))
        y = Variable((x0.data + x1.data) % 2)
        ypred = cell(x0, x1)
        err = ypred - y
        loss = torch.sum(err*err)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lossbuf.append(loss.data[0])
        print(np.mean(lossbuf))

def numpy2var(x):
    return Variable(torch.from_numpy(x.astype(np.float32)))

def train_gameeval(celltype, nouteriter, width, maxpiles):
    np.set_printoptions(precision=3)
    clip = 1.0
    lr = 1e-4
    ninneriter = 1000
    if celltype=='gru':
        memsize = 32
        cell = nn.GRUCell(width, memsize)
        lastlayer = nn.Sequential(nn.Linear(memsize, memsize), nn.ReLU(), nn.Linear(memsize, width))
        parameters = list(cell.parameters()) + list(lastlayer.parameters())
    elif celltype=='mlp':
        memsize = width
        cell = Mlp1(width, hidsize=64)
        lastlayer = lambda x:x
        parameters = cell.parameters()
    else:
        raise NotImplementedError

    optimizer = torch.optim.Adam(parameters, lr=lr)    

    nupdates = 0
    for _ in range(nouteriter):
        lossvals = []
        for _ in range(ninneriter):
            h = Variable(torch.zeros(1, memsize))
            nimstate = gen_nimstate(width, maxpiles)
            for pilesize in nimstate:
                binvec = int2bin(pilesize, width)[None,:]
                x = numpy2var(binvec)
                h = cell(x, h)
            ypred = lastlayer(h)
            y = calc_nimsum(nimstate, width)[None,:].astype(np.float32)

            err = ypred - numpy2var(y)
            loss = torch.sum(err*err)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(parameters,clip)

            optimizer.step()
            lossvals.append(loss.data[0])
            nupdates += 1
        print(f'{nupdates:>8d} {np.mean(lossvals):10.4f}     {str(ypred.data.numpy()):<30}  {str(y):<30}')

def sanitycheck_game():
    nimstate = NimState([2,10,3,1,1,1])
    while len(nimstate) > 0:
        nimaction, val = calc_act_val(nimstate)
        newnimstate = update_nimstate(nimstate, nimaction)
        print(f'{str(nimstate):<20} {str(nimaction)[9:]:<20} {str(calc_nimsum(nimstate,5)):<10} {str(calc_nimsum(newnimstate,5)):<10} {val:>5}')
        nimstate = newnimstate

if __name__ == '__main__':
    # sanitycheck_game()
    # train_elwise_xor()
    train_gameeval(celltype='mlp', nouteriter=200)