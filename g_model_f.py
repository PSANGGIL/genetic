from torch import nn
import random
import numpy as np
import torch
from sympy import *
from copy import copy
from tqdm import tqdm
import itertools
list_func = [
             ### double argument
             lambda x, a: x+a,
             lambda x, a: x*a,
             lambda x, a: torch.exp(-torch.abs(a)*x*x),

             ### single argument
             lambda x, a: a*torch.log(torch.abs(x)),
             lambda x, a: a*torch.exp(-torch.abs(x)),
             lambda x, a: a*torch.sigmoid(x),
             lambda x, a: a*torch.tanh(x),
             lambda x, a: a*torch.erf(x),

             lambda x, a: a*torch.abs(x),
             lambda x, a: a*torch.pow(torch.abs(x), torch.tensor(2  , requires_grad=False)),
             lambda x, a: a*torch.pow(torch.abs(x), torch.tensor(3  , requires_grad=False)),
             lambda x, a: a*torch.pow(torch.abs(x), torch.tensor(1/2, requires_grad=False)),
             lambda x, a: a*torch.pow(torch.abs(x), torch.tensor(1/3, requires_grad=False)),

             # special form
             lambda x, a: torch.sqrt(   a/(1+x*x) ),
             lambda x, a: torch.sqrt( a*x*x/(1+x*x) )
            ]

class ExpressionNode(object):
    def __init__(self, my_list_func, my_list_var):
        super(ExpressionNode, self).__init__()
        self.my_list_func= my_list_func
        self.my_list_var = my_list_var 
   
    def make_child(self, func, var):
        return ExpressionNode(copy(self.my_list_func)+[func], copy(self.my_list_var)+[var]) 
    
    def eval_error (self, node, nabla, inp1, inp2, grid_w, out):
        
        factor1 = torch.sigmoid(node * nabla )
        factor2 = 1-factor1
        
        return torch.sum(torch.abs( grid_w*(out - factor1*inp1 - factor2*inp2)))
    
    def random_guess(self, inp1,inp2):
        self.device = inp1.device 
        self.dtype = inp1.dtype 
        
        func = random.choice(list_func)
        
        if func.__code__.co_argcount==1:
            var= None
        else:
            var = random.choice( [torch.rand(1, requires_grad=True, device=self.device, dtype=self.dtype),
                                  1         *torch.ones(1, requires_grad=False, device=self.device, dtype=self.dtype),
                                  2         *torch.ones(1, requires_grad=False, device=self.device, dtype=self.dtype),
                                  np.pi     *torch.ones(1, requires_grad=False, device=self.device, dtype=self.dtype),
                                  np.exp(1) *torch.ones(1, requires_grad=False, device=self.device, dtype=self.dtype),
                                  -1        *torch.ones(1, requires_grad=False, device=self.device, dtype=self.dtype),
                                  -2        *torch.ones(1, requires_grad=False, device=self.device, dtype=self.dtype)
                                  ] )
            var = torch.rand(1, requires_grad=True, device=self.device, dtype=self.dtype)
        return  func, var
           
def w(directory, mol_idx):
    f = np.load(directory + '/' + str(mol_idx).zfill(6) + '/grid.npy')
    xyz_w = f[:,-1]
    return xyz_w

def rho(directory, mol_idx):
    f = np.load(directory + '/' + str(mol_idx).zfill(6) + '/rho.npy')
    rho_c = f[0,:]
    grad  = f[1:4,:]
    nabla = f[-2,:]
    return rho_c

def grad(directory, mol_idx):
    f = np.load(directory + '/' + str(mol_idx).zfill(6) + '/rho.npy')
    grad_c  = f[1:4,:]
    return grad_c

def nabla(directory, mol_idx):
    f = np.load(directory + '/' + str(mol_idx).zfill(6) + '/rho.npy')
    grad_c  = f[-2,:]
    return grad_c

def target(directory, mol_idx):
    f = np.load(directory + '/' + str(mol_idx).zfill(6) + '/rho.npy')
    target_c = f[-1,:]
    return target_c

if __name__=='__main__':

    random_seed =1 

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


    mol_idx = 3
    directory = '../0119/data_isomer' 
 
    f1 = (lambda x,a :x + a)
    v1 = torch.zeros(1, requires_grad=False)
    
    node = ExpressionNode([[f1,f1,f1]],[[v1,v1,v1]])
    node = ExpressionNode([[f1]],[[v1]])
    
    epochs = 10
    batch_size = 8196 
    
    num_gen   =  20
    num_child =  3 
    max_child =  5 

    bohr_r = 1.88973 # bohr to ang
    
    rho    = rho(directory, mol_idx)
    grad   = grad(directory, mol_idx)
    target = target(directory, mol_idx)
    nabla  = nabla(directory, mol_idx)
    grid_w = w(directory, mol_idx)
    
    TF_factor =  (3*np.pi**2 )
    
    ke_vw     =  (1/8) * (np.linalg.norm((grad), axis = 0))**2 / (rho).reshape(-1)
    ke_tf     =  (3/10)*(TF_factor**(2/3))*((rho)**(5/3))
    
    with torch.no_grad():
        rho   = torch.tensor(rho    )
        nabla = torch.tensor(nabla  ) 
        target= torch.tensor(target )
        grid_w= torch.tensor(grid_w )
        ke_tf = torch.tensor(ke_tf  )
        ke_vw = torch.tensor(ke_vw  )
  
    dataset = torch.utils.data.TensorDataset(rho , nabla , ke_tf , ke_vw , grid_w, target)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  shuffle=True)

    func_tree = []
    var_tree  = []

    for ge in tqdm(range(num_gen)):
        pre_func = []
        pre_var  = []
        min_child= []
        test_child = []
        for i in range(max_child):
            func, var = node.random_guess(ke_tf , ke_vw)
            pre_func.append(func)
            pre_var.append(var)
            for j in range(num_child):
                optimizer = torch.optim.Adam([pre_var[i]], lr = 1e-2)
                for epoch in (range(epochs)):
                    for (inp, nabla, ke_tf, ke_vw, grid_w, target) in loader:
                        trial_value = pre_func[i](inp , pre_var[i]) 
                        error = node.eval_error(trial_value, nabla,  ke_tf , ke_vw ,grid_w, target).requires_grad_(True)
                        error.backward()
                        optimizer.step()
                
                sum_error = 0.0
                sum_total_error = 0.0
                ge_idx = ge
                
                if ge == 0:
                    for (inp, nabla, ke_tf, ke_vw, grid_w, target) in loader:
                        trial_value = pre_func[i](inp , pre_var[i])
                        error = node.eval_error(trial_value, nabla,  ke_tf , ke_vw ,grid_w, target)
                        sum_error += error
                    min_child.append([pre_func[i], pre_var[i], sum_error])

                else:
                    for pre_idx in range(num_child):
                        test_func = []
                        test_var  = []
                        for (inp, nabla, ke_tf, ke_vw, grid_w, target) in loader: 
                            for ge_idx in reversed(([ge for ge in range(ge)])):
                                trial_value = pre_func[i](inp , pre_var[i])
                                total_value = func_tree[ge_idx-1][pre_idx](trial_value , var_tree[ge_idx-1][pre_idx])
                                trial_value = total_value 
                            error = node.eval_error(total_value, nabla,  ke_tf , ke_vw ,grid_w, target)
                            sum_total_error += error
                           
                        test_func = list(itertools.chain.from_iterable(func_tree)) + [pre_func[i]]
                        test_var = list(itertools.chain.from_iterable(var_tree)) + [pre_var[i]]
                    min_child.append([pre_func[i], pre_var[i], sum_total_error])
                    test_child.append([test_func, test_var, sum_total_error])

        min_child = sorted(min_child, key=lambda x: x[-1])[:num_child] 
        test_child = sorted(test_child, key=lambda x: x[-1])[:num_child] 
        
        min_var  = [min_child[i][-2] for i in range(len(min_child))]
        min_error= [min_child[i][-1] for i in range(len(min_child))]
        min_func = [min_child[i][0] for i in range(len(min_child))]
        
        func_tree.append(min_func)
        var_tree.append(min_var)
        
        print(ge, min_error)
        print(ge, min_func )
        node = node.make_child(min_func , min_var)
