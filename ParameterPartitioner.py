import torch, torch.nn as nn, torch.optim as optim, numpy as np

def partition_params(model):
    '''
    model: pytorch model; which subclasses nn.Module
    returns set; of param names
    '''
    def params_not_in_module(model):
        children = lambda dct: set([mn.split('.')[0] for mn, m in dct])
        child_modules = children(model.named_modules())
        child_params = children(model.named_parameters())
        return list(child_params - child_modules)

    partition = set([])
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            partition.add('%s.%s' % (mn, pn) if mn else pn)

    for pn in params_not_in_module(model): 
        partition.add(pn)

    return partition

def rec(conditions, paths, depth=0):
    if len(conditions)<=depth: return paths
    cond = conditions[depth]
    # partition paths
    satisfies, fails = [], []
    for path in paths:
        if cond in path:
            satisfies.append(path)
        else:
            fails.append(path)
    return (rec(conditions, fails, depth+1), 
            rec(conditions, satisfies, depth+1))

def multiple_partition(model, conditions, comb=None):
    '''
    partitions all of the model parameters,
    in the order of the conditions given 
    '''
    prt = partition_params(model)# returns all parameters 
    partition = rec(conditions, prt)

    def rec_tup(ixlist, tup):
        if len(ixlist)>1:
            return rec_tup(ixlist[1:], tup[ixlist[0]])
        return tup[ixlist[0]]

    n = len(conditions)
    x = np.array([0, 1], dtype=int)
    xs = np.array(np.meshgrid(*([x]*n))).T.reshape(-1, n)
    remaining = set([''.join([str(xi) for xi in x_]) for x_ in xs])

    if comb is not None:
        p = {}    
        for pname, ixs in comb.items():
            if ixs is not None:
                p[pname] = []
                for ix in ixs:
                    p[pname] += rec_tup(ix, partition)
                    remaining.remove(''.join([str(xi) for xi in ix]))
        # put the remainder into the key with value None
        for pname, ixs in comb.items():
            if ixs is None:
                p[pname] = []
                for r in remaining:
                    ix = [int(i) for i in r]
                    p[pname] += rec_tup(ix, partition)
        return p
    return {''.join([str(xi) for xi in ix]): rec_tup(ix, partition) for ix in xs}

def optimisation_groups(model, partition, opt_kwargs):
    '''
    model: pytorch model; which subclasses nn.Module
    partition: dict; keys are types and values are param names
    opt_kwargs: dict; keys are types and values are optimiser kwargs
    returns list; each elem are dicts for each optimiser
    '''
    assert sorted(partition.keys())==sorted(opt_kwargs.keys()), 'partition, opt_kwargs must have same keys'
    
    param_dict = {pn: p for pn, p in model.named_parameters()}
    optim_groups = []

    for type_, param_names in partition.items():
        group = {"params": [param_dict[pn] for pn in sorted(list(param_names))]}
        group.update(opt_kwargs[type_])
        optim_groups.append(group)

    return optim_groups



if __name__ == '__main__':

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(16, 33, 3, stride=2)
            self.deep_param = nn.Parameter(torch.zeros(1, 20, 10))
            self.ln = nn.LayerNorm(10)
            self.attn = nn.MultiheadAttention(10, 2)
            self.mlp = nn.Sequential(
                nn.Linear(10, 4 * 10),
                nn.GELU(),
                nn.Linear(4 * 10, 10),
                nn.Dropout(0.1),
            )

    class SubModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.Linear(10, 53, bias=False)

    class SomeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(53, 10)
            self.pos_emb = nn.Parameter(torch.zeros(1, 20, 10))
            self.drop = nn.Dropout(0.1)
            self.blocks = nn.Sequential(*[Block() for _ in range(2)])
            self.ln_f = nn.LayerNorm(10)
            self.head = nn.Linear(10, 53, bias=False)
            self.sub = SubModel()



    model = SomeModel()


    conditions = ['attn', 'weight', 'mlp']
    '''
    partition the parameters in the model by the according to 
    the conditions
    '''
    partition = multiple_partition(model, conditions)

    for k, v in partition.items():
        print('\n'+k+'\n%s'%v)

    '''
    same again, but now we can bucket the partition into groups
    '''

    combine = {'gr1':[[0,0,0],[1,0,0],[0,0,1],[1,0,1]],# combine all non weights
               'gr2':[[0,1,0],[1,1,0]],# weight and not mlp
               'gr3': None}

    partition = multiple_partition(model, conditions, comb=combine)

    for k, v in partition.items():
        print('\n'+k+'\n%s'%v)

    '''
    now we may want to give these partitioned parameters different
    optimisers
    '''

    # e.g. split by weight/bias
    typ_opt_kwargs = {'gr1': {"weight_decay": 0.1},
                      'gr2':   {"weight_decay": 0.0},
                      'gr3':   {}}
    opt_groups = optimisation_groups(model, partition, typ_opt_kwargs)






