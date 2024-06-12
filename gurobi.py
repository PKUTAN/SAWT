import numpy as np
import torch
from gurobipy import Model,GRB,quicksum
from tqdm import tqdm

import utils

if __name__ == '__main__':

    dataset_fname = ['./synthetic_data/erdos10_0.7_F_test.npy','./synthetic_data/erdos10_0.7_positions_test.npy']

    Flows = np.load(dataset_fname[0])
    positions = np.load(dataset_fname[1])

    b = Flows.shape[0]
    N = Flows.shape[1]

    positions = torch.from_numpy(positions)

    D_mat = []
    for i in range(b):
        D_mat.append(utils.distance_matrix(positions[i],positions[i]))

    D_mat = np.stack(D_mat,axis=0)

    out = []
    for i in tqdm(range(b)):
        F,D = Flows[i] , D_mat[i]

        m = Model('QAP')
        # m.Params.TimeLimit = 600
        m.Params.OutputFlag = 0
        x = m.addMVar(shape = (N,N), vtype= GRB.BINARY,name='x')
        m.setObjective(quicksum(quicksum(F*(x@D@x.T))),GRB.MINIMIZE)

        m.addConstrs(quicksum(x[i,j] for j in range(N)) ==1 for i in range(N))
        m.addConstrs(quicksum(x[i,j] for i in range(N)) ==1 for j in range(N))
        # m.addConstr(quicksum(quicksum(x)) == 20)
        # m.Params.Method = 4
        # m.Params.Presolve = 0
        m.optimize()
        out.append(m.objVal)
    
    out = np.array(out)
    # np.save('./Gurobi/gurobi_20_600s.npy',out)
    print('mean:{}'.format(out.mean()))
    import pdb; pdb.set_trace()
        

