import networkx as nx
import numpy as np
from gurobipy import Model,GRB,quicksum
import os

def generate_erdos_qap_instances_0_1(N = 20, p = 0.7):
    weight_F = np.zeros((N, N), dtype=float)
    location_D = np.zeros((N, 2),dtype=float)

    for i in range(N-1):
        for j in range(i+1,N):
            weight_F[i,j] = np.round(np.random.uniform(0,1),2)
            weight_F[j,i] = weight_F[i,j]

    for i in range(N):
        location_D[i][0], location_D[i][1] = np.round(np.random.uniform(0,1),2),np.round(np.random.uniform(0,1),2)

    G_F = nx.erdos_renyi_graph(N, p)

    for i, j in G_F.edges():
        G_F[i][j]['weight'] = weight_F[i, j]

    weight_F_final = np.zeros((N, N), dtype=float)

    for i,j in G_F.edges():
        weight_F_final[i,j] = G_F[i][j]['weight']
        weight_F_final[j,i] = weight_F_final[i,j]

    
    return weight_F_final,location_D

def generate_erdos_qap_instances_int(N = 20, p = 0.7):
    weight_F = np.zeros((N, N), dtype=float)
    location_D = np.zeros((N, 2),dtype=float)

    for i in range(N-1):
        for j in range(i+1,N):
            weight_F[i,j] = np.round(np.random.uniform(0,50))
            weight_F[j,i] = weight_F[i,j]

    for i in range(N):
        location_D[i][0], location_D[i][1] = np.round(np.random.uniform(0,50)),np.round(np.random.uniform(0,50))

    G_F = nx.erdos_renyi_graph(N, p)

    for i, j in G_F.edges():
        G_F[i][j]['weight'] = weight_F[i, j]

    weight_F_final = np.zeros((N, N), dtype=float)

    for i,j in G_F.edges():
        weight_F_final[i,j] = G_F[i][j]['weight']
        weight_F_final[j,i] = weight_F_final[i,j]

    
    return weight_F_final,location_D

def generate_erdos_qap_instances(N = 20, p = 0.7, F_weight = (0,50),D_weight = (0,50)):
    weight_F = np.zeros((N, N), dtype=int)
    weight_D = np.zeros((N, N), dtype=int)
    F_lower, F_upper = F_weight
    D_lower, D_upper = D_weight

    for i in range(N-1):
        for j in range(i+1,N):
            weight_D[i,j] = np.random.randint(D_lower,D_upper)
            weight_D[j,i] = weight_D[i,j]
            weight_F[i,j] = np.random.randint(F_lower,F_upper)
            weight_F[j,i] = weight_F[i,j]

    G_F = nx.erdos_renyi_graph(N, p)
    G_D = nx.erdos_renyi_graph(N, p)

    for i, j in G_F.edges():
        G_F[i][j]['weight'] = weight_F[i, j]
        
    for i, j in G_D.edges():
        G_D[i][j]['weight'] = weight_D[i, j]

    weight_F_final = np.zeros((N, N), dtype=int)
    weight_D_final = np.zeros((N, N), dtype=int)

    for i,j in G_F.edges():
        weight_F_final[i,j] = G_F[i][j]['weight']
        weight_F_final[j,i] = weight_F_final[i,j]

    for i,j in G_D.edges():
        weight_D_final[i,j] = G_D[i][j]['weight']
        weight_D_final[j,i] = weight_D_final[i,j]
    
    return weight_F_final,weight_D_final

def generate_barabasi_qap_instances(N = 20, m = 15, F_weight = (0,50),D_weight = (0,50)):
    weight_F = np.zeros((N, N), dtype=int)
    weight_D = np.zeros((N, N), dtype=int)
    F_lower, F_upper = F_weight
    D_lower, D_upper = D_weight

    for i in range(N-1):
        for j in range(i+1,N):
            weight_D[i,j] = np.random.randint(D_lower,D_upper)
            weight_D[j,i] = weight_D[i,j]
            weight_F[i,j] = np.random.randint(F_lower,F_upper)
            weight_F[j,i] = weight_F[i,j]

    G_F = nx.barabasi_albert_graph(N,m)
    G_D = nx.barabasi_albert_graph(N,m)

    for i, j in G_F.edges():
        G_F[i][j]['weight'] = weight_F[i, j]
        
    for i, j in G_D.edges():
        G_D[i][j]['weight'] = weight_D[i, j]

    weight_F_final = np.zeros((N, N), dtype=int)
    weight_D_final = np.zeros((N, N), dtype=int)

    for i,j in G_F.edges():
        weight_F_final[i,j] = G_F[i][j]['weight']
        weight_F_final[j,i] = weight_F_final[i,j]

    for i,j in G_D.edges():
        weight_D_final[i,j] = G_D[i][j]['weight']
        weight_D_final[j,i] = weight_D_final[i,j]
    
    return weight_F_final,weight_D_final


if __name__ == '__main__':
    from pathlib import Path
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--N',default=20,type=int,help='Scale of QAP')
    parser.add_argument('--p',default=0.7,type=float,help='Probability of Erdos graph')
    parser.add_argument('--train_size',default=5120,type=int,help='training_data size')
    parser.add_argument('--test_size',default=256,type=int,help='testing data size')

    args = parser.parse_args()


    N = args.N
    F_weight = (0,50)
    D_weight = (0,50)
    p = args.p
    m = 50
    instances_name = '0-1'


    if instances_name == 'erdos':
        for i in range(1):
            F,D = generate_erdos_qap_instances(N,p,F_weight,D_weight)

            # Create file content without left padding
            file_content_no_padding = str(N) + "\n\n"
            for matrix in [F, D]:
                for row in matrix:
                    file_content_no_padding += " ".join(map(str, row)) + "\n"
                file_content_no_padding += "\n"

            file_path  = './data/synthetic_data/erdos'+str(N)+'_'+str(p)

            if not Path.exists(Path(file_path)):
                os.mkdir(file_path)
            # Write to file without left padding
            output_filename_no_padding = file_path +'/erdos' + str(N) + '_' + str(i) +'.dat'
            with open(output_filename_no_padding, "w") as file:
                file.write(file_content_no_padding)
    elif instances_name == 'barabasi':
        for i in range(1000):
            F,D = generate_erdos_qap_instances(N,p,F_weight,D_weight)

            # Create file content without left padding
            file_content_no_padding = str(N) + "\n\n"
            for matrix in [F, D]:
                for row in matrix:
                    file_content_no_padding += " ".join(map(str, row)) + "\n"
                file_content_no_padding += "\n"
            
            file_path  = './data/synthetic_data/barabasi'+str(N)+'_'+str(p)

            if not Path.exists(Path(file_path)):
                os.mkdir(file_path)
            # Write to file without left padding
            output_filename_no_padding = file_path + '/barabasi' + str(N) + '_' + str(i) +'.dat'
            with open(output_filename_no_padding, "w") as file:
                file.write(file_content_no_padding)
    elif instances_name == '0-1':
        F_total = []
        postions_total = []
        for i in range(args.train_size):
            F, postions = generate_erdos_qap_instances_0_1(N,p)
            F_total.append(F)
            postions_total.append(postions)
        
            F_total = np.stack(F_total,axis=0)
            postions_total = np.stack(postions_total,axis=0)

            np.save('./synthetic_data/erdos'+str(N)+'_'+str(p)+'_F_train.npy',F_total)
            np.save('./synthetic_data/erdos'+str(N)+'_'+str(p)+'_F_positions_train.npy',postions_total)


        for i in range(args.test_size):
            F, postions = generate_erdos_qap_instances_0_1(N,p)
            F_total.append(F)
            postions_total.append(postions)
        
            F_total = np.stack(F_total,axis=0)
            postions_total = np.stack(postions_total,axis=0)

            np.save('./synthetic_data/erdos'+str(N)+'_'+str(p)+'_F_test.npy',F_total)
            np.save('./synthetic_data/erdos'+str(N)+'_'+str(p)+'_F_positions_test.npy',postions_total)

        # F_total = []
        # postions_total = []
        # for i in range(64):
        #     F, postions = generate_erdos_qap_instances_0_1(N,p)
        #     F_total.append(F)
        #     postions_total.append(postions)
        
        # np.save('./synthetic_data/erdos100_0.7_F_test.npy',F_total)
        # np.save('./synthetic_data/erdos100_0.7_positions_test.npy',postions_total)
    elif instances_name == 'erdos_int':
        F_total = []
        postions_total = []

        for i in range(1):
            F, postions = generate_erdos_qap_instances_int(N,p)
            F_total.append(F)
            postions_total.append(postions)
        
        F_total = np.stack(F_total,axis=0)
        postions_total = np.stack(postions_total,axis=0)

        np.save('./synthetic_data/erdos10_0.7_F_train_toy_1.npy',F_total)
        np.save('./synthetic_data/erdos10_0.7_positions_train_toy_1.npy',postions_total)

        F_total = []
        postions_total = []
        for i in range(1):
            F, postions = generate_erdos_qap_instances_int(N,p)
            F_total.append(F)
            postions_total.append(postions)
        
        np.save('./synthetic_data/erdos100_0.7_F_test_toy_1.npy',F_total)
        np.save('./synthetic_data/erdos10_0.7_positions_test_toy_1.npy',postions_total)


