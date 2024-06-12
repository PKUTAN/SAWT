import random
import numpy as np
from tqdm import tqdm

def form_per(sol):
    # Assume sol is a list of indices between (0,N)
    N = len(sol)
    per = np.zeros((N,N))
    
    for i,j in zip(range(N), sol):
        per[i,j] = 1

    return per

def objective_function(F,D,sol):
    per = form_per(sol)
    obj = np.sum(F*(per@D@per.T))  
    return obj

def create_neighbour(index1,index2,solution):
    neighbour_solution = solution[:]
    neighbour_solution[index1], neighbour_solution[index2] = neighbour_solution[index2], neighbour_solution[index1]
    return neighbour_solution


def tabu_search(F,D,max_iterations, num_values,max_tabu_size):
    current_solution = [x for x in range(num_values)]
    best_solution = current_solution[:]

    tabu_list = []
    for iteration in range(max_iterations):
        neighbours = [create_neighbour(i,j,current_solution) for i in range(num_values-1) for j in range(i+1,num_values)]
        neighbours = [x for x in neighbours if str(x) not in tabu_list]

        if not neighbours:
            break

        best_neighbour = min(neighbours, key=lambda x: objective_function(F, D, x))

        current_solution = best_neighbour
        tabu_list.append(str(current_solution))

        if len(tabu_list) > max_tabu_size:
            tabu_list.pop(0)

        if objective_function(F, D, best_neighbour) < objective_function(F, D, best_solution):
            best_solution = best_neighbour
            
    return objective_function(F,D,best_solution)
    
if __name__ == '__main__':
    F = np.load('/home/tzt/learning-2opt-drl/synthetic_data/erdos100_0.7_F_test.npy')
    D = np.load('/home/tzt/learning-2opt-drl/synthetic_data/erdos100_0.7_positions_test.npy')
    from utils import distance_matrix

    out = []
    for i in tqdm(range(F.shape[0])):
        out.append(tabu_search(F[i],distance_matrix(D[i],D[i]),1000,100,1000))

    out = np.array(out)
    np.save('100_tabu_1000.npy',out)
    print('max:{},min:{},mean:{}'.format(out.max(),out.min(),out.mean()))