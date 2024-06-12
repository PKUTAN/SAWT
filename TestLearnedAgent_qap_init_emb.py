import argparse
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from DataGenerator import QAPDataset
from tqdm import tqdm
from QAPEnvironment import QAPInstanceEnv, VecEnv
from ActorCriticNetwork_qap_mixEncoder_test import ActorCriticNetwork
import time

from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description='QAPNet')

# ----------------------------------- Data ---------------------------------- #
parser.add_argument('--test_size',
                    default=256, type=int, help='Test data size')
parser.add_argument('--test_from_data',
                    default=True,
                    action='store_true', help='Render')
parser.add_argument('--n_points',
                    type=int, default=20, help='Number of points in TSP')
# ---------------------------------- Train ---------------------------------- #
parser.add_argument('--n_steps',
                    default=2000,
                    type=int, help='Number of steps in each episode')
parser.add_argument('--render',
                    default=True,
                    action='store_true', help='Render')
# ----------------------------------- GPU ----------------------------------- #
parser.add_argument('--gpu',
                    default=True, action='store_true', help='Enable gpu')
# --------------------------------- Network --------------------------------- #
parser.add_argument('--input_dim',
                    type=int, default=2, help='Input size')
parser.add_argument('--embedding_dim',
                    type=int, default=64, help='Embedding size')
parser.add_argument('--hidden_dim',
                    type=int, default=64, help='Number of hidden units')
parser.add_argument('--n_rnn_layers',
                    type=int, default=1, help='Number of LSTM layers')
parser.add_argument('--n_actions',
                    type=int, default=2, help='Number of nodes to output')
parser.add_argument('--T',type=int,default=3)
parser.add_argument('--graph_ref',
                    default=False,
                    action='store_true',
                    help='Use message passing as reference')

# --------------------------------- Misc --------------------------------- #
parser.add_argument('--load_path', type=str,
    default='best_policy/policy-TSP20-epoch-189.pt')
parser.add_argument('--data_dir', type=str, default='data')

args = parser.parse_args()

if args.gpu and torch.cuda.is_available():
    USE_CUDA = True
    print('Using GPU, %i devices available.' % torch.cuda.device_count())
else:
    USE_CUDA = False

# loading the model from file
if args.load_path != '':
    print('  [*] Loading model from {}'.format(args.load_path))

    model = ActorCriticNetwork(args.input_dim,
                                args.embedding_dim,
                                args.hidden_dim,
                                args.n_points,
                                args.n_actions,
                                args.T,
                                args.graph_ref)
    checkpoint = torch.load(os.path.join(os.getcwd(), args.load_path))
    # import pdb; pdb.set_trace()
    policy = checkpoint['policy']
    model.load_state_dict(policy)

# Move model to the GPU
if USE_CUDA:
    model.cuda()
    device = 'cuda:0'

if args.n_points == 10:
    data_path = ['./synthetic_data/erdos10_0.7_F_test.npy','./synthetic_data/erdos10_0.7_positions_test.npy']
    args.test_size = 100
elif args.n_points == 20:
    data_path = ['./synthetic_data/erdos20_0.7_F_test.npy','./synthetic_data/erdos20_0.7_positions_test.npy']
    args.test_size = 256
elif args.n_points == 50:
    data_path = ['./synthetic_data/erdos50_0.7_F_test.npy','./synthetic_data/erdos50_0.7_positions_test.npy']
    args.test_size = 256
else:
    data_path = ['./synthetic_data/erdos100_0.7_F_test.npy','./synthetic_data/erdos100_0.7_positions_test.npy']
    args.test_size = 256

if args.test_from_data:
    test_data = QAPDataset(dataset_fname= data_path,
                           num_samples=args.test_size, seed=1234)


test_loader = DataLoader(test_data,
                         batch_size=args.test_size,
                         shuffle=False,
                         num_workers=6)



# run agent
# model = model.eval()
rewards = []
best_distances = []
step_best_distances = []
distances = []
initial_distances = []
distances_per_step = []
start_time = time.time()
pca = PCA(n_components=20)
for batch_idx, batch_sample in enumerate(test_loader):
    b_sample = batch_sample.clone().detach().numpy()
    sum_reward = 0
    env = VecEnv(QAPInstanceEnv,
                 b_sample.shape[0],
                 args.n_points)
    state, initial_distance, best_state,flow_g = env.reset(b_sample)
    
    
    flow_g = torch.from_numpy(flow_g).to(torch.float32).to(device)
    batch_size = flow_g.size(0)
    node_cnt = flow_g.size(1)
    embedding_dim = 128

    f_init_emb = torch.zeros(size=(batch_size, node_cnt, embedding_dim)).to(device)
    # shape: (batch, node, embedding)

    seed_cnt = 100
    rand = torch.rand(batch_size, seed_cnt)
    batch_rand_perm = rand.argsort(dim=1)
    rand_idx = batch_rand_perm[:, :node_cnt]

    b_idx = torch.arange(batch_size)[:, None].expand(batch_size, node_cnt)
    n_idx = torch.arange(node_cnt)[None, :].expand(batch_size, node_cnt)
    f_init_emb[b_idx, n_idx, rand_idx] = 1

    t = 0
    hidden = None
    pbar = tqdm(total=args.n_steps)
    while t < args.n_steps:
        # if args.render:
        #     env.render()
        state = torch.from_numpy(state).float()
        best_state = torch.from_numpy(best_state).float().cuda()
        if USE_CUDA:
            state = state.cuda()
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            probs, action, _, _, _ = model(state, best_state, flow_g, f_init_emb)
        action = action.cpu().numpy()
        state, reward, _, best_distance, distance, best_state = env.step(action)
        sum_reward += reward
        t += 1
        step_best_distances.append(np.mean(best_distance))
        distances_per_step.append(best_distance)
        pbar.update(1)
    pbar.close()
    rewards.append(sum_reward)
    best_distances.append(best_distance)
    distances.append(distance)
    initial_distances.append(initial_distance)
avg_reward = np.mean(rewards)
avg_best_distances = np.mean(best_distances)
avg_initial_distances = np.mean(initial_distances)
gap = ((avg_best_distances/np.mean(test_data.opt))-1)*100
# gap = ((avg_best_distances/np.mean(11.789159596642524))-1)*100



print('Initial Cost: {:.5f} Best Cost: {:.5f} Opt Cost: {:.5f} Gap: {:.2f} % Time: {}'.format(
    avg_initial_distances, avg_best_distances, np.round(np.mean(test_data.opt),2), gap , time.time() - start_time))

