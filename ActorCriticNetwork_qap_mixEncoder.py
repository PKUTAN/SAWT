import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch import tanh
import math
import utils

from transformer.Layers import EncoderLayer, MixEncoderLayer,CrossEncoderLayer,Mix_flow_EncoderLayer

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Affinity(nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    M = X * A * Y^T
    Parameter: scale of weight d
    Input: feature X, Y
    Output: affinity matrix M
    """
    def __init__(self, d):
        super(Affinity, self).__init__()
        self.d = d
        self.A = Parameter(torch.Tensor(self.d, self.d))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.d)
        self.A.data.uniform_(-stdv, stdv)
        self.A.data += torch.eye(self.d)

    def forward(self, X, Y):
        assert X.shape[2] == Y.shape[2] == self.d
        M = torch.matmul(X, self.A)
        #M = torch.matmul(X, (self.A + self.A.transpose(0, 1)) / 2)
        M = torch.matmul(M, Y.transpose(1, 2))
        return M


class Encoder(nn.Module):
    """
    Encoder of TSP-Net
    """
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 num_nodes,
                 sat_layers,
                ):
        """
        Initialise Encoder
        :param int input_dim: Number of input dimensions
        :param int embedding_dim: Number of embbeding dimensions
        :param int hidden_dim: Number of hidden units of the RNN
        :param int num_nodes: Problem size of QAP
        """
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.sat_layers = sat_layers

        self.embedding_f = nn.Linear(128,embedding_dim)
        
        self.embedding_d = nn.Linear(2, embedding_dim)

        self.f_encoder_1 = MixEncoderLayer(hidden_dim,hidden_dim,8,hidden_dim//8,hidden_dim//8)
        self.f_encoder_2 = MixEncoderLayer(hidden_dim,hidden_dim,8,hidden_dim//8,hidden_dim//8)

        self.d_embedding = nn.Linear(embedding_dim,hidden_dim)
        self.d_embedding1 = nn.Linear(hidden_dim,hidden_dim)
        self.d_embedding2 = nn.Linear(hidden_dim,hidden_dim)

        self.fuse_embedding = nn.Linear(2*hidden_dim,hidden_dim)

        self.sat_transformers = nn.ModuleList()

        for i in range(sat_layers):
            self.sat_transformers.append(MixEncoderLayer(hidden_dim,hidden_dim,8,hidden_dim//8,hidden_dim//8))
        

    def forward(self, input, flow_g,f_init_emb):
        """
        Encoder: Forward-pass

        :param Tensor input: Graph inputs (bs, n_nodes, 2+n_nodes)
        :param Tensor hidden: hidden vectors passed as inputs from t-1
        """

        batch_size = input.size(0)

        location = input[:,:,:2]
        permutation = input[:,:,2:]

        # import pdb;pdb.set_trace()
        location = torch.bmm(permutation,location)

        edges = utils.batch_pair_squared_dist(location, location)
        # edges = utils.pre_process_adj(edges)
        edges.requires_grad = False

        # embedding of flow matrix
        # import pdb; pdb.set_trace()
        flow_embedded_input = self.embedding_f(f_init_emb)
        d_embedding_input = self.embedding_d(location)

        # 
        
        d_embedding = d_embedding_input + F.relu(torch.bmm(edges,self.d_embedding(d_embedding_input)))
        d_embedding = d_embedding + F.relu(torch.bmm(edges,self.d_embedding1(d_embedding)))
        d_embedding = d_embedding + F.relu(torch.bmm(edges,self.d_embedding2(d_embedding)))

        flow_g_embedding = self.f_encoder_1(flow_embedded_input,flow_g)
        flow_g_embedding = self.f_encoder_2(flow_g_embedding,flow_g)

        flow_g_embedding = self.fuse_embedding(torch.cat((flow_g_embedding,d_embedding),dim=-1))
       
        dis = utils.batch_pair_squared_dist_no_norm(location,location)
        flow_g = dis*flow_g
        
        for i in range(self.sat_layers):
            flow_g_embedding = self.sat_transformers[i](flow_g_embedding,flow_g)


        return flow_g_embedding


class Decoder(nn.Module):
    """
    Decoder
    """

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_actions):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        
        self.W1 = nn.Linear(hidden_dim,hidden_dim)
        self.W2 = nn.Linear(hidden_dim,hidden_dim)

        self.select_act_1 = nn.Sequential(nn.Linear(2*hidden_dim,hidden_dim),
                                          nn.Tanh(),
                                          nn.Linear(hidden_dim,hidden_dim),
                                          nn.Tanh(),
                                          nn.Linear(hidden_dim,1))
        
        self.select_act_2 = nn.Sequential(nn.Linear(3*hidden_dim,hidden_dim),
                                          nn.Tanh(),
                                          nn.Linear(hidden_dim,hidden_dim),
                                          nn.Tanh(),
                                          nn.Linear(hidden_dim,1))

        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

        self._inf = float('-inf')

    def forward(self, star, f_emb, actions=None):

        batch_size = f_emb.size(0)
        n_nodes = f_emb.size(1)

        # mask: (batch, n_nodes) filled with 1's
        mask = self.mask.repeat((batch_size, n_nodes))

        # runner: (input_lenght) tensor filled with 0's
        runner = self.runner.repeat(n_nodes)
        # runner: (input_lenght) tensor from {0 to input_lenght-1}
        for i in range(n_nodes):
            runner.data[i] = i
        # (batch, seq_len) filled with {0,...,seq-len-1}
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        # lists for the outputs
        probs = []
        pointers = []
        log_probs_pts = []
        entropy = []
        
        
        for i in range(self.n_actions):
            if i == 0:
                # mask[:,-1] = 0

                input = torch.cat((f_emb,star.unsqueeze(1).expand(batch_size,n_nodes,-1)),dim=-1)

                act_score = self.select_act_1(input).squeeze(-1)
                act_score = act_score.masked_fill_(torch.eq(mask,0),self._inf)

                prob = F.softmax(act_score,dim=-1)

                masked_prob = prob*mask
            if i == 1:
                # mask[:,-1] = 1.

                input = torch.cat((f_emb,act_1_inp.unsqueeze(1).expand(batch_size,n_nodes,-1),star.unsqueeze(1).expand(batch_size,n_nodes,-1)),dim= -1)
                
                act_score = self.select_act_2(input).squeeze(-1)

                act_score = act_score.masked_fill_(torch.eq(mask,0),self._inf)
                prob = F.softmax(act_score,dim=-1)

                masked_prob = prob*mask

            
            
            c = torch.distributions.Categorical(masked_prob)
            if actions is None:
                indices = c.sample()
                log_probs_idx = c.log_prob(indices)
                dist_entropy = c.entropy()
            else:
                indices = actions[:, i]
                log_probs_idx = c.log_prob(indices)
                dist_entropy = c.entropy()

            repeat_indices = indices.unsqueeze(1).expand(-1, n_nodes)
            # 1-pointers probs indices i.e. if idx= 4 and len = 5
            # one_hot_pointers[0] = [0, 0 , 0 , 0 , 1]
            # one_hot_pointers: (batch_size, seq_len)
            one_pointers = (runner == repeat_indices).float()
            lower_pointers = (runner <= repeat_indices).float()

            # Update mask to ignore seen indices
            # (mask gets updated from 1 --> 0 for seen indices)
            # mask: (batch_size, seq_len)
            mask = mask * (1 - one_pointers)

            # embbeding mask: boolean (batch size, seq_len, embbeding_dim)
            # True for the pointed input False otherwise
            one_pointers = one_pointers.unsqueeze(2)
            dec_input_mask = one_pointers.expand(-1,
                                                 -1,
                                                 self.hidden_dim).bool()
            masked_dec_input = f_emb[dec_input_mask.data]
            act_1_inp = masked_dec_input.view(batch_size, self.hidden_dim)

            # outputs: list of softmax outputs of size (1, batch_size, seq_len)
            probs.append(prob.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))
            log_probs_pts.append(log_probs_idx.unsqueeze(1))
            entropy.append(dist_entropy.unsqueeze(1))

        probs = torch.cat(probs).permute(1, 0, 2)

        # pointers: index outputs (batch_size, n_actions)
        pointers = torch.cat(pointers, 1)
        log_probs_pts = torch.cat(log_probs_pts, 1)
        entropies = torch.cat(entropy, 1)

        return probs, pointers, log_probs_pts, entropies


class ActorCriticNetwork(nn.Module):

    """
    ActorCritic-Net
    """

    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 n_nodes,
                 n_actions,
                 sat_layers,
                 graph_ref=False):
        """
        :param int embedding_dim: Number of embbeding dimensions
        :param int hidden_dim: Encoder/Decoder hidden units
        :param int lstm_layers: Number of LSTM layers
        :param bool bidir: Bidirectional
        :param bool batch_first: Batch first in the LSTM
        """

        super(ActorCriticNetwork, self).__init__()

        self.encoder = Encoder(input_dim,
                               embedding_dim,
                               hidden_dim,
                               n_nodes,
                               sat_layers
                               )

        self.encoder_star = Encoder(input_dim,
                                    embedding_dim,
                                    hidden_dim,
                                    n_nodes,
                                    sat_layers
                                    )
        
        self.max_star = nn.Linear(hidden_dim,hidden_dim)

        self.decoder_a = Decoder(embedding_dim,
                                 hidden_dim,
                                 n_actions)

        self.W_star = nn.Linear(hidden_dim, hidden_dim)

        self.decoder_c = nn.Sequential(
                         nn.Linear(hidden_dim, hidden_dim),
                         nn.ReLU(),
                         nn.Linear(hidden_dim, 1))
        self.graph_ref = graph_ref

    def forward(self, inputs, inputs_star, flow_g, f_init_emb ,actions=None):
        f_emb_star = self.encoder_star(inputs_star,flow_g,f_init_emb)

        f_emb = self.encoder(inputs,flow_g,f_init_emb)

        star_emb = self.max_star(torch.max(f_emb_star,dim=1)[0])

        probs, pts, log_probs_pts, entropies = self.decoder_a(star_emb,
                                                              f_emb,
                                                              actions)
        # import pdb;pdb.set_trace()
        v_g = torch.mean(f_emb, dim=1).squeeze(1)
        h_v = self.W_star(star_emb)
        v = self.decoder_c(v_g + h_v)

        return probs, pts, log_probs_pts, v, entropies


if __name__ == '__main__':
    weight = torch.rand((2,20,20)).to(device)
    weight = utils.pre_process_adj(weight)
    adj = (weight != 0).float().to(device)
    loc = torch.rand((2,20,2)).to(device)
    x = torch.eye(20).unsqueeze(0).repeat(2,1,1).to(device)
    input = torch.cat((loc,x),dim=-1)

    policy = ActorCriticNetwork(2,64,64,20,2,5).to(device)

    probs,pts,logs,v,en = policy(input,input,(adj,weight))

    import pdb;pdb.set_trace()