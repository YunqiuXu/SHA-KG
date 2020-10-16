# In this version, I will perform KG2GRU, then GRU2KG
# KG2GRU: full KG + score as query, GRUs as regions --> 1 step
# GRU2KG: updated query as query, sub KGs as regions --> 1 step 
# Based on small model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import spacy
import numpy as np
from layers import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KGA2C(nn.Module):
    def __init__(self, params, 
                        templates, 
                        max_word_length, 
                        vocab_act,
                        vocab_act_rev, 
                        input_vocab_size, 
                        gat=True):
        super(KGA2C, self).__init__()
        self.templates = templates
        self.gat = gat
        assert(self.gat), "[Yunqiu Xu] assert self.gat to be True"
        self.max_word_length = max_word_length
        self.vocab = vocab_act
        self.vocab_rev = vocab_act_rev
        self.batch_size = params['batch_size']
        print("Action embedding")
        self.action_emb = nn.Embedding(len(vocab_act), params['embedding_size'])
        print("Action_drqa: ActionDrQA")
        self.action_drqa = ActionDrQA(input_vocab_size, 
                                        params['embedding_size'],
                                        params['batch_size'], 
                                        params['recurrent'])
        print("State network (GATs): StateNetwork")
        self.state_gat = StateNetwork(params['gat_emb_size'],
                                        vocab_act, 
                                        params['embedding_size'],
                                        params['dropout_ratio'], 
                                        params['tsv_file'])
        print("Template encoder: EncoderLSTM")
        self.template_enc = EncoderLSTM(input_vocab_size, 
                                        params['embedding_size'],
                                        int(params['hidden_size'] / 2),
                                        params['padding_idx'], 
                                        params['dropout_ratio'],
                                        self.action_emb)
        # SHA-related layers
        self.SAN1_Q_fc_0 = nn.Linear(60, 100)
        self.SAN1_A_fc_1 = nn.Linear(100, 100, bias=False)
        self.SAN1_Q_fc_1 = nn.Linear(100, 100)
        self.SAN1_P_fc_1 = nn.Linear(100, 100)
        self.SAN2_Q_fc_0 = nn.Linear(100, 50)
        self.SAN2_A_fc_1 = nn.Linear(50, 50, bias=False)
        self.SAN2_Q_fc_1 = nn.Linear(50, 50)
        self.SAN2_P_fc_1 = nn.Linear(50, 50)
        self.state_fc = nn.Linear(50, 100)
        
        print("Template decoder: DecoderRNN")
        self.decoder_template = DecoderRNN(params['hidden_size'], len(templates))
        print("Object decoder: ObjectDecoder")
        self.decoder_object = ObjectDecoder(50, 100, len(self.vocab.keys()),
                                            self.action_emb, params['graph_dropout'],
                                            params['k_object'])
        self.softmax = nn.Softmax(dim=1)
        self.critic = nn.Linear(100, 1)


    def forward(self, obs, scores, graph_rep, graph_rep1, graph_rep2,graph_rep3,graph_rep4, graphs):
        '''
        :param obs: The encoded ids for the textual observations (shape 4x300):
        The 4 components of an observation are: look - ob_l, inventory - ob_i, response - ob_r, and prev_action. 
        :type obs: ndarray
        :param scores:
        :param graph_rep: full kg
        :param graph_rep1-4: sub KGs 
        :param graphs: kg mask based on full kg
        '''
        batch = self.batch_size
        # Step 1: consider GRU outputs as regions
        o_t_l, o_t_i, o_t_o, o_t_p, h_t = self.action_drqa.forward(obs)
        # Step 2. score representation [batch, 10]
        src_t = []
        for scr in scores:
            if scr >= 0:
                cur_st = [0]
            else:
                cur_st = [1]
            cur_st.extend([int(c) for c in '{0:09b}'.format(abs(scr))])
            src_t.append(cur_st)
        src_t = torch.FloatTensor(src_t).cuda()
        # Step 3: graph representation 
        g_t_1, g_t_2, g_t_3, g_t_4, g_t_full = self.state_gat.forward(graph_rep1, 
                                                                        graph_rep2,
                                                                        graph_rep3,
                                                                        graph_rep4,
                                                                        graph_rep)

        # Step 4-1: high level, full KG + score as query, GRUs as regions
        h_regions_gru = torch.stack([o_t_l, o_t_i, o_t_o, o_t_p],dim=1)
        h_query_0_san1 = F.relu(self.SAN1_Q_fc_0(torch.cat((g_t_full, src_t), dim=1)))
        h_A_1_san1 = torch.tanh(self.SAN1_A_fc_1(h_regions_gru) + 
                                self.SAN1_Q_fc_1(h_query_0_san1).unsqueeze(1))
        att_1_san1 = F.softmax(self.SAN1_P_fc_1(h_A_1_san1), dim=1)
        h_query_1_san1 = h_query_0_san1 + (att_1_san1 * h_regions_gru).sum(1)
        
        # Step 4-2: low level, updated query as query, sub KGs as regions
        h_regions_kg = torch.stack([g_t_1, g_t_2, g_t_3, g_t_4], dim=1)
        h_query_0_san2 = F.relu(self.SAN2_Q_fc_0(h_query_1_san1))
        h_A_1_san2 = torch.tanh(self.SAN2_A_fc_1(h_regions_kg) + 
                                self.SAN2_Q_fc_1(h_query_0_san2).unsqueeze(1))
        att_1_san2 = F.softmax(self.SAN2_P_fc_1(h_A_1_san2), dim=1)
        h_query_1_san2 = h_query_0_san2 + (att_1_san2 * h_regions_kg).sum(1)
        state_emb = F.relu(self.state_fc(h_query_1_san2))
        
        # Step 5: compute value
        det_state_emb = state_emb.clone()
        value = self.critic(det_state_emb)

        # Step 6: decode
        # Get template
        decoder_t_output, decoder_t_hidden = self.decoder_template(state_emb, h_t)
        templ_enc_input = []
        decode_steps = []
        topi = self.softmax(decoder_t_output).multinomial(num_samples=1)
        # Select template for each agent (n_batch agents in parallel)
        for i in range(batch):
            templ, decode_step = self.get_action_rep(self.templates[topi[i].squeeze().detach().item()])
            templ_enc_input.append(templ)
            decode_steps.append(decode_step)
        # Select object based on template: seems that only "decoder_o_hidden_init0" is useful
        decoder_o_input, decoder_o_hidden_init0, decoder_o_enc_oinpts = self.template_enc.forward(torch.tensor(templ_enc_input).cuda().clone())
        # Here use graph_mask to filter objects not in KG
        decoder_o_output, decoded_o_words = self.decoder_object.forward(
                                                decoder_o_hidden_init0.cuda(), 
                                                decoder_t_hidden.squeeze_(0).cuda(), 
                                                self.vocab, 
                                                self.vocab_rev, 
                                                decode_steps, 
                                                graphs)
        return decoder_t_output, decoder_o_output, decoded_o_words, topi, value, decode_steps
        
    def get_action_rep(self, action):
        action = str(action)
        decode_step = action.count('OBJ')
        action = action.replace('OBJ', '')
        action_desc_num = 20 * [0]
        for i, token in enumerate(action.split()[:20]):
            short_tok = token[:self.max_word_length]
            action_desc_num[i] = self.vocab_rev[short_tok] if short_tok in self.vocab_rev else 0
        return action_desc_num, decode_step

    def clone_hidden(self):
        self.action_drqa.clone_hidden()

    def restore_hidden(self):
        self.action_drqa.restore_hidden()

    def reset_hidden(self, done_mask_tt):
        self.action_drqa.reset_hidden(done_mask_tt)


class StateNetwork(nn.Module):
    def __init__(self, gat_emb_size, vocab, embedding_size, dropout_ratio, tsv_file, embeddings=None):
        super(StateNetwork, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.embedding_size = 25                                    # [Yunqiu Xu] hard code as 25
        self.dropout_ratio = dropout_ratio
        self.gat_emb_size = gat_emb_size                            
        self.gat1 = GAT(gat_emb_size, 3, dropout_ratio, 0.2, 1)    
        self.gat2 = GAT(gat_emb_size, 3, dropout_ratio, 0.2, 1)
        self.gat3 = GAT(gat_emb_size, 3, dropout_ratio, 0.2, 1)
        self.gat4 = GAT(gat_emb_size, 3, dropout_ratio, 0.2, 1)
        self.vocab_kge = self.load_vocab_kge(tsv_file)
        self.state_ent_emb = nn.Embedding.from_pretrained(torch.zeros((len(self.vocab_kge), self.embedding_size)), freeze=False)  # Shared
        self.fc1 = nn.Linear(self.state_ent_emb.weight.size()[0] * 3 * 1, 50) 
        self.fc2 = nn.Linear(self.state_ent_emb.weight.size()[0] * 3 * 1, 50) 
        self.fc3 = nn.Linear(self.state_ent_emb.weight.size()[0] * 3 * 1, 50) 
        self.fc4 = nn.Linear(self.state_ent_emb.weight.size()[0] * 3 * 1, 50)
        # for fullKG
        self.gat_full = GAT(gat_emb_size, 3, dropout_ratio, 0.2, 1)
        self.fc_full = nn.Linear(self.state_ent_emb.weight.size()[0] * 3 * 1, 50)

    def load_vocab_kge(self, tsv_file):
        ent = {}
        with open(tsv_file, 'r') as f:
            for line in f:
                e, eid = line.split('\t')
                ent[int(eid.strip())] = e.strip()
        return ent
    
    def forward(self, graph_rep1, graph_rep2,graph_rep3,graph_rep4, graph_rep_full):
        out1, out2, out3, out4, out_full = [], [], [], [], []
        batch_size = len(graph_rep1)
        for i in range(batch_size):
            # Get adjacent matrix
            adj1 =  torch.IntTensor(graph_rep1[i][1]).cuda()
            adj2 =  torch.IntTensor(graph_rep2[i][1]).cuda()
            adj3 =  torch.IntTensor(graph_rep3[i][1]).cuda()
            adj4 =  torch.IntTensor(graph_rep4[i][1]).cuda()
            adj_full = torch.IntTensor(graph_rep_full[i][1]).cuda()
            # Compute gat
            out1.append(self.gat1(self.state_ent_emb.weight, adj1).view(-1).unsqueeze_(0))
            out2.append(self.gat2(self.state_ent_emb.weight, adj2).view(-1).unsqueeze_(0))
            out3.append(self.gat3(self.state_ent_emb.weight, adj3).view(-1).unsqueeze_(0))
            out4.append(self.gat4(self.state_ent_emb.weight, adj4).view(-1).unsqueeze_(0)) 
            out_full.append(self.gat_full(self.state_ent_emb.weight, adj_full).view(-1).unsqueeze_(0)) 
        # compute as batch
        g_t_1 = self.fc1(torch.cat(out1))
        g_t_2 = self.fc2(torch.cat(out2))
        g_t_3 = self.fc3(torch.cat(out3))
        g_t_4 = self.fc4(torch.cat(out4))
        g_t_full = self.fc_full(torch.cat(out_full))
        return g_t_1, g_t_2, g_t_3, g_t_4, g_t_full
        

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout)
        # [n_node, 3*1] --> 3 is out_feat, 1 is nheads
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout)
        return x


class ActionDrQA(nn.Module):
    def __init__(self, vocab_size, embedding_size, batch_size, recurrent=True):
        super(ActionDrQA, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.recurrent = recurrent
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.enc_look = PackedEncoderRNN(self.vocab_size, 100)
        self.h_look = self.enc_look.initHidden(self.batch_size)
        self.enc_inv = PackedEncoderRNN(self.vocab_size, 100)
        self.h_inv = self.enc_inv.initHidden(self.batch_size)
        self.enc_ob = PackedEncoderRNN(self.vocab_size, 100)
        self.h_ob = self.enc_ob.initHidden(self.batch_size)
        self.enc_preva = PackedEncoderRNN(self.vocab_size, 100)
        self.h_preva = self.enc_preva.initHidden(self.batch_size)
        self.fch = nn.Linear(100 * 4, 100)

    def reset_hidden(self, done_mask_tt):
        '''
        Reset the hidden state of episodes that are done.
        :param done_mask_tt: Mask indicating which parts of hidden state should be reset.
        :type done_mask_tt: Tensor of shape [BatchSize x 1]
        '''
        self.h_look = done_mask_tt.detach() * self.h_look
        self.h_inv = done_mask_tt.detach() * self.h_inv
        self.h_ob = done_mask_tt.detach() * self.h_ob
        self.h_preva = done_mask_tt.detach() * self.h_preva

    def clone_hidden(self):
        ''' Makes a clone of hidden state. '''
        self.tmp_look = self.h_look.clone().detach()
        self.tmp_inv = self.h_inv.clone().detach()
        self.h_ob = self.h_ob.clone().detach()
        self.h_preva = self.h_preva.clone().detach()

    def restore_hidden(self):
        '''Restores hidden state from clone made by clone_hidden.'''
        self.h_look = self.tmp_look
        self.h_inv = self.tmp_inv
        self.h_ob = self.h_ob
        self.h_preva = self.h_preva

    def forward(self, obs):
        '''
        :param obs: Encoded observation tokens.
        :type obs: np.ndarray of shape (Batch_Size x 4 x 300)
        :output: 4 * 100 -> [batch, 400] -> [batch, 100]
        '''
        x_l, h_l = self.enc_look(torch.LongTensor(obs[:,0,:]).cuda(), self.h_look)
        x_i, h_i = self.enc_inv(torch.LongTensor(obs[:,1,:]).cuda(), self.h_inv)
        x_o, h_o = self.enc_ob(torch.LongTensor(obs[:,2,:]).cuda(), self.h_ob)
        x_p, h_p = self.enc_preva(torch.LongTensor(obs[:,3,:]).cuda(), self.h_preva)
        if self.recurrent:
            self.h_look = h_l
            self.h_ob = h_o
            self.h_preva = h_p
            self.h_inv = h_i
        h = F.relu(self.fch(torch.cat((h_l, h_i, h_o, h_p), dim=2)))
        return x_l, x_i, x_o, x_p, h


class ObjectDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embeddings, graph_dropout, k):
        super(ObjectDecoder, self).__init__()
        self.k = k
        self.decoder = DecoderRNN2(hidden_size, output_size, embeddings, graph_dropout)
        self.max_decode_steps = 2
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, input_hidden, vocab, vocab_rev, decode_steps_t, graphs):
        all_outputs, all_words = [], []
        decoder_input = torch.tensor([vocab_rev['<s>']] * input.size(0)).cuda()
        decoder_hidden = input_hidden.unsqueeze(0)
        torch.set_printoptions(profile="full")
        # decode for 2 steps
        for di in range(self.max_decode_steps):
            ret_decoder_output, decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, input, graphs)
            if self.k == 1:
                all_outputs.append(ret_decoder_output)
                dec_objs = []
                # batch
                for i in range(decoder_output.shape[0]):
                    dec_probs = F.softmax(ret_decoder_output[i][graphs[i]], dim=0)
                    idx = dec_probs.multinomial(1)
                    graph_list = graphs[i].nonzero().cpu().numpy().flatten().tolist()
                    assert len(graph_list) == dec_probs.numel()
                    dec_objs.append(graph_list[idx])
                topi = torch.LongTensor(dec_objs).cuda()
                decoder_input = topi.squeeze().detach()
                all_words.append(topi)
            else:
                all_outputs.append(decoder_output)
                topv, topi = decoder_output.topk(self.k) # select top-k
                topv = self.softmax(topv)
                topv = topv.cpu().numpy()
                topi = topi.cpu().numpy()
                cur_objs = []
                # batch
                for i in range(graphs.size(0)):
                    cur_obj = np.random.choice(topi[i].reshape(-1), p=topv[i].reshape(-1))
                    cur_objs.append(cur_obj)
                decoder_input = torch.LongTensor(cur_objs).cuda()
                all_words.append(decoder_input)
        return torch.stack(all_outputs), torch.stack(all_words)

    def flatten_parameters(self):
        self.encoder.gru.flatten_parameters()
        self.decoder.gru.flatten_parameters()
