import networkx as nx
import numpy as np
import openie
from fuzzywuzzy import fuzz
from jericho.util import clean


class StateAction(object):
    def __init__(self, spm, vocab, vocab_rev, tsv_file, max_word_len):
        self.graph_state = nx.DiGraph()
        self.max_word_len = max_word_len
        self.graph_state_rep = []
        self.visible_state = ""
        self.drqa_input = ""
        self.vis_pruned_actions = []
        self.pruned_actions_rep = []
        self.sp = spm
        self.vocab_act = vocab
        self.vocab_act_rev = vocab_rev
        self.vocab_kge = self.load_vocab_kge(tsv_file)
        self.adj_matrix = np.zeros((len(self.vocab_kge['entity']), len(self.vocab_kge['entity'])))
        self.room = ""

        self.graph_state_1_connectivity = nx.DiGraph()  # Need to track room connectivity          
        self.graph_state_2_roomitem = None
        self.graph_state_3_youritem = None  
        self.graph_state_4_otherroom = None  
        self.graph_state_5_mask = None
        self.graph_state_rep_1_connectivity = []        
        self.graph_state_rep_2_roomitem = []
        self.graph_state_rep_3_youritem = []
        self.graph_state_rep_4_otherroom = []  
        self.graph_state_rep_5_mask = [] 
        self.adj_matrix_1_connectivity = np.zeros((len(self.vocab_kge['entity']), len(self.vocab_kge['entity'])))
        self.adj_matrix_2_roomitem = np.zeros((len(self.vocab_kge['entity']), len(self.vocab_kge['entity'])))
        self.adj_matrix_3_youritem = np.zeros((len(self.vocab_kge['entity']), len(self.vocab_kge['entity'])))
        self.adj_matrix_4_otherroom = np.zeros((len(self.vocab_kge['entity']), len(self.vocab_kge['entity'])))
        self.adj_matrix_5_mask = np.zeros((len(self.vocab_kge['entity']), len(self.vocab_kge['entity'])))
    
    def visualize(self, graph_to_vis=None):
        assert(graph_to_vis is not None), "visualize(): the graph should not be None!"
        pos = nx.spring_layout(graph_to_vis)
        edge_labels = {e: graph_to_vis.edges[e]['rel'] for e in graph_to_vis.edges}
        print(edge_labels)
        nx.draw_networkx_edge_labels(graph_to_vis, pos, edge_labels)
        nx.draw(graph_to_vis, pos=pos, with_labels=True, node_size=200, font_size=10)

    def load_vocab_kge(self, tsv_file):
        ent = {}
        with open(tsv_file, 'r') as f:
            for line in f:
                e, eid = line.split('\t')
                ent[e.strip()] = int(eid.strip())
        rel = {}
        with open(tsv_file, 'r') as f:
            for line in f:
                r, rid = line.split('\t')
                rel[r.strip()] = int(rid.strip())
        return {'entity': ent, 'relation': rel}


    def update_state(self, visible_state, inventory_state, objs, prev_action=None, cache=None):
        # Step 1: Build a copy of past KG (full)
        graph_copy = self.graph_state.copy()
        prev_room = self.room
        prev_room_subgraph = None
        con_cs = [graph_copy.subgraph(c) for c in nx.weakly_connected_components(graph_copy)]
        for con_c in con_cs:
            for node in con_c.nodes:
                node = set(str(node).split())
                if set(prev_room.split()).issubset(node):
                    prev_room_subgraph = nx.induced_subgraph(graph_copy, con_c.nodes)
        # Step 2: Bemove old ones with "you" --> past KG without "you"
        for edge in self.graph_state.edges:
            if 'you' in edge[0]:
                graph_copy.remove_edge(*edge)
        self.graph_state = graph_copy
        # Keep room connectivity only, remove "you"
        # <you, in, room>, <room, connect, room> --> <room, connect, room>
        graph_copy_1_connectivity = self.graph_state_1_connectivity.copy()
        for edge in self.graph_state_1_connectivity.edges:
            if 'you' in edge[0]:
                graph_copy_1_connectivity.remove_edge(*edge)
        self.graph_state_1_connectivity = graph_copy_1_connectivity
        # Step 3: Reinitialize sub-KG
        self.graph_state_2_roomitem = nx.DiGraph()   # re-init
        self.graph_state_3_youritem = nx.DiGraph()  # re-init
        self.graph_state_4_otherroom = graph_copy.copy() # Just past information
        # Preprocess visible state --> get sents
        visible_state = visible_state.split('\n')
        room = visible_state[0]
        visible_state = clean(' '.join(visible_state[1:]))
        self.visible_state = str(visible_state)
        if cache is None:
            sents = openie.call_stanford_openie(self.visible_state)['sentences']
        else:
            sents = cache
        if sents == "":
            return []
        dirs = ['north', 'south', 'east', 'west', 'southeast', 'southwest', 'northeast', 'northwest', 'up', 'down']
        in_aliases = ['are in', 'are facing', 'are standing', 'are behind', 'are above', 'are below', 'are in front']
        # Update graph, "rules" are new triples to be added
        # Add two rule lists for "you" and "woyou"
        rules_1_connectivity = [] # <you,in>, <room,connect>
        rules_2_roomitem = []     # <you,in>, <room,have>
        rules_3_youritem = []     # <you,have>
        rules = []
        in_rl = []
        in_flag = False
        for i, ov in enumerate(sents):
            sent = ' '.join([a['word'] for a in ov['tokens']])
            triple = ov['openie']
            # 1.1 -> check directions
            # direction rules: <room, has, exit to direction>
            for d in dirs:
                if d in sent and i != 0:
                    rules.append((room, 'has', 'exit to ' + d))
                    rules_1_connectivity.append((room, 'has', 'exit to ' + d))
            # 1.2 -> check OpenIE triples
            for tr in triple:
                h, r, t = tr['subject'].lower(), tr['relation'].lower(), tr['object'].lower()
                # case 1: "you", "in"
                if h == 'you':
                    for rp in in_aliases:
                        if fuzz.token_set_ratio(r, rp) > 80:
                            r = "in"
                            in_rl.append((h, r, t)) # <you, in, >
                            in_flag = True
                            break
                # case 2: should not be "it"
                if h == 'it':
                    break
                # case 3: other triples
                if not in_flag:
                    rules.append((h, r, t))
                    rules_2_roomitem.append((h, r, t))
        # 1.3 "you are in" cases
        if in_flag:
            cur_t = in_rl[0]
            for h, r, t in in_rl:
                if set(cur_t[2].split()).issubset(set(t.split())):
                    cur_t = h, r, t
            rules.append(cur_t)
            rules_1_connectivity.append(cur_t)
            rules_2_roomitem.append(cur_t)
            room = cur_t[2]
            self.room=room
        # 1.4 inventory: <you, have, ...>
        try:
            items = inventory_state.split(':')[1].split('\n')[1:]
            for item in items:
                rules.append(('you', 'have', str(' ' .join(item.split()[1:]))))
                rules_3_youritem.append(('you', 'have', str(' ' .join(item.split()[1:])))) # [20200420] 3
        except:
            pass
        # 1.5 room connectivity: <room, dir, room>
        if prev_action is not None:
            for d in dirs:
                if d in prev_action and self.room != "":
                    rules.append((prev_room, d + ' of', room))
                    rules_1_connectivity.append((prev_room, d + ' of', room))
                    if prev_room_subgraph is not None:
                        for ed in prev_room_subgraph.edges:
                            rules.append((ed[0], "prev_graph_relations", ed[1]))
                    break
        # 1.6 room item: <item,in,room>
        # If the action is "drop" --> something will be in this room
        # Therefore binary exploration bonus should not be used!
        for o in objs:
            rules.append((str(o), 'in', room))
            rules_2_roomitem.append((str(o), 'in', room))

        # add edges: if this edge already exists, adding will not show difference
        add_rules = rules
        for rule in add_rules:
            u = '_'.join(str(rule[0]).split())
            v = '_'.join(str(rule[2]).split())
            if u in self.vocab_kge['entity'].keys() and v in self.vocab_kge['entity'].keys():
                if u != 'it' and v != 'it':
                    self.graph_state.add_edge(rule[0], rule[2], rel=rule[1])
        
        # build graph_state_1_connectivity
        for rule in rules_1_connectivity:
            u = '_'.join(str(rule[0]).split())
            v = '_'.join(str(rule[2]).split())
            if u in self.vocab_kge['entity'].keys() and v in self.vocab_kge['entity'].keys():
                if u != 'it' and v != 'it':
                    self.graph_state_1_connectivity.add_edge(rule[0], rule[2], rel=rule[1])       
        # build graph_state_5_mask
        self.graph_state_5_mask = self.graph_state_1_connectivity.copy()
        # build graph_state_2_roomitem (and graph_state_5_mask)
        for rule in rules_2_roomitem:
            u = '_'.join(str(rule[0]).split())
            v = '_'.join(str(rule[2]).split())
            if u in self.vocab_kge['entity'].keys() and v in self.vocab_kge['entity'].keys():
                if u != 'it' and v != 'it':
                    self.graph_state_2_roomitem.add_edge(rule[0], rule[2], rel=rule[1])     
                    self.graph_state_5_mask.add_edge(rule[0], rule[2], rel=rule[1])     
        # build graph_state_3_youritem (and graph_state_5_mask)
        for rule in rules_3_youritem:
            u = '_'.join(str(rule[0]).split())
            v = '_'.join(str(rule[2]).split())
            if u in self.vocab_kge['entity'].keys() and v in self.vocab_kge['entity'].keys():
                if u != 'it' and v != 'it':
                    self.graph_state_3_youritem.add_edge(rule[0], rule[2], rel=rule[1])    
                    self.graph_state_5_mask.add_edge(rule[0], rule[2], rel=rule[1])     
        return add_rules, sents


    def get_state_rep_kge(self):
        ret = []
        self.adj_matrix = np.zeros((len(self.vocab_kge['entity']), len(self.vocab_kge['entity'])))
        for u, v in self.graph_state.edges:
            u = '_'.join(str(u).split())
            v = '_'.join(str(v).split())
            if u not in self.vocab_kge['entity'].keys() or v not in self.vocab_kge['entity'].keys():
                break
            u_idx = self.vocab_kge['entity'][u]
            v_idx = self.vocab_kge['entity'][v]
            self.adj_matrix[u_idx][v_idx] = 1
            ret.append(self.vocab_kge['entity'][u])
            ret.append(self.vocab_kge['entity'][v])
        return list(set(ret))
        

    def get_state_rep_kge_1(self):
        ret = []
        self.adj_matrix_1_connectivity = np.zeros((len(self.vocab_kge['entity']), len(self.vocab_kge['entity'])))
        for u, v in self.graph_state_1_connectivity.edges:
            u = '_'.join(str(u).split())
            v = '_'.join(str(v).split())
            if u not in self.vocab_kge['entity'].keys() or v not in self.vocab_kge['entity'].keys():
                break
            u_idx = self.vocab_kge['entity'][u]
            v_idx = self.vocab_kge['entity'][v]
            self.adj_matrix_1_connectivity[u_idx][v_idx] = 1
            ret.append(self.vocab_kge['entity'][u])
            ret.append(self.vocab_kge['entity'][v])
        return list(set(ret))
        

    def get_state_rep_kge_2(self):
        ret = []
        self.adj_matrix_2_roomitem = np.zeros((len(self.vocab_kge['entity']), len(self.vocab_kge['entity'])))
        for u, v in self.graph_state_2_roomitem.edges:
            u = '_'.join(str(u).split())
            v = '_'.join(str(v).split())
            if u not in self.vocab_kge['entity'].keys() or v not in self.vocab_kge['entity'].keys():
                break
            u_idx = self.vocab_kge['entity'][u]
            v_idx = self.vocab_kge['entity'][v]
            self.adj_matrix_2_roomitem[u_idx][v_idx] = 1
            ret.append(self.vocab_kge['entity'][u])
            ret.append(self.vocab_kge['entity'][v])
        return list(set(ret))
        

    def get_state_rep_kge_3(self):
        ret = []
        self.adj_matrix_3_youritem = np.zeros((len(self.vocab_kge['entity']), len(self.vocab_kge['entity'])))
        for u, v in self.graph_state_3_youritem.edges:
            u = '_'.join(str(u).split())
            v = '_'.join(str(v).split())
            if u not in self.vocab_kge['entity'].keys() or v not in self.vocab_kge['entity'].keys():
                break
            u_idx = self.vocab_kge['entity'][u]
            v_idx = self.vocab_kge['entity'][v]
            self.adj_matrix_3_youritem[u_idx][v_idx] = 1
            ret.append(self.vocab_kge['entity'][u])
            ret.append(self.vocab_kge['entity'][v])
        return list(set(ret))
        
        
    def get_state_rep_kge_4(self):
        ret = []
        self.adj_matrix_4_otherroom = np.zeros((len(self.vocab_kge['entity']), len(self.vocab_kge['entity'])))
        for u, v in self.graph_state_4_otherroom.edges:
            u = '_'.join(str(u).split())
            v = '_'.join(str(v).split())
            if u not in self.vocab_kge['entity'].keys() or v not in self.vocab_kge['entity'].keys():
                break
            u_idx = self.vocab_kge['entity'][u]
            v_idx = self.vocab_kge['entity'][v]
            self.adj_matrix_4_otherroom[u_idx][v_idx] = 1
            ret.append(self.vocab_kge['entity'][u])
            ret.append(self.vocab_kge['entity'][v])
        return list(set(ret))
        

    def get_state_rep_kge_5(self):
        ret = []
        self.adj_matrix_5_mask = np.zeros((len(self.vocab_kge['entity']), len(self.vocab_kge['entity'])))
        for u, v in self.graph_state_5_mask.edges:
            u = '_'.join(str(u).split())
            v = '_'.join(str(v).split())
            if u not in self.vocab_kge['entity'].keys() or v not in self.vocab_kge['entity'].keys():
                break
            u_idx = self.vocab_kge['entity'][u]
            v_idx = self.vocab_kge['entity'][v]
            self.adj_matrix_5_mask[u_idx][v_idx] = 1
            ret.append(self.vocab_kge['entity'][u])
            ret.append(self.vocab_kge['entity'][v])
        return list(set(ret))
    

    def get_obs_rep(self, *args):
        ret = [self.get_visible_state_rep_drqa(ob) for ob in args]
        return pad_sequences(ret, maxlen=300)

    def get_visible_state_rep_drqa(self, state_description):
        remove = ['=', '-', '\'', ':', '[', ']', 'eos', 'EOS', 'SOS', 'UNK', 'unk', 'sos', '<', '>']
        for rm in remove:
            state_description = state_description.replace(rm, '')
        return self.sp.encode_as_ids(state_description)

    def get_action_rep_drqa(self, action):
        action_desc_num = 20 * [0]
        action = str(action)
        for i, token in enumerate(action.split()[:20]):
            short_tok = token[:self.max_word_len]
            action_desc_num[i] = self.vocab_act_rev[short_tok] if short_tok in self.vocab_act_rev else 0
        return action_desc_num


    def step(self, visible_state, inventory_state, objs, prev_action=None, cache=None, gat=True):
        # Update graph_states
        ret, ret_cache = self.update_state(visible_state, inventory_state, objs, prev_action, cache)
        self.pruned_actions_rep = [self.get_action_rep_drqa(a) for a in self.vis_pruned_actions]
        inter = self.visible_state
        self.drqa_input = self.get_visible_state_rep_drqa(inter)
        # Get graph_state_reps
        self.graph_state_rep = self.get_state_rep_kge(), self.adj_matrix
        self.graph_state_rep_1_connectivity = self.get_state_rep_kge_1(), self.adj_matrix_1_connectivity
        self.graph_state_rep_2_roomitem = self.get_state_rep_kge_2(), self.adj_matrix_2_roomitem
        self.graph_state_rep_3_youritem = self.get_state_rep_kge_3(), self.adj_matrix_3_youritem
        self.graph_state_rep_4_otherroom = self.get_state_rep_kge_4(), self.adj_matrix_4_otherroom
        self.graph_state_rep_5_mask = self.get_state_rep_kge_5(), self.adj_matrix_5_mask
        return ret, ret_cache



def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    '''
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x


