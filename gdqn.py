import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import os
from os.path import basename, splitext
import numpy as np
import time
import sentencepiece as spm
from jericho import *
from jericho.template_action_generator import TemplateActionGenerator
from jericho.util import unabbreviate, clean
import jericho.defines
# This work
from models import KGA2C
from env import load_vocab, KGA2CEnv
from vec_env import VecEnv
import logger


device = torch.device("cuda")


def configure_logger(log_dir):
    print("save log at: {}".format(log_dir))
    logger.configure(log_dir, format_strs=['log'])
    global tb
    tb = logger.Logger(log_dir, [logger.make_output_format('tensorboard', log_dir),
                                 logger.make_output_format('csv', log_dir),
                                 logger.make_output_format('stdout', log_dir)])
    global log
    log = logger.log


class KGA2CTrainer(object):
    '''
    KGA2C main class.
    '''
    def __init__(self, params):
        print("----- Initiating ----- ")
        print("----- step 1 configure logger")
        configure_logger(params['output_dir'])
        log('Parameters {}'.format(params))
        self.params = params
        print("----- step 2 load pre-collected things")
        self.binding = load_bindings(params['rom_file_path'])
        self.max_word_length = self.binding['max_word_length']
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(params['spm_file'])
        print("----- step 3 build KGA2CEnv")
        kg_env = KGA2CEnv(params['rom_file_path'], 
                        params['seed'], 
                        self.sp,
                        params['tsv_file'], 
                        step_limit=params['reset_steps'],
                        stuck_steps=params['stuck_steps'], 
                        gat=params['gat'])
        self.vec_env = VecEnv(params['batch_size'], 
                            kg_env, 
                            params['openie_path'])
        print("----- step 4 build FrotzEnv and templace generator")
        env = FrotzEnv(params['rom_file_path'])
        self.vocab_act, self.vocab_act_rev = load_vocab(env)
        self.template_generator = TemplateActionGenerator(self.binding)
        print("----- step 5 build kga2c model")
        self.model = KGA2C(params, 
                           self.template_generator.templates, 
                           self.max_word_length,
                           self.vocab_act, 
                           self.vocab_act_rev, 
                           len(self.sp), 
                           gat=self.params['gat']).cuda()
        if params['preload_weights']:
            print("load pretrained")
            self.model = torch.load(self.params['preload_weights'])['model']
        else:
            print("train from scratch")
        print("----- step 6 set training parameters")
        self.batch_size = params['batch_size']
        self.optimizer = optim.Adam(self.model.parameters(), lr=params['lr'])
        self.loss_fn1 = nn.BCELoss()
        self.loss_fn2 = nn.BCEWithLogitsLoss()
        self.loss_fn3 = nn.MSELoss()
        print("----- Init finished! ----- ")


    def generate_targets(self, admissible, objs):
        '''
        Generates ground-truth targets for admissible actions.
        :param admissible: List-of-lists of admissible actions. Batch_size x Admissible
        :param objs: List-of-lists of interactive objects. Batch_size x Objs
        :returns: template targets and object target tensors
        '''
        tmpl_target = []
        obj_targets = []
        for adm in admissible:
            obj_t = set()
            cur_t = [0] * len(self.template_generator.templates)
            for a in adm:
                cur_t[a.template_id] = 1
                obj_t.update(a.obj_ids)
            tmpl_target.append(cur_t)
            obj_targets.append(list(obj_t))
        tmpl_target_tt = torch.FloatTensor(tmpl_target).cuda()
        # Note: Adjusted to use the objects in the admissible actions only
        object_mask_target = []
        for objl in obj_targets: # in objs
            cur_objt = [0] * len(self.vocab_act)
            for o in objl:
                cur_objt[o] = 1
            object_mask_target.append([[cur_objt], [cur_objt]])
        obj_target_tt = torch.FloatTensor(object_mask_target).squeeze().cuda()
        return tmpl_target_tt, obj_target_tt


    def generate_graph_mask(self, graph_infos):
        assert len(graph_infos) == self.batch_size
        mask_all = []
        for graph_info in graph_infos:
            mask = [0] * len(self.vocab_act.keys())
            # Case 1 (default): KG as mask
            if self.params['masking'] == 'kg':
                graph_state = graph_info.graph_state          # Full KG as mask --> same as KG-A2C
                # graph_state = graph_info.graph_state_5_mask # sub_KG_5 as mask, disabled
                ents = set()
                # Obtain entities ---> maybe I can perform graph pooling before this step
                for u, v in graph_state.edges:
                    ents.add(u)
                    ents.add(v)
                # Build mask: only use those related to entities
                for ent in ents:
                    for ent_word in ent.split():
                        if ent_word[:self.max_word_length] in self.vocab_act_rev:
                            idx = self.vocab_act_rev[ent_word[:self.max_word_length]]
                            mask[idx] = 1
            # Case 2: interactive objects ground truth as the mask.
            elif self.params['masking'] == 'interactive':
                for o in graph_info.objs:
                    o = o[:self.max_word_length]
                    if o in self.vocab_act_rev.keys() and o != '':
                        mask[self.vocab_act_rev[o]] = 1
            # Case 3: no mask.
            elif self.params['masking'] == 'none':
                mask = [1] * len(self.vocab_act.keys())
            else:
                assert False, 'Unrecognized masking {}'.format(self.params['masking'])
            mask_all.append(mask)
        return torch.BoolTensor(mask_all).cuda().detach()


    def discount_reward(self, transitions, last_values):
        returns, advantages = [], []
        R = last_values.data
        for t in reversed(range(len(transitions))):
            _, _, values, rewards, done_masks, _, _, _, _, _, _ = transitions[t]
            R = rewards + self.params['gamma'] * R * done_masks
            adv = R - values
            returns.append(R)
            advantages.append(adv)
        return returns[::-1], advantages[::-1]


    def train(self, max_steps):
        print("=== === === start training!!! === === ===")
        start = time.time()
        transitions = []
        obs, infos, graph_infos = self.vec_env.reset()
        for step in range(1, max_steps + 1):
            # Step 1: build model inputs
            tb.logkv('Step', step)
            obs_reps = np.array([g.ob_rep for g in graph_infos])
            scores = [info['score'] for info in infos]
            graph_mask_tt = self.generate_graph_mask(graph_infos)
            graph_state_reps = [g.graph_state_rep for g in graph_infos]
            graph_rep_1 = [g.graph_state_rep_1_connectivity for g in graph_infos]
            graph_rep_2 = [g.graph_state_rep_2_roomitem for g in graph_infos]
            graph_rep_3 = [g.graph_state_rep_3_youritem for g in graph_infos]
            graph_rep_4 = [g.graph_state_rep_4_otherroom for g in graph_infos]
            # Step 2: predict probs, actual items
            tmpl_pred_tt, obj_pred_tt, dec_obj_tt, dec_tmpl_tt, value, dec_steps = self.model(
                                                                                obs_reps, 
                                                                                scores, 
                                                                                graph_state_reps, 
                                                                                graph_rep_1, 
                                                                                graph_rep_2,
                                                                                graph_rep_3,
                                                                                graph_rep_4,
                                                                                graph_mask_tt)
            tb.logkv_mean('Value', value.mean().item())
            # Step 3: Log the predictions and ground truth values
            topk_tmpl_probs, topk_tmpl_idxs = F.softmax(tmpl_pred_tt[0]).topk(5)
            topk_tmpls = [self.template_generator.templates[t] for t in topk_tmpl_idxs.tolist()]
            tmpl_pred_str = ', '.join(['{} {:.3f}'.format(tmpl, prob) for tmpl, prob in zip(topk_tmpls, topk_tmpl_probs.tolist())])
            # Step 4: Generate the ground truth and object mask
            admissible = [g.admissible_actions for g in graph_infos]
            objs = [g.objs for g in graph_infos]
            tmpl_gt_tt, obj_mask_gt_tt = self.generate_targets(admissible, objs)
            # Step 5: Log template/object predictions/ground_truth
            gt_tmpls = [self.template_generator.templates[i] for i in tmpl_gt_tt[0].nonzero().squeeze().cpu().numpy().flatten().tolist()]
            gt_objs = [self.vocab_act[i] for i in obj_mask_gt_tt[0,0].nonzero().squeeze().cpu().numpy().flatten().tolist()]
            topk_o1_probs, topk_o1_idxs = F.softmax(obj_pred_tt[0,0]).topk(5)
            topk_o1 = [self.vocab_act[o] for o in topk_o1_idxs.tolist()]
            o1_pred_str = ', '.join(['{} {:.3f}'.format(o, prob) for o, prob in zip(topk_o1, topk_o1_probs.tolist())])
            chosen_actions = self.decode_actions(dec_tmpl_tt, dec_obj_tt)
            # Step 6: Next step
            obs, rewards, dones, infos, graph_infos = self.vec_env.step(chosen_actions)
            # Step 7: logging
            tb.logkv_mean('TotalStepsPerEpisode', sum([i['steps'] for i in infos]) / float(len(graph_infos)))
            tb.logkv_mean('Valid', infos[0]['valid'])
            if dones[0]:
                log('Step {} EpisodeScore {}'.format(step, infos[0]['score']))
            for done, info in zip(dones, infos):
                if done:
                    tb.logkv_mean('EpisodeScore', info['score'])
            # Step 8: append into transitions
            rew_tt = torch.FloatTensor(rewards).cuda().unsqueeze(1)
            done_mask_tt = (~torch.tensor(dones)).float().cuda().unsqueeze(1)
            self.model.reset_hidden(done_mask_tt)
            transitions.append((tmpl_pred_tt, 
                                obj_pred_tt, 
                                value, 
                                rew_tt,
                                done_mask_tt, 
                                tmpl_gt_tt, 
                                dec_tmpl_tt,
                                dec_obj_tt, 
                                obj_mask_gt_tt, 
                                graph_mask_tt, 
                                dec_steps))
            # Step 9: update model per 8 steps
            if len(transitions) >= self.params['bptt']:
                tb.logkv('StepsPerSecond', float(step) / (time.time() - start))
                self.model.clone_hidden()
                obs_reps = np.array([g.ob_rep for g in graph_infos])
                scores = [info['score'] for info in infos]
                graph_mask_tt = self.generate_graph_mask(graph_infos)
                graph_state_reps = [g.graph_state_rep for g in graph_infos]
                graph_rep_1 = [g.graph_state_rep_1_connectivity for g in graph_infos]
                graph_rep_2 = [g.graph_state_rep_2_roomitem for g in graph_infos]
                graph_rep_3 = [g.graph_state_rep_3_youritem for g in graph_infos]
                graph_rep_4 = [g.graph_state_rep_4_otherroom for g in graph_infos]
                _, _, _, _, next_value, _ = self.model(obs_reps, 
                                                        scores, 
                                                        graph_state_reps, 
                                                        graph_rep_1,
                                                        graph_rep_2,
                                                        graph_rep_3,
                                                        graph_rep_4,
                                                        graph_mask_tt)
                returns, advantages = self.discount_reward(transitions, next_value)
                tb.logkv_mean('Advantage', advantages[-1].median().item())
                loss = self.update(transitions, returns, advantages)
                del transitions[:]
                self.model.restore_hidden()
                print("Total time: {:.2f} mins".format((time.time() - start) / 60.))
            if step % self.params['checkpoint_interval'] == 0:
                parameters = { 'model': self.model }
                torch.save(parameters, os.path.join(self.params['output_dir'], 'kga2c.pt'))
        self.vec_env.close_extras()

    
    def update(self, transitions, returns, advantages):
        assert len(transitions) == len(returns) == len(advantages)
        loss = 0
        for trans, ret, adv in zip(transitions, returns, advantages):
            tmpl_pred_tt, obj_pred_tt, value, _, _, tmpl_gt_tt, dec_tmpl_tt, \
                dec_obj_tt, obj_mask_gt_tt, graph_mask_tt, dec_steps = trans
            # Supervised Template Loss
            tmpl_probs = F.softmax(tmpl_pred_tt, dim=1)
            template_loss = self.params['template_coeff'] * self.loss_fn1(tmpl_probs, tmpl_gt_tt)
            # Supervised Object Loss
            object_mask_target = obj_mask_gt_tt.permute((1, 0, 2))
            obj_probs = F.softmax(obj_pred_tt, dim=2)
            object_mask_loss = self.params['object_coeff'] * self.loss_fn1(obj_probs, object_mask_target)
            # Build the object mask
            o1_mask, o2_mask = [0] * self.batch_size, [0] * self.batch_size
            for d, st in enumerate(dec_steps):
                if st > 1:
                    o1_mask[d] = 1
                    o2_mask[d] = 1
                elif st == 1:
                    o1_mask[d] = 1
            o1_mask = torch.FloatTensor(o1_mask).cuda()
            o2_mask = torch.FloatTensor(o2_mask).cuda()
            # Policy Gradient Loss
            policy_obj_loss = torch.FloatTensor([0]).cuda()
            cnt = 0
            for i in range(self.batch_size):
                if dec_steps[i] >= 1:
                    cnt += 1
                    batch_pred = obj_pred_tt[0, i, graph_mask_tt[i]]
                    action_log_probs_obj = F.log_softmax(batch_pred, dim=0)
                    dec_obj_idx = dec_obj_tt[0,i].item()
                    graph_mask_list = graph_mask_tt[i].nonzero().squeeze().cpu().numpy().flatten().tolist()
                    idx = graph_mask_list.index(dec_obj_idx)
                    log_prob_obj = action_log_probs_obj[idx]
                    policy_obj_loss += -log_prob_obj * adv[i].detach()
            if cnt > 0:
                policy_obj_loss /= cnt
            tb.logkv_mean('PolicyObjLoss', policy_obj_loss.item())
            log_probs_obj = F.log_softmax(obj_pred_tt, dim=2)
            log_probs_tmpl = F.log_softmax(tmpl_pred_tt, dim=1)
            action_log_probs_tmpl = log_probs_tmpl.gather(1, dec_tmpl_tt).squeeze()
            policy_tmpl_loss = (-action_log_probs_tmpl * adv.detach().squeeze()).mean()
            tb.logkv_mean('PolicyTemplateLoss', policy_tmpl_loss.item())
            policy_loss = policy_tmpl_loss + policy_obj_loss
            value_loss = self.params['value_coeff'] * self.loss_fn3(value, ret)
            tmpl_entropy = -(tmpl_probs * log_probs_tmpl).mean()
            tb.logkv_mean('TemplateEntropy', tmpl_entropy.item())
            object_entropy = -(obj_probs * log_probs_obj).mean()
            tb.logkv_mean('ObjectEntropy', object_entropy.item())
            # Minimizing entropy loss will lead to increased entropy
            entropy_loss = self.params['entropy_coeff'] * -(tmpl_entropy + object_entropy)
            loss += template_loss + object_mask_loss + value_loss + entropy_loss + policy_loss
        tb.logkv('Loss', loss.item())
        tb.logkv('TemplateLoss', template_loss.item())
        tb.logkv('ObjectLoss', object_mask_loss.item())
        tb.logkv('PolicyLoss', policy_loss.item())
        tb.logkv('ValueLoss', value_loss.item())
        tb.logkv('EntropyLoss', entropy_loss.item())
        tb.dumpkvs()
        loss.backward()
        # Compute the gradient norm
        grad_norm = 0
        for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
            grad_norm += p.grad.data.norm(2).item()
        tb.logkv('UnclippedGradNorm', grad_norm)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.params['clip'])
        # Clipped Grad norm
        grad_norm = 0
        for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
            grad_norm += p.grad.data.norm(2).item()
        tb.logkv('ClippedGradNorm', grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    
    def decode_actions(self, decoded_templates, decoded_objects):
        '''
        Returns string representations of the given template actions.
        :param decoded_template: Tensor of template indices.
        :type decoded_template: Torch tensor of size (Batch_size x 1).
        :param decoded_objects: Tensor of o1, o2 object indices.
        :type decoded_objects: Torch tensor of size (2 x Batch_size x 1).
        '''
        decoded_actions = []
        for i in range(self.batch_size):
            decoded_template = decoded_templates[i].item()
            decoded_object1 = decoded_objects[0][i].item()
            decoded_object2 = decoded_objects[1][i].item()
            decoded_action = self.tmpl_to_str(decoded_template, decoded_object1, decoded_object2)
            decoded_actions.append(decoded_action)
        return decoded_actions


    def tmpl_to_str(self, template_idx, o1_id, o2_id):
        """ Returns a string representation of a template action. """
        template_str = self.template_generator.templates[template_idx]
        holes = template_str.count('OBJ')
        assert holes <= 2
        if holes <= 0:
            return template_str
        elif holes == 1:
            return template_str.replace('OBJ', self.vocab_act[o1_id])
        else:
            return template_str.replace('OBJ', self.vocab_act[o1_id], 1)\
                               .replace('OBJ', self.vocab_act[o2_id], 1)
