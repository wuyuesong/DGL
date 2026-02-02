
from logging import getLogger

import torch
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from DGL.TSP.TSPModel import TSPModel as Model
# from LEHD_modify2.TSP.test import main_test
from DGL.TSP.TSPEnv import TSPEnv as Env
from DGL.utils.utils import *
from DGL.utils.beamsearch import Beamsearch
import random
import numpy as np

import time

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  

class TSPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 tester_params,
                 valid_params):


        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.tester_params = tester_params
        self.valid_params = valid_params

        # result folder, logger
        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.tester_params['use_cuda'] # True
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.dtypeFloat = torch.cuda.FloatTensor
            self.dtypeLong = torch.cuda.LongTensor
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
            self.dtypeFloat = torch.FloatTensor
            self.dtypeLong = torch.LongTensor

        random_seed = 789
        set_seed(random_seed)
        # Main Components
        self.model = Model(**self.model_params)

        self.env = Env(**self.env_params)

        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = tester_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/{file}'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):

        self.time_estimator.reset(self.start_epoch)

        episode = 0

        batch_size = self.tester_params['test_batch_size']
        beam_size = self.tester_params['beam_size']
        problem_size = self.tester_params['problem_size']
        test_episodes_all = self.tester_params['test_episodes_all']
        test_episodes = self.tester_params['test_episodes']
        self.env.load_data(test_episodes_all, problem_size)

        print("greedy:", self.env.solution_len)

        score_AM = AverageMeter()
        gap_AM = AverageMeter()
        best_score_AM = AverageMeter()

        print("=======================self.env.baseline_len:", torch.mean(self.env.baseline_len))
        save_gap = []
        self.model.eval()


        with torch.no_grad():
            while episode < test_episodes:
                
                self.env.load_problems(episode,batch_size, problem_size, beam_size)
                # Train
                self.env.start_idx = 0
                self.env.step_size = self.tester_params['test_batch_size']
                # print("self.env.step_size", self.env.step_size)
                while self.env.start_idx < batch_size:
                    best_gap, gap, best_score,score = self.beamsearch_tour_nodes_shortest(beam_size, self.env.batch_size, self.env.problem_size, 
                                                                    self.dtypeFloat, self.dtypeLong, probs_type='logits', random_start=False)
                    del self.model.dis_matrix
                    torch.cuda.empty_cache()


                    score_AM.update(score, self.env.step_size)
                    best_score_AM.update(best_score, self.env.step_size)

                    self.env.start_idx += self.env.step_size

                batch_baseline_len = torch.mean(self.env.baseline_len[episode:episode + batch_size])
                # self.logger.info("batch_baseline: %s", self.env.baseline_len[episode:episode + batch_size])
                episode += batch_size
                

                batch_gap = (score - batch_baseline_len)/batch_baseline_len

                gap_AM.update(batch_gap, batch_size)

                self.logger.info("episode {:3d}: avg_score {:.4f} batch_gap {:.4f} avg_gap {:.4f}".format(episode,score_AM.avg,batch_gap,gap_AM.avg))

        avg_gap = (score_AM.avg - torch.mean(self.env.baseline_len[:test_episodes]))/torch.mean(self.env.baseline_len[:test_episodes])
        self.logger.info("gap {:.4f}".format(avg_gap))

        return avg_gap

    @torch.no_grad()
    def beamsearch_tour_nodes_shortest(self, beam_size, batch_size, num_nodes,
                                   dtypeFloat, dtypeLong, probs_type='raw', random_start=False):
        # Perform beamsearch
        beamsearch = Beamsearch(beam_size, self.env.batch_size*self.env.pomo_size, num_nodes, dtypeFloat, dtypeLong, probs_type, random_start)

        current_step = 0

        reset_state, _, _ = self.env.reset('test')
        torch.cuda.empty_cache()

        state, reward, done = self.env.pre_step()

        self.model.eval()
        self.model.mode = 'test'
        self.model.pre_forward(self.env.dis_matrix, self.env.batch_size)

        knn = self.tester_params['knn']

        while not done:
            if current_step == 0:
                random_index = torch.randperm(self.env.problem_size)[:self.env.pomo_size]
                random_index = random_index.repeat(self.env.batch_size)
                random_index = random_index.repeat_interleave(self.env.beam_size)
                selected = random_index.to(torch.int64) 

            else:
                _, trans_probs, _ = self.model(
                            state,self.env.selected_node_list,None, knn ,current_step)

                self.env.selected_node_list = beamsearch.advance(torch.log(trans_probs.view(self.env.batch_size*self.env.pomo_size, beam_size, -1)), self.env, knn , current_step)

                selected = beamsearch.next_nodes[-1].view(-1)
                
            current_step += 1

            state, reward, done = self.env.step(selected)


        if self.tester_params["test_mode"] == "aug_test":
            view_reward = reward.view(self.env.batch_size//self.env.aug_size, beam_size*self.env.pomo_size*self.env.aug_size)
        else:
            view_reward = reward.view(self.env.batch_size,beam_size*self.env.pomo_size)

        shortest_lens, index = torch.min(view_reward, dim=1)

        if self.tester_params["test_mode"] == "aug_test":
            shortest_tours = torch.gather(self.env.selected_node_list.view(self.env.batch_size//self.env.aug_size,beam_size*self.env.pomo_size*self.env.aug_size,-1), 1, index[:,None].unsqueeze(1).expand(self.env.batch_size//self.env.aug_size,1,num_nodes-1)).squeeze(1)
        else:
            shortest_tours = torch.gather(self.env.selected_node_list.view(self.env.batch_size,beam_size*self.env.pomo_size,-1), 1, index[:,None].unsqueeze(1).expand(self.env.batch_size,1,num_nodes-1)).squeeze(1)


        score = shortest_lens.mean()

        # return keep, best_gap.item(), gap.item(), score.item()
        return 0, 0, 0, score.item()
    





