
from logging import getLogger

import torch
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
import random
import numpy as np
import os
from new_version.CVRP.CVRPModel import CVRPModel as Model
from new_version.CVRP.CVRPEnv import CVRPEnv as Env
from new_version.utils.utils import *
from new_version.utils.beamsearch import Beamsearch
# from torch_cluster import knn

 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True)
    # os.environ['PYTHONHASHSEED'] = str(seed)


class CVRPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):


        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda'] # True
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
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

        random_seed = 123
        set_seed(random_seed)
        # Main Components
        self.model = Model(**self.model_params)

        self.env = Env(**self.env_params)

        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)

        score_student_AM = AverageMeter()
                       
        score_avg_ = AverageMeter()

        episode = 0

        batch_size = self.trainer_params['train_batch_size']
        beam_size = self.trainer_params['beam_size']
        self.env.beam_size = self.trainer_params['beam_size']
        problem_size = self.trainer_params['problem_size']
        self.env.step_size = self.trainer_params['step_size']

        self.env.load_problems(episode,batch_size, problem_size)

        # self.env.pomo_size = min(self.trainer_params['knn'],self.env.pomo_size)

        best_score = self.env.greedy_search()

        # print("env solution", self.env.solution)
        # exit(0)


        save_gap = []
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')
            # Train
            train_score, train_student_score, train_loss, replace_score = self._train_one_epoch(epoch, score_avg_)

            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_student_score', epoch, train_student_score)
            self.result_log.append('train_loss', epoch, train_loss)
            self.result_log.append('replace_score', epoch, replace_score)

            if epoch % 5 == 0 and epoch != 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))
                self.scheduler.step()


            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_student_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_3'],
                                    self.result_log, labels=['replace_score'])

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch, score_avg_):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        gap_AM = AverageMeter()
        best_score_AM = AverageMeter()
        replace_score_AM = AverageMeter()

        episode = 0
        batch_size = self.trainer_params['train_batch_size']
        beam_size = self.trainer_params['beam_size']

        result_AM = AverageMeter()
        self.env.replace_size = (((self.env_params['max_replace_ratio'] - self.env_params['min_replace_ratio']) * (1 - epoch/self.trainer_params['epochs']))  + self.env_params['min_replace_ratio']) * self.trainer_params['train_batch_size']
        print("self.env.replace_size: ", self.env.replace_size)
        self.env.replace_size = int(self.env.replace_size)
        for i in range(100):
            loss_AM.reset()

            random_index = self.env.random_replace()
            # print("random_index", random_index)
            self.env.start_idx = 0
            self.env.start_idx = 0

            while self.env.start_idx < batch_size:
                best_score, score, gap = self.beamsearch_tour_nodes_shortest(beam_size, self.env.step_size, self.env.problem_size + 1, 
                                                                self.dtypeFloat, self.dtypeLong, probs_type='logits', random_start=False)
            
                gap_AM.update(gap, self.env.step_size)
                score_AM.update(score, self.env.step_size)
                best_score_AM.update(best_score, self.env.step_size)
                self.env.start_idx += self.env.step_size

            # best_score, score, gap = self.beamsearch_tour_nodes_shortest(beam_size, batch_size, self.env.problem_size + 1, 
            #                                                 self.dtypeFloat, self.dtypeLong, probs_type='logits', random_start=False)
            
            # gap_AM.update(gap, self.env.batch_size)
            # score_AM.update(score, self.env.batch_size)
            # best_score_AM.update(best_score, self.env.batch_size)

            # self.env.start_idx += self.env.batch_size

            # print("self.env.solution_len[random_index]", self.env.solution_len[random_index])
            replace_score = torch.mean(self.env.solution_len[random_index]).item()
            replace_score_AM.update(replace_score, batch_size)

            avg_loss = self._train_one_batch(episode,batch_size,epoch)

            result_AM.update(score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            self.logger.info('Epoch {:3d}:  Loop:{:3d}  avg_score {:.4f} Loss: {:.4f} replace_score: {:.4f}'.format(epoch, i, result_AM.avg ,loss_AM.avg, replace_score_AM.avg))

            self.logger.info('best_score {:4f}:  score:{:4f}  gap {:.4f}'.format(best_score, score, gap))
 

        return score_AM.avg, result_AM.avg, loss_AM.avg, replace_score_AM.avg

    def _train_one_batch(self, episode,batch_size,epoch):
        # print("self.env.solution", self.env.solution)


        torch.cuda.empty_cache()
        ###############################################
        self.model.train()
        
        reset_state, _, _ = self.env.reset('train')

        prob_list = torch.ones(size=(batch_size, 0))

        state, reward, done = self.env.pre_step()

        self.model.mode = 'train'
        self.model.pre_forward(self.env.dis_matrix, self.env.batch_size)


        current_step=0
        knn = self.trainer_params['knn']
        depot_knn_nums = self.trainer_params['depot_knn']

        loss_mean = 0
        while not done:
            if current_step == 0:
                selected = self.env.solution[:, 0] # starting node
                prob = torch.ones(self.env.solution.shape[0], 1)

                selected_flag = self.env.solution_flag[:, current_step]
                is_via_depot = selected_flag==1
                selected[is_via_depot] += (self.env.problem_size + 1)

                # print("selected", selected)

            else:

                selected, prob, probs = self.model(state, self.env.selected_node_list, self.env.solution, self.env.solution_flag ,knn ,current_step, depot_knn_nums)  # 更新被选择的点和概率
                loss_mean = -prob.type(torch.float64).log().mean()
                self.model.zero_grad()
                loss_mean.backward()
                self.optimizer.step()

            current_step+=1

            state, reward, done = self.env.step(selected)
            
            prob_list = torch.cat((prob_list, prob), dim=1)
        
        loss_mean = -prob_list.log().mean()

        return loss_mean.item()

    @torch.no_grad()
    def beamsearch_tour_nodes_shortest(self, beam_size, batch_size, num_nodes,
                                   dtypeFloat, dtypeLong, probs_type='raw', random_start=False):
        torch.cuda.empty_cache()
        # Perform beamsearch
        beamsearch = Beamsearch(beam_size, batch_size*self.env.pomo_size, num_nodes, dtypeFloat, dtypeLong, probs_type, random_start, "CVRP")

        current_step = 0

        reset_state, _, _ = self.env.reset('valid')

        state, reward, done = self.env.pre_step()

        self.model.eval()
        self.model.mode = 'valid'
        self.model.pre_forward(self.env.dis_matrix, self.env.step_size)

        knn_nums = self.trainer_params['knn']
        depot_knn_nums = self.trainer_params['depot_knn']

        while not done:
            if current_step == 0:

                # batch_size = self.env.batch_size
                # problem_size = state.data.shape[1]

                # new_list = torch.arange(problem_size)[None, :].repeat(batch_size, 1)

                # new_list_len = problem_size - 1  # shape: [B, V-current_step]

                # index_1 = torch.arange(batch_size, dtype=torch.long)
                # new_list[index_1, 0] = -2

                # unselect_list = new_list[torch.gt(new_list, -1)].view(batch_size, new_list_len)
                
                # index_1_unselect = torch.arange(0,batch_size).repeat_interleave(new_list_len)

                # index_2 = unselect_list.view((-1,))

                # unselect_list_node = state.data[:,:,:2][index_1_unselect,index_2].view((-1,2))
            
                # index_1_last = torch.arange(0,batch_size)

                # depot_node = state.data[index_1_last,0,:2]

                # knn_output = knn(unselect_list_node, depot_node, knn_nums, index_1_unselect, index_1_last)

                # knn_idx = knn_output[1,:] % new_list_len

                # knn_idx = knn_idx.view((batch_size, knn_nums)).contiguous()

                # knn_origin_idx = unselect_list.gather(1, knn_idx)

                # perm = torch.randperm(knn_origin_idx.shape[1])

                # shuffled_tensor = knn_origin_idx[:, perm]

                # knn_origin_idx = knn_origin_idx[:,:self.env.pomo_size]

                # selected = knn_origin_idx.reshape(-1) + 1 + self.env.problem_size

                # selected = selected.repeat_interleave(self.env.beam_size)
                # print("self.env.problem_size", self.env.problem_size)
                random_index = torch.randperm(self.env.problem_size)[:self.env.pomo_size]
                random_index = random_index.repeat(batch_size)
                random_index = random_index.repeat_interleave(self.env.beam_size)

                # print("random_index", random_index.tolist())

                # selected = torch.randint(self.env.problem_size, (self.env.batch_size*self.env.pomo_size*beam_size,),dtype=torch.int64) + 2 + self.env.problem_size
                selected = random_index.to(torch.int64) + 2 + self.env.problem_size

            else:
                _, trans_probs, _ = self.model(
                            state,self.env.selected_node_list, None, None, knn_nums ,current_step, depot_knn_nums)

                self.env.selected_node_list = beamsearch.advance(torch.log(trans_probs.view(batch_size*self.env.pomo_size, beam_size, -1)), self.env, knn_nums , current_step)

                selected = beamsearch.next_nodes[-1].view(-1)
                
            current_step += 1

            state, reward, done = self.env.step(selected)

        view_reward = reward.view(batch_size,beam_size*self.env.pomo_size)

        shortest_lens, index = torch.min(view_reward, dim=1)

        shortest_tours = torch.gather(self.env.selected_node_list.view(batch_size,beam_size*self.env.pomo_size,-1), 1, index[:,None].unsqueeze(1).expand(batch_size,1,num_nodes-1)).squeeze(1)
        shortest_tours_flag = torch.gather(self.env.step_state.selected_flag.view(batch_size,beam_size*self.env.pomo_size,-1), 1, index[:,None].unsqueeze(1).expand(batch_size,1,num_nodes-1)).squeeze(1)

        if self.env.solution == None:
            self.env.solution = shortest_tours
            self.env.solution_flag = shortest_tours_flag
            self.env.solution_len = shortest_lens
        else:
            index = torch.gt(self.env.solution_len[self.env.start_idx:self.env.start_idx + self.env.step_size], shortest_lens)
            self.env.solution[self.env.start_idx:self.env.start_idx + self.env.step_size][index] = shortest_tours[index]
            self.env.solution_flag[self.env.start_idx:self.env.start_idx + self.env.step_size][index] = shortest_tours_flag[index]
            self.env.solution_len[self.env.start_idx:self.env.start_idx + self.env.step_size][index] = shortest_lens[index]


        
        # print("===========self.env.solution_len=========")
        # print(self.env.solution_len)

        best_score = self.env.solution_len[self.env.start_idx:self.env.start_idx + self.env.step_size].mean()
        score = shortest_lens.mean()


        gap = (score - best_score) / best_score

        self.env.data_augmentation()

        return best_score.item(), score.item(), gap.item()
    





