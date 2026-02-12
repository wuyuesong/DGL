
from dataclasses import dataclass
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.utils import *


import time
import functools

def time_count(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        wrapper.total_time += run_time
        wrapper.calls += 1
        print(f"Function {func.__name__} called {wrapper.calls} times. Total run time: {wrapper.total_time:.4f} seconds. Last run time: {run_time:.4f} seconds.")
        return result
    wrapper.total_time = 0
    wrapper.calls = 0
    return wrapper


def get_random_problems(batch_size, problem_size):
    problems = torch.rand(size=(batch_size, problem_size, 2))
    # problems.shape: (batch, problem, 2)
    return problems

@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)

@dataclass
class Step_State:
    data: torch.Tensor
    avg_unselect_distence: torch.Tensor = None
    std_dev_unselect_distence: torch.Tensor = None

class TSPEnv:
    def __init__(self, **env_params):

        self.env_params = env_params
        self.problem_size = None
        self.data_path = env_params['data_path']
        self.mode = env_params['mode']


        if self.mode == 'test':
            self.baseline_path = env_params['baseline_path']

        if self.mode == 'test':
            if env_params['test_mode'] == 'pomo_test':
                self.pomo_size = env_params['aug_size']
            elif env_params['test_mode'] == 'aug_test':
                self.aug_size = env_params['aug_size']
                self.pomo_size = 1
        else:
            self.pomo_size = env_params["pomo_size"]

        self.append_information = env_params['append_information']
        self.batch_size = None
        self.problems = None
        self.raw_data_nodes = []
        self.raw_data_tours = []
        self.selected_count = None
        self.selected_node_list = None
        self.selected_student_list = None
        self.episode = None
        self.solution = None
        self.solution_len = None
        self.step_state = None
        self.best_solution = []
        self.best_solution_len = []
        self.aggregation_nums = env_params['aggregation_nums']
    
    def load_data(self, test_episode, problem_size):
        # read from file
        with open(self.data_path, 'r') as file:
            content = file.read()

        # split by space
        points = content.split()

        # split by comma
        point_list = [point.split(',') for point in points]
        point_list = [[float(item) for item in inner_list] for inner_list in point_list]

        tensor_nodes = torch.tensor(point_list, dtype=torch.float32)

        self.baseline_len = []
        if self.baseline_path != "" :
            with open(self.baseline_path, 'r') as file:

                for line in file:
                    len = float(line.split()[1])
                    self.baseline_len.append(len)

            self.baseline_len = torch.tensor(self.baseline_len)
        
        self.raw_data_nodes = tensor_nodes.view(test_episode, problem_size, 2)

    def load_problems(self, episode, batch_size, problem_size, beam_size, ):
        self.episode = episode

        self.batch_size = batch_size

        self.problem_size = problem_size
        if self.mode == 'train':
            self.problems = get_random_problems(batch_size, problem_size)
        elif self.mode == 'test':
            self.problems = self.raw_data_nodes[episode:episode + batch_size]

            if self.env_params['test_mode'] == "aug_test":
                self.problems = self.problems.repeat_interleave(self.aug_size, dim=0) 
                self.batch_size = self.batch_size * self.aug_size

        self.beam_size = beam_size

        self.dis_matrix = torch.cdist(self.problems, self.problems, p=2)
        torch.cuda.empty_cache()
        
        self.reset_solution()

    def random_replace(self):
        random_index = torch.randperm(self.batch_size)[:self.replace_size]
        replace_data = get_random_problems(self.replace_size, self.problem_size)
        self.problems[random_index] = replace_data
        self.solution_len[random_index] = torch.tensor(float('inf')).repeat(self.replace_size)
        self.dis_matrix = torch.cdist(self.problems, self.problems, p=2)

        return random_index


    def greedy_search(self):
        selected_node_list = torch.zeros(self.batch_size, dtype=torch.int64).unsqueeze(-1)

        step = 1

        while step < self.problem_size:

            last_node = selected_node_list[:,-1]

            distance = self.dis_matrix.gather(1, last_node.view(self.batch_size, 1, 1).expand(-1, -1, self.problem_size)).squeeze(1)

            mask = torch.zeros_like(distance)

            index_1 = torch.arange(self.batch_size, dtype=torch.long)[:, None].expand(self.batch_size, selected_node_list.shape[1])

            mask[index_1, selected_node_list] = 1e5

            selected_node = torch.argsort(distance + mask, dim=1)[:, :1]

            selected_node_list = torch.cat((selected_node_list, selected_node), dim=1)

            step += 1

        self.solution = selected_node_list

        index = torch.arange(self.batch_size)[:,None].expand_as(selected_node_list)

        self.solution_len = torch.sum(self.dis_matrix[index, selected_node_list, torch.roll(selected_node_list, -1, dims=1)], dim=1)

        return self.solution_len.mean().item()

    @torch.no_grad()
    def get_best_solution(self):

        self.best_solution = []
        self.best_solution_len = []

        for problem in self.problems:
            xs = []
            ys = []
            for (x, y) in problem:
                xs.append(int(x * 10000))
                ys.append(int(y * 10000))
            solver = TSPSolver.from_data(xs, ys, norm="EUC_2D")
            solution = solver.solve()
            print(solution.found_tour)
            self.best_solution.append(solution.tour)
            self.best_solution_len.append(solution.optimal_value/10000)

        self.best_solution = torch.tensor(self.best_solution, dtype=torch.long)
        self.best_solution_len = torch.tensor(self.best_solution_len, dtype=torch.float)

    @torch.no_grad()
    def get_unselect_list(self):

        batch_size = self.selected_node_list.shape[0]

        new_list = torch.arange(self.problem_size)[None, :].repeat(batch_size, 1)

        new_list_len = self.problem_size - self.selected_node_list.shape[1]  # shape: [B, V-current_step]

        index_2 = self.selected_node_list.type(torch.long)

        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, index_2.shape[1])

        new_list[index_1, index_2] = -2

        unselect_list = new_list[torch.gt(new_list, -1)].view(batch_size, new_list_len)

        return unselect_list
        
    @torch.no_grad()
    def reset(self, mode,):

        self.selected_count = 0

        if mode == 'test':
            self.mode = mode
            with torch.no_grad():
                self.selected_node_list = torch.zeros((self.batch_size*self.pomo_size*self.beam_size, 0), dtype=torch.long)


                if self.env_params['test_mode'] == "aug_test":
                    self.step_state = Step_State(data=self.problems[self.start_idx:self.start_idx + self.batch_size].clone())
                    self.step_state.data[:,:,:2] = run_aug("mix", self.step_state.data[:,:,:2])
                else:
                    self.step_state = Step_State(data=self.problems[self.start_idx:self.start_idx + self.batch_size])

                self.dis_matrix = torch.cdist(self.step_state.data[:,:,:2], self.step_state.data[:,:,:2], p=2)

                self.step_state.data = torch.repeat_interleave(self.step_state.data ,self.pomo_size ,0)

                self.step_state.data = torch.repeat_interleave(self.step_state.data ,self.beam_size ,0)

                self.step_state.avg_unselect_distence = torch.mean(self.dis_matrix, dim=-1).repeat_interleave(self.pomo_size*self.beam_size, dim=0)
                self.step_state.std_dev_unselect_distence = torch.std(self.dis_matrix, unbiased=False, dim=-1).repeat_interleave(self.pomo_size*self.beam_size, dim=0)

        
        if mode == 'valid':
            self.mode = mode
            with torch.no_grad():
                self.selected_node_list = torch.zeros((self.step_size*self.pomo_size*self.beam_size, 0), dtype=torch.long)

                self.step_state = Step_State(data=self.problems[self.start_idx:self.start_idx + self.step_size])

                self.dis_matrix = torch.cdist(self.step_state.data[:,:,:2], self.step_state.data[:,:,:2], p=2)

                self.step_state.data = torch.repeat_interleave(self.step_state.data ,self.pomo_size ,0)

                self.step_state.data = torch.repeat_interleave(self.step_state.data ,self.beam_size ,0)

                self.step_state.avg_unselect_distence = torch.mean(self.dis_matrix, dim=-1).repeat_interleave(self.pomo_size*self.beam_size, dim=0)
                self.step_state.std_dev_unselect_distence = torch.std(self.dis_matrix, unbiased=False, dim=-1).repeat_interleave(self.pomo_size*self.beam_size, dim=0)
                

        if mode == 'train':
            self.mode = mode
            self.selected_node_list = torch.zeros((self.batch_size, 0), dtype=torch.long)

            self.step_state = Step_State(data=self.problems.clone())

            self.step_state.data[:,:,:2] = run_aug("mix", self.step_state.data[:,:,:2])
            
            self.dis_matrix = torch.cdist(self.step_state.data, self.step_state.data, p=2)

            self.step_state.avg_unselect_distence = torch.mean(self.dis_matrix, dim=-1)
            self.step_state.std_dev_unselect_distence = torch.std(self.dis_matrix, unbiased=False, dim=-1)

        reward = None
        done = False
        return Reset_State(self.problems), reward, done


    def reset_solution(self):
        self.solution = None
        self.solution_len = None

    # @time_count
    def perm_attr(self, attr_name, prev_k):

        if self.mode == 'test' or self.mode == 'train':
            batch_size = self.batch_size
        elif self.mode == 'valid':
            batch_size = self.step_size

        attr = getattr(self.step_state, attr_name)
        
        if len(attr.shape) == 1:
            attr = attr.view(batch_size*self.pomo_size, self.beam_size)
            perm_attr = prev_k
            attr = attr.gather(1, perm_attr)
            attr = attr.view(batch_size*self.pomo_size*self.beam_size)
            setattr(self.step_state, attr_name, attr)

        elif len(attr.shape) == 2:
            
            attr = attr.view(batch_size*self.pomo_size, self.beam_size,-1)
            perm_attr = prev_k.unsqueeze(2).expand_as(attr)
            attr = attr.gather(1, perm_attr)
            attr = attr.view(batch_size*self.pomo_size*self.beam_size, -1)
            setattr(self.step_state, attr_name, attr)

        elif len(attr.shape) == 3:
            shape1 = attr.shape[1]
            attr = attr.view(batch_size*self.pomo_size, self.beam_size, shape1 ,-1)
            perm_attr = prev_k.unsqueeze(2).unsqueeze(3).expand_as(attr)
            attr = attr.gather(1, perm_attr)
            attr = attr.view(batch_size*self.pomo_size*self.beam_size, shape1, -1)
            setattr(self.step_state, attr_name, attr)


    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    @torch.no_grad()
    # @TimeCount
    def step(self, selected):

        self.selected_count += 1

        # print("self.selected_node_list", self.selected_node_list.shape)
        # print("selected[:, None]", selected[:, None].shape)

        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, None]), dim=1)  # shape: [B, current_step]

        range_index = torch.arange(self.step_state.data.shape[0], dtype=torch.long)

        dest_node_idx = None
        distance_to_dest = None
        last_node = None

        bs = self.step_state.data.shape[0]

        unselect_count = self.problem_size - self.selected_count


        if self.mode == "valid" or self.mode == "test":
            pomo_size = self.pomo_size
            beam_size = self.beam_size
        elif self.mode == "train":
            pomo_size = 1
            beam_size = 1

        if self.mode == 'test' or self.mode == 'train':
            batch_size = self.batch_size
        elif self.mode == 'valid':
            batch_size = self.step_size


        if self.append_information[2] == True:
            before_avg_unselect_distence = self.step_state.avg_unselect_distence


        feature_num = 0

        if self.append_information[0] == True:

            current_node = self.selected_node_list[:,-1].view(-1)

            current_node = current_node.repeat_interleave(self.problem_size)
            index1 = torch.arange(batch_size).repeat_interleave(pomo_size*beam_size*self.problem_size)
            index2 = torch.arange(self.problem_size).repeat(bs)

            distance_to_current = self.dis_matrix[index1, index2, current_node].view(bs, self.problem_size)

            feature_num = feature_num + 1
            if self.selected_count == 1:
                self.step_state.data = torch.cat((self.step_state.data, distance_to_current.unsqueeze(-1)), dim=-1)
            else:
                self.step_state.data[:,:,1 + feature_num] = distance_to_current

        if self.append_information[1] == True:
            
            if self.append_information[0] == False:

                current_node = self.selected_node_list[:,-1].view(-1)

                current_node = current_node.repeat_interleave(self.problem_size)
                index1 = torch.arange(batch_size).repeat_interleave(pomo_size*beam_size*self.problem_size)
                index2 = torch.arange(self.problem_size).repeat(bs)

                distance_to_current = self.dis_matrix[index1, index2, current_node].view(bs, self.problem_size)


            self.step_state.avg_unselect_distence = (self.step_state.avg_unselect_distence * (unselect_count + 1) - distance_to_current) / unselect_count            


            feature_num = feature_num + 1
            if self.selected_count == 1:
                self.step_state.data = torch.cat((self.step_state.data, self.step_state.avg_unselect_distence.unsqueeze(-1).to(torch.float32)), dim=-1)
            else:
                self.step_state.data[:,:,1 + feature_num] = self.step_state.avg_unselect_distence


        if self.append_information[2] == True:

            if self.append_information[0] == False:

                current_node = self.selected_node_list[:,-1].view(-1)

                current_node = current_node.repeat_interleave(self.problem_size)
                index1 = torch.arange(batch_size).repeat_interleave(pomo_size*beam_size*self.problem_size)
                index2 = torch.arange(self.problem_size).repeat(bs)

                distance_to_current = self.dis_matrix[index1, index2, current_node].view(bs, self.problem_size)
            
            self.step_state.std_dev_unselect_distence = \
            torch.sqrt((torch.square(self.step_state.std_dev_unselect_distence)*(unselect_count + 1) - torch.square(distance_to_current - before_avg_unselect_distence)) / unselect_count)

            feature_num = feature_num + 1
            if self.selected_count == 1:
                self.step_state.data = torch.cat((self.step_state.data, self.step_state.std_dev_unselect_distence.unsqueeze(-1).to(torch.float32)), dim=-1)
            else:
                self.step_state.data[:,:,1 + feature_num] = self.step_state.std_dev_unselect_distence

        if self.selected_count == 1 and self.append_information[3] == True:

            dest_node_idx = self.selected_node_list[:,-1].view(-1)

            dest_node_idx = dest_node_idx.repeat_interleave(self.problem_size)
            index1 = torch.arange(batch_size).repeat_interleave(pomo_size*beam_size*self.problem_size)
            index2 = torch.arange(self.problem_size).repeat(bs)

            distance_to_dest = self.dis_matrix[index1, index2, dest_node_idx].view(bs, self.problem_size, 1)

            self.step_state.data = torch.cat((self.step_state.data, distance_to_dest), dim=-1)


        if self.selected_count == 1 and self.append_information[4] == True:

            dest_node_idx = self.selected_node_list[:,-1].view(-1)

            dest_node_idx_repeat = dest_node_idx.repeat_interleave(self.problem_size)
            index1 = torch.arange(batch_size).repeat_interleave(pomo_size*beam_size*self.problem_size)
            index2 = torch.arange(self.problem_size).repeat(bs)

            distance_to_dest = self.dis_matrix[index1, index2, dest_node_idx_repeat].view(bs, self.problem_size, 1)

            # print("range_index: shape", range_index.shape)
            # print("self.step_state.data: shape", self.step_state.data.shape)
            # print("dest_node_idx: shape", dest_node_idx.shape)

            last_node = self.step_state.data[range_index,dest_node_idx,:2]

            # print("last_node[:,1].unsqueeze(-1).repeat(1, self.problem_size)", last_node[:,1].unsqueeze(-1).repeat(1, self.problem_size))
            # print("self.step_state.data[:,:,1]: shape", self.step_state.data[:,:,1])
            delta_y = last_node[:,1].unsqueeze(-1).repeat(1, self.problem_size) - self.step_state.data[:,:,1]

            # print("delta_y", delta_y)

            sin_to_dest = delta_y.unsqueeze(-1) / (distance_to_dest + 1e-7)

            # sin_to_dest = (sin_to_dest + 1) / 2

            self.step_state.data = torch.cat((self.step_state.data, sin_to_dest), dim=-1)

        if self.selected_count == 1 and self.append_information[5] == True:

            dest_node_idx = self.selected_node_list[:,-1].view(-1)

            dest_node_idx_repeat = dest_node_idx.repeat_interleave(self.problem_size)
            index1 = torch.arange(batch_size).repeat_interleave(pomo_size*beam_size*self.problem_size)
            index2 = torch.arange(self.problem_size).repeat(bs)

            distance_to_dest = self.dis_matrix[index1, index2, dest_node_idx_repeat].view(bs, self.problem_size, 1)

            last_node = self.step_state.data[range_index,dest_node_idx,:2]
            delta_x = last_node[:,0].unsqueeze(-1).repeat(1, self.problem_size) - self.step_state.data[:,:,0]

            cos_to_dest = delta_x.unsqueeze(-1) / (distance_to_dest + 1e-7)

            # cos_to_dest = (cos_to_dest + 1) / 2

            self.step_state.data = torch.cat((self.step_state.data, cos_to_dest), dim=-1)

        if self.selected_count == 1 and self.append_information[6] == True:

            avg_distence_all = torch.mean(self.dis_matrix, dim=-1).repeat_interleave(pomo_size*beam_size, dim=0)

            self.step_state.data = torch.cat((self.step_state.data, avg_distence_all.unsqueeze(-1)), dim=-1)

        if self.selected_count == 1 and self.append_information[7] == True:

            std_dev_distence_all = torch.std(self.dis_matrix, unbiased=False, dim=-1).repeat_interleave(pomo_size*beam_size, dim=0)

            self.step_state.data = torch.cat((self.step_state.data, std_dev_distence_all.unsqueeze(-1)), dim=-1)


        if self.selected_count == 1 and self.append_information[9] == True:

            avg_distence_aggregation = torch.mean(self.dis_matrix.topk(k=self.aggregation_nums, dim=-1, largest=False)[0], dim=-1).repeat_interleave(pomo_size*beam_size, dim=0)

            self.step_state.data = torch.cat((self.step_state.data, avg_distence_aggregation.unsqueeze(-1)), dim=-1)

        if self.selected_count == 1 and self.append_information[10] == True:

            std_dev_distence_aggregation = torch.std(self.dis_matrix.topk(k=self.aggregation_nums, dim=-1, largest=False)[0], unbiased=False, dim=-1).repeat_interleave(pomo_size*beam_size, dim=0)

            self.step_state.data = torch.cat((self.step_state.data, std_dev_distence_aggregation.unsqueeze(-1)), dim=-1)


        torch.cuda.empty_cache()

        done = (self.selected_count == self.problems.shape[1])

        if done:
            if self.mode == "test" and self.env_params['test_mode'] == "aug_test":
                reward = self._get_travel_distance(self.problems[:,:,:2].repeat_interleave(self.beam_size, dim=0))
            else:
                reward = self._get_travel_distance(self.step_state.data[:,:,:2])
            # self.drawPic(self.problems, self.selected_node_list,name="1",optimal_tour_=None, index=1)
        else:
            reward = None, None

        return self.step_state, reward, done

    def make_dir(self,path_destination):
        isExists = os.path.exists(path_destination)
        if not isExists:
            os.makedirs(path_destination)
        return

    def drawPic(self, arr_, tour_, name='xx',optimal_tour_=None,index=None):
        arr = arr_[index].clone().cpu().numpy()
        tour =  tour_[self.beam_size*index].clone().cpu().numpy()
        arr_max = np.max(arr)
        arr_min = np.min(arr)
        arr = (arr -arr_min) / (arr_max - arr_min)

        fig, ax = plt.subplots(figsize=(20, 20 ))

        plt.scatter(arr[:, 0], arr[:, 1], color='black', linewidth=1)

        plt.axis('off')

        start = [arr[tour[0], 0], arr[tour[-1], 0]]
        end = [arr[tour[0], 1], arr[tour[-1], 1]]
        plt.plot(start, end, color='red', linewidth=2, )


        for i in range(len(tour) - 1):
            tour = np.array(tour, dtype=int)
            start = [arr[tour[i], 0], arr[tour[i + 1], 0]]
            end = [arr[tour[i], 1], arr[tour[i + 1], 1]]
            plt.plot(start,end,color='red',linewidth=2)

        b = os.path.abspath(".")
        path = b+'/figure'
        self.make_dir(path)
        plt.savefig(path+f'/{name}.pdf',bbox_inches='tight', pad_inches=0)


    def _get_travel_distance(self, instances):
        # trained model's distance
        gathering_index = self.selected_node_list.unsqueeze(2).expand(-1, self.problems.shape[1], 2)
        ordered_seq = instances.gather(dim=1, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2)
        segment_lengths = segment_lengths.sum(2).sqrt()
        # shape: (batch,problem)
        travel_distances = segment_lengths.sum(1)
        # shape: (batch)
        return travel_distances


