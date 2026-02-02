import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_cluster import knn
import numpy as np
import matplotlib.pyplot as plt
import os


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

import os


class CVRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.mode = model_params['mode']
        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)
        self.encoded_nodes = None
        self.append_information = torch.tensor(model_params['append_information'])
        self.features_nums = torch.sum(self.append_information) + 4

        if self.append_information[8] == True:
            self.features_nums = self.features_nums -1


    def pre_forward(self, dis_matrix, batch_size):
        self.dis_matrix = dis_matrix
        self.batch_size = batch_size

    def get_knn_state(self, state, selected_node_list, knn_nums, depot_knn_nums):

        batch_size = state.data.shape[0]  # B
        problem_size = state.data.shape[1]

        new_list = torch.arange(problem_size)[None, :].repeat(batch_size, 1)

        new_list_len = problem_size - selected_node_list.shape[1] - 1  # shape: [B, V-current_step]

        # mask selected nodes
        index_2 = selected_node_list.type(torch.long)
        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, index_2.shape[1])
        new_list[index_1, index_2] = -2
        index_1 = torch.arange(batch_size, dtype=torch.long)
        new_list[index_1, 0] = -2

        unselect_list = new_list[torch.gt(new_list, -1)].view(batch_size, new_list_len)

        # print("unselect_list:", unselect_list[-9].tolist())
        
        index_1_unselect = torch.arange(0,batch_size).repeat_interleave(new_list_len)

        index_2 = unselect_list.view((-1,))

        unselect_list_node = state.data[:,:,:2][index_1_unselect,index_2].view((-1,2))

        last_node_index = selected_node_list[:, -1]
    
        index_1_last = torch.arange(0,batch_size)

        last_node = state.data[index_1_last,last_node_index,:2]


        depot_node = state.data[index_1_last,0,:2]

        last_node_all = torch.cat((last_node[:,None,:], depot_node[:,None,:]), dim=1).view(-1,2)

        min_knn_nums = min(depot_knn_nums, new_list_len)
        # index_1_last = index_1_last.repeat(2)

        # knn_output = knn(unselect_list_node, last_node_all, min_knn_nums, index_1_unselect, index_1_last)

        # last_output = knn(unselect_list_node, last_node, min_knn_nums, index_1_unselect, index_1_last)
        # depot_output = knn(unselect_list_node, depot_node, min_knn_nums, index_1_unselect, index_1_last)


        last_node = selected_node_list[:,-1] 

        last_node = last_node.view(self.batch_size,batch_size//self.batch_size,1).expand(-1,-1, problem_size)

        depot_node = torch.zeros_like(last_node)

        distance_last = self.dis_matrix.gather(1, last_node).view(batch_size, -1)

        distance_depot = self.dis_matrix.gather(1, depot_node).view(batch_size, -1)

        mask = torch.zeros_like(distance_last)

        distance = torch.cat((distance_last, distance_depot), dim=0)


        index_2 = selected_node_list.type(torch.long)
        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, index_2.shape[1])
        mask[index_1, index_2] = 1e2
        index_1 = torch.arange(batch_size, dtype=torch.long)
        mask[index_1, 0] = 1e2
        # mask[index_1 ,selected_node_list] = 1e2
        # mask[: ,0] = 1e2
        mask = mask.repeat(2, 1)

        unselect_list = torch.argsort(distance + mask,dim=1)[:,:min_knn_nums]

        last_knn_origin_idx = unselect_list[:batch_size,:min(knn_nums,min_knn_nums)].sort(dim=-1)[0]

        depot_knn_origin_idx = unselect_list[batch_size:,:].sort(dim=-1)[0]


        # index_1_last = index_1_last.repeat_interleave(2)

        # output = knn(unselect_list_node, last_node_all, min_knn_nums, index_1_unselect, index_1_last)

        # knn_idx = output[1,:] % new_list_len

        # # depot_knn_idx = depot_output[1,:] % new_list_len

        # # knn_idx = knn_idx.view((batch_size*2, min_knn_nums)).contiguous()

        # last_knn_idx = knn_idx.view((batch_size, min_knn_nums*2))[:, :min(knn_nums,min_knn_nums)].contiguous()

        # depot_knn_idx = knn_idx.view((batch_size, min_knn_nums*2))[:, min_knn_nums:].contiguous()

        # last_knn_idx = last_knn_idx[:,:min(knn_nums,min_knn_nums)]

        # depot_knn_idx = depot_knn_idx[:,:]

        # last_knn_origin_idx = unselect_list.gather(1, last_knn_idx)

        # last_knn_origin_idx = last_knn_origin_idx.sort(dim=-1)[0]

        # depot_knn_origin_idx = unselect_list.gather(1, depot_knn_idx)

        # depot_knn_origin_idx = depot_knn_origin_idx.sort(dim=-1)[0]

        # ----------------------------------------------------------------------------

        emb_dim = state.data.shape[-1]

        last_knn_node = state.data.gather(1, last_knn_origin_idx.unsqueeze(-1).repeat(1,1,emb_dim)).contiguous()

        depot_knn_node = state.data.gather(1, depot_knn_origin_idx.unsqueeze(-1).repeat(1,1,emb_dim)).contiguous()

        knn_node_ninf_mask = state.ninf_mask.gather(1, last_knn_origin_idx).contiguous()

        capacity = state.capacity

        capacity[capacity < 0.00001] = capacity[capacity < 0.00001] + 0.0000001

        index = (capacity < 0.00001)

        capacity = capacity.unsqueeze(-1)

        last_knn_node[~index,:,2] = last_knn_node[~index,:,2] / capacity[~index]

        last_knn_node[index,:,2] = 1.1

        depot_knn_node[:,:,2] = depot_knn_node[:,:,2]

        # print("last_knn_node[:,:,2]", )

        if self.append_information[0] == True:
            knn_dis_to_depot = state.distance_to_depot.gather(1, depot_knn_origin_idx).contiguous()

            depot_knn_node[:, :, 3] = knn_dis_to_depot

        # print("last_knn_origin_idx", last_knn_origin_idx[-9].tolist())

        return last_knn_node, depot_knn_node, last_knn_origin_idx, depot_knn_origin_idx, min_knn_nums, knn_node_ninf_mask, 

    def drawPic(self, arr_, tour_, flag_,name='xx',optimal_tour_=None,index=None , knn_node_=None):
        arr = arr_[index].clone().cpu().numpy()
        tour =  tour_[index].clone().cpu().numpy()
        flag =  flag_[index].clone().cpu().numpy()
        if knn_node_ != None:
            knn_node =  knn_node_[index].clone().cpu().numpy()
        arr_max = np.max(arr)
        arr_min = np.min(arr)
        arr = (arr -arr_min) / (arr_max - arr_min)

        if knn_node_ != None:
            knn_node = (knn_node -arr_min) / (arr_max - arr_min)

        fig, ax = plt.subplots(figsize=(20, 20 ))

        plt.scatter(arr[:, 0], arr[:, 1], color='black', linewidth=1)

        plt.scatter(arr[0, 0], arr[0, 1], color='green', linewidth=1)

        if knn_node_ != None:
            plt.scatter(knn_node[:, 0], knn_node[:, 1], color='blue', linewidth=1)

        plt.axis('off')

        # start = [arr[tour[0], 0], arr[tour[-1], 0]]
        # end = [arr[tour[0], 1], arr[tour[-1], 1]]
        # plt.plot(start, end, color='red', linewidth=2, )
        print("flag", flag)

        for i in range(len(tour) - 1):
            tour = np.array(tour, dtype=int)
            
            if flag[i+1] == 1:
                x = [arr[tour[i], 0], arr[0, 0], arr[tour[i + 1], 0]]
                y = [arr[tour[i], 1], arr[0, 1], arr[tour[i + 1], 1]]
                plt.plot(x,y,color='red',linewidth=2)
            else:
                x = [arr[tour[i], 0], arr[tour[i + 1], 0]]
                y = [arr[tour[i], 1], arr[tour[i + 1], 1]]
                plt.plot(x,y,color='red',linewidth=2)

        b = os.path.abspath(".")
        path = b+'/figure'
        self.make_dir(path)
        plt.savefig(path+f'/{name}.pdf',bbox_inches='tight', pad_inches=0)

    def make_dir(self,path_destination):
        isExists = os.path.exists(path_destination)
        if not isExists:
            os.makedirs(path_destination)
        return
    

    def _get_node(self, state, node_index_to_pick):

        batch_size = node_index_to_pick.size(0)
        node_len = node_index_to_pick.size(1)
            
        gathering_index = node_index_to_pick[:, :, None].expand(batch_size, node_len, self.features_nums-1)

        picked_nodes = state.data.gather(dim=1, index=gathering_index)

        return picked_nodes
    
    def _clip(self, u):
        v = torch.zeros_like(u)
        w = torch.ones_like(u)
        return torch.max(v, torch.min(u, w))

    def _norm_node(self, last_knn_node):

        tmp = last_knn_node[:,:,2]

        max_val = last_knn_node.max(dim=1)[0] - last_knn_node.min(dim=1)[0]
        min_val = last_knn_node.min(dim=1)[0]

        max_val = max_val.unsqueeze(1).repeat(1, last_knn_node.shape[1], 1)
        min_val = min_val.unsqueeze(1).repeat(1, last_knn_node.shape[1], 1)

        index = max_val != 0

        last_knn_node[index] = (last_knn_node[index] - min_val[index]) / max_val[index]

        last_knn_node[~index] = 0

        last_knn_node[:,:,2] = tmp

        return last_knn_node
    
    def forward(self, state, selected_node_list, solution, solution_flag, knn, current_step, depot_knn_nums, selected_flag = None):

        batch_size = state.data.shape[0]
        problem_size = state.data.shape[1] - 1

        last_knn_node, depot_knn_node, last_unselect_list, depot_unselect_list, knn_nums, knn_node_ninf_mask = self.get_knn_state(state, selected_node_list, knn, depot_knn_nums)

        last_node = self._get_node(state, selected_node_list[:,[-1]])

        depot_node = self._get_node(state, torch.zeros(batch_size,1).to(torch.int64))

        # print("current_step: ", current_step)
        # self.drawPic(state.data[:,:,:2], selected_node_list, selected_flag, name='knn-{}'.format(current_step), optimal_tour_=None,index=3 , knn_node_=last_knn_node[:,:,:2])
        # if self.mode == "train":
        #     self.drawPic(state.data[:,:,:2], solution, name='knn-opt-{}'.format(current_step), optimal_tour_=None,index=3 , knn_node_=None)
        # print("knn_node", knn_node.shape)
        # print("last_node", last_node.shape)

        last_knn_node = torch.cat((last_knn_node, last_node), dim=1)
        depot_knn_node = torch.cat((depot_knn_node, depot_node), dim=1)

        norm_last_knn_node = self._norm_node(last_knn_node)

        norm_depot_knn_node = self._norm_node(depot_knn_node)

        norm_last_knn_node = torch.cat((norm_last_knn_node,torch.zeros(batch_size, norm_last_knn_node.shape[1], 1)), dim=-1)
        norm_depot_knn_node = torch.cat((norm_depot_knn_node,torch.ones(batch_size, norm_depot_knn_node.shape[1], 1)), dim=-1)

        # print("norm_last_knn_node", norm_last_knn_node.shape)
        # print("norm_depot_knn_node", norm_depot_knn_node.shape)
        
        norm_knn_node = torch.cat((norm_last_knn_node, norm_depot_knn_node), dim=1)

        if self.mode == 'train':
            # exit(0)

            probs = self.decoder(self.encoder(norm_knn_node), last_unselect_list, depot_unselect_list, problem_size, state.capacity, knn_node_ninf_mask=knn_node_ninf_mask)
            
            # print("solution[:, current_step - 1]", solution[:, current_step - 1])
            selected_teacher = solution[:, current_step]  # shape: B
            # print("current_step", current_step)
            # print("selected_teacher", selected_teacher)
            # print("selected_teacher", selected_teacher)
            solution_flag_teacher = solution_flag[:, current_step]
            # print("solution_flag_teacher", solution_flag_teacher)

            # torch.set_printoptions(threshold=1000000)
            # print("last_unselect_list", last_unselect_list)
            # print("selected_node_list",selected_node_list)
            # print("solution", solution)
            is_via_depot = solution_flag_teacher==1
            selected_teacher[is_via_depot] += (problem_size + 1)

            # print("selected_teacher", selected_teacher)

            # print("probs", probs)
            prob = probs[torch.arange(batch_size)[:, None], selected_teacher[:, None]].reshape(batch_size, 1)  # shape: [B, 1]
            

            # print("last_unselect_list", last_unselect_list)
            # print("depot_unselect_list", depot_unselect_list)
            # # print("solution", solution)
            # # print("solution_flag", solution_flag)
            # if current_step == 9:
            #     exit(0)
            # print("probs", torch.sum(prob==1e-20))

            return selected_teacher, prob, 1

        
        with torch.no_grad():
            if self.mode == 'valid' or self.mode == 'test':

                probs = self.decoder(self.encoder(norm_knn_node), last_unselect_list, depot_unselect_list, problem_size, state.capacity, knn_node_ninf_mask=knn_node_ninf_mask)

                # print("probs", probs[-11,33])

                # torch.set_printoptions(threshold=1000000)
                # print("probs", probs[:10])
                # torch.set_printoptions(threshold=None)
                # print("selected_node_list", selected_node_list)

                # print("last_unselect_list", last_unselect_list)
                # print("depot_unselect_list", depot_unselect_list)
                # if current_step == 9:
                #     exit(0)
                # if current_step == 1:
                #     print("last_unselect_list", last_unselect_list)
                
            
                return None, probs, 1


        return selected_teacher, prob, 1



########################################
# ENCODER
########################################
class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = 1

        self.append_information = model_params['append_information']
        self.features_nums = torch.sum(torch.tensor(self.append_information)) + 4

        if self.append_information[8] == True:
            self.features_nums = self.features_nums - 1
        
        self.embedding = nn.Linear(self.features_nums, embedding_dim, bias=True)

        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])



    def forward(self, data):

        embedded_input = self.embedding(data)
        out = embedded_input
        for layer in self.layers:
            out = layer(out)
        
        return out


class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['decoder_layer_num']
        self.append_information = model_params['append_information']

        if self.append_information[8] == True:
            self.embedding_last_node = nn.Linear(embedding_dim + 2, embedding_dim, bias=True)
        else:
            self.embedding_last_node_not_via_depot = nn.Linear(embedding_dim, embedding_dim, bias=True)
            self.embedding_last_node_via_depot = nn.Linear(embedding_dim, embedding_dim, bias=True)
        
        self.embedding_last_node_pos = nn.Linear(embedding_dim, embedding_dim, bias=True)

        self.layers = nn.ModuleList([DecoderLayer(**model_params) for _ in range(encoder_layer_num)])

        self.k_1 = nn.Linear(embedding_dim, embedding_dim, bias=True)

        self.Linear_final = nn.Linear(embedding_dim, 1, bias=True)
    def forward(self,embedded_norm_last_knn_node, last_unselect_list, depot_unselect_list, problem_size, capacity,first_node = None, knn_node_ninf_mask=None):

        
        batch_size = embedded_norm_last_knn_node.shape[0]  # B
        last_knn_size = last_unselect_list.shape[-1]
        depot_knn_size = depot_unselect_list.shape[-1]


        embedded_knn_node_not_via_depot = embedded_norm_last_knn_node[:,0:last_knn_size]
        embedded_last_node_not_via_depot = embedded_norm_last_knn_node[:,last_knn_size]

        embedded_knn_node_via_depot = embedded_norm_last_knn_node[:,last_knn_size+1:-1]
        embedded_last_node_via_depot = embedded_norm_last_knn_node[:,-1]

        # embedded_last_node_not_via_depot = self.embedding_last_node_not_via_depot(torch.cat((embedded_last_node_not_via_depot, capacity.unsqueeze(1)), dim=-1))
        # embedded_last_node_via_depot = self.embedding_last_node_via_depot(torch.cat((embedded_last_node_via_depot, capacity.unsqueeze(1)), dim=-1))

        embedded_last_node_not_via_depot = self.embedding_last_node_not_via_depot(embedded_last_node_not_via_depot)
        embedded_last_node_via_depot = self.embedding_last_node_via_depot(embedded_last_node_via_depot)

        # out = torch.cat((embedded_knn_node_, embedded_last_node_.unsqueeze(1)), dim=1)
        out = torch.cat((embedded_last_node_not_via_depot.unsqueeze(1),embedded_knn_node_not_via_depot, embedded_last_node_via_depot.unsqueeze(1),embedded_knn_node_via_depot), dim=1)

        layer_count=0

        for layer in self.layers:

            out = layer(out)
            layer_count += 1

        out = self.Linear_final(out).squeeze(-1)

        # split = last_knn_size + 1

        depot_unselect_list = depot_unselect_list + problem_size + 1

        # if last_unselect_list.shape[1] == 4:
        #     print("last_unselect_list", last_unselect_list)
        # print("depot_unselect_list", depot_unselect_list)

        out[:, [0, last_knn_size + 1]] = out[:, [0, last_knn_size + 1]] + float('-inf')

        # print("out.shape:", out[:, 1:last_knn_size].shape)
        # print("knn_node_ninf_mask", knn_node_ninf_mask.shape)

        # rows_all_inf = torch.all(torch.isinf(knn_node_ninf_mask), dim=1)

        out[:, 1:last_knn_size + 1] = out[:, 1:last_knn_size + 1] + knn_node_ninf_mask

        # print("rows_all_inf:", rows_all_inf)
        # print("rows_all_inf:", rows_all_inf.shape)

        # out[~rows_all_inf, (last_knn_size + 1):] = out[~rows_all_inf, (last_knn_size + 1):] + float('-inf')

        props = F.softmax(out, dim=-1)

        # print("props:", props)

        props = torch.cat((props[:, 1:last_knn_size + 1],props[:, last_knn_size + 2:]),dim=1)

        # print("props:", props.shape)

        index_small = torch.le(props, 1e-5)
        props_clone = props.clone()
        props_clone[index_small] = props_clone[index_small] + torch.tensor(1e-7, dtype=props_clone[index_small].dtype)  # prevent the probability from being too small
        props = props_clone

        new_props = torch.zeros(batch_size, (problem_size + 1)*2)

        # print("================================problem_size=============================")
        # print(problem_size)

        index_1_ = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, last_unselect_list.shape[1])  # shape: [B*(V-1), n]

        new_props[index_1_, last_unselect_list] = 1e6

        index_1_ = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, depot_unselect_list.shape[1])  # shape: [B*(V-1), n]

        new_props[index_1_, depot_unselect_list] = 1e6

        index = torch.gt(new_props, 1e5).view(batch_size, -1)

        # print("=========index new_props", new_props)

        # print("props", props.shape)

        new_props[index] = props.ravel()

        new_props[~index] = 1e-20

        # torch.set_printoptions(threshold=1000000)
        # print("last_unselect_list", last_unselect_list)
        # print("depot_unselect_list", depot_unselect_list)
        # print("new_props", new_props)

        return new_props

class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.feedForward = Feed_Forward_Module(**model_params)


    def forward(self, input1):

        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)

        multi_head_out = self.multi_head_combine(out_concat)

        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 +  out2
        return out3


class DecoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.feedForward = Feed_Forward_Module(**model_params)

    # @time_count
    def forward(self, input1):

        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)

        multi_head_out = self.multi_head_combine(out_concat)

        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 +  out2
        return out3


def reshape_by_heads(qkv, head_num):

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)

    q_transposed = q_reshaped.transpose(1, 2)

    return q_transposed


def multi_head_attention(q, k, v):

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))  # shape: (B, head_num, n, n)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    weights = nn.Softmax(dim=3)(score_scaled)  # shape: (B, head_num, n, n)

    out = torch.matmul(weights, v)  # shape: (B, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)  # shape: (B, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)  # shape: (B, n, head_num*key_dim)

    return out_concat



class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
