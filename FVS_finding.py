# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 11:04:44 2021

@author: jwKim
"""
#tuple form 의 link로 이루어진 정보를 FVS_finding에 시작할 때 넣어준다.
#그 후 set_SCC_decomposition_function method를 사용하여 SCC decomposition에 사용할 함수를 선택해준다.
#그 후 set_FVS_finding_strategy method 를 사용하여 각 SCC에서 FVS를 찾는 방법을 선택해준다. 이 방법의 예시는 _default_selection method를 참고할 것.
#그 후 FVS_finding 객체의 find_FVS method 를 수행해 준다. 위의 2단계는 기본 값이 입력되어 있으므로 생략 가능.

import functools
import operator
import itertools
import math
import random
import pickle
import numpy as np

from SCC_decomposition import SCC_decomposition

class FVS_finding:
    def __init__(self, iter_t_links):
        """iter_t_links 는 각 link를 tuple형태로 표현한 반복형을 받는다.
        여기서 link t의 source node는 t[0], target node는 t[-1]이어야 한다."""
        self.iter_t_links = iter_t_links
        self.t_nodes = tuple([])
        self._get_list_of_nodes()
        self.l_t_indexform_links = []
        self._convert_node_to_index_form()
        
        
        self._function_SCC_decomposition = SCC_decomposition#default function
        self._select_FVS_finding_strategy = self._default_selection
        
        self.l_iter_t_FVSs = []
        
    
    def _get_list_of_nodes(self):
        """self.l_t_links에 있는 link로부터 network를 이루는 nodes를 중복없이 추출한 후, tuple form으로 return한다."""
        l_nodes = []
        for t_link in self.iter_t_links:
            if t_link[0] not in l_nodes:
                l_nodes.append(t_link[0])
            if t_link[-1] not in l_nodes:
                l_nodes.append(t_link[-1])
        
        self.t_nodes = tuple(l_nodes)
    
    def _take_self_loop_nodes_as_FVS(self):
        """self loop 를 가진 nodes 를 찾아서 FVS에 포함시킨 뒤, links에서 그 nodes와 관련된 link를 제거한다."""
        l_self_loop_nodes_index_form = []
        for index_link in range(len(self.l_t_indexform_links)-1,-1,-1):
            t_link = self.l_t_indexform_links[index_link]
            if t_link[0] == t_link[-1]:
                l_self_loop_nodes_index_form.append(t_link[0])
                self.l_t_indexform_links.pop(index_link)
        
        set_self_loos_nodes = set(l_self_loop_nodes_index_form)
        for index_link in range(len(self.l_t_indexform_links)-1,-1,-1):
            t_link = self.l_t_indexform_links[index_link]
            if t_link[0] in set_self_loos_nodes or t_link[-1] in set_self_loos_nodes:
                self.l_t_indexform_links.pop(index_link)
        
        self.l_iter_t_FVSs.append((tuple(l_self_loop_nodes_index_form),))
    
    def _convert_node_to_index_form(self):
        """(node,,,,node) 형태로 된 links에서 node부분을 l_nodes 의 index 로 대체하여 새로운 links list 를 작성한다.
        이렇게 하면 node의 type에 상관 없이 module을 자유롭게 바꿀 수 있음."""
        for t_link in self.iter_t_links:
            t_link_index_form = (self.t_nodes.index(t_link[0]), 
                                 self.t_nodes.index(t_link[-1]))
            self.l_t_indexform_links.append(t_link_index_form)
    
    def set_SCC_decomposition_function(self, fuction_SCC_decomposition):
        """FVS finding 객체에서 사용할 SCC decomposition 함수를 설정한다.
        SCC_decomposition 함수는 tuple 형태의 link들의 리스트를 받는다.
        그리고 그 네트워크를 SCC decompose 한 후 각 쪼개진 SCC에 속하는 nodes를 list에 모으고,
        그 SCC가 담긴 list를 다시 list로 묶어서 return해줘야 한다. 
        즉 return [list of SCC1, list of SCC2,,,,]"""
        self._function_SCC_decomposition = fuction_SCC_decomposition
    
    def set_FVS_finding_strategy(self, function_select_strategy_of_FVS_finding):
        """주어진 SCC에서 FVS를 찾는 객체를 정하는 함수를 설정한다.
        설정해주는 함수는 다음 조건을 만족해야 한다.
        
        이 함수는 SCC를 이루는 nodes와 그 SCC에 존재하는 links를 argument로 받는다. 
        이 정보를 사용해서 어떠한 방법론이 이 SCC에서 FVS를 찾는데 최적일지 결정한다.
        
        default는 node 개수가 15개 이하이면 full search로 찾고 그 이상이면 SA_FVSP_NNS로 찾는 것.
        결정되면 이 FVS 탐색을 수행하는 객체를 return한다.
        
        이 객체는 set_links로 links 정보를 받고, set_nodes로 nodes 정보를 받은 뒤,
        calculate_FVSs method를 수행하여 가능한 FVS 조합을 return해야 한다.
        즉 return [(FVS가 될 수 있는 nodes 조합1), (FVS가 될 수 있는 nodes 조합2),,,]
        find_FVS_in_SCC method를 참고할 것.
        
        이 함수가 l_SCC에 따라서 어떤 방법론을 쓸 것인지 runtime 도중 멈추고
        방법론 입력받도록 만들 수도 있다. """
        self._select_FVS_finding_strategy = function_select_strategy_of_FVS_finding
    
    @staticmethod
    def _default_selection(l_SCC, l_t_indexform_links_in_SCC):
        """15개 이하의 SCC에 대해서는 full search 방법으로 모든 가능한 FVS 조합을 찾는다.
        16개 이상의 SCC에 대해서는 SA_FVSP_NNS 방법으로 가능한 FVS 조합을 찾는다."""
        if len(l_SCC) <= 15:
            return FVS_brutal_force_searching()
        else:
            return SA_FVSP_NNS(i_max_move=50*len(l_SCC))###### parameter 넣어줘야 함.#############################
    
    def _find_FVS_in_SCC(self, l_SCC, l_t_indexform_links):
        """network의 SCC에 대해, 그 SCC 내부의 links 만 추출한 뒤, SCC network 정보를
        _select_FVS_finding_strategy 메소드에 넣어서 FVS 계산 전략을 세운다.
        이 객체에 network 정보를 넣은 뒤, 계산을 수행하고, 
        가능한 FVS 조합을 iterable form 으로 return 한다."""
        l_t_indexform_links_in_SCC = [t_link for t_link in l_t_indexform_links 
                                      if set((t_link[0],t_link[-1])).issubset(l_SCC)]
        print("FVS of ",len(l_SCC)," nodes SCC is calculating")
        #print(l_t_indexform_links_in_SCC)
        FVS_finding_object = self._select_FVS_finding_strategy(l_SCC, l_t_indexform_links_in_SCC)
        FVS_finding_object.set_nodes(l_SCC)
        FVS_finding_object.set_links(l_t_indexform_links_in_SCC)
        iter_t_FVS = FVS_finding_object.calculate_FVSs()
        return iter_t_FVS
    
    def _combine_all_FVSs(self):
        """self.l_iter_t_FVSs 가 있을 때, 이 리스트에 포함된 각 iter_t_FVSs에서 t_FVS를 하나씩 골라서 합친다.
        그렇게 만들 수 있는 모든 조합들을 list로 정리하여 return한다.
        [iter((1,2),(3,4)), iter((5,6),(7,8,9))] 가 self.l_iter_t_FVSs일 경우,
        [(1,2,5,6), (1,2,7,8,9), (3,4,5,6),(3,4,7,8,9)] 가 return 된다."""
        l_t_FVS = []
        t_iter_t_FVSs = tuple(self.l_iter_t_FVSs)
        for combination_t_FVS in itertools.product(*t_iter_t_FVSs):
            t_FVS = functools.reduce(operator.add, combination_t_FVS, tuple([]))
            l_t_FVS.append(t_FVS)
        
        return l_t_FVS
        
    
    def _convert_index_form_FVSs_to_original_form_FVS(self, l_t_FVSs):
        """l_t_FVSs는 리스트 안에 tuple이 있고, tuple은 FVS가 될 수 있는 nodes의 index를 담고 있다.
        이 index 형태의 정보를 t_nodes에 담긴 data로 복구해준다."""
        l_t_FVSs_original_form = [tuple((self.t_nodes[i] for i in t_FVS)) for t_FVS in l_t_FVSs]
        return l_t_FVSs_original_form
        
    
    def find_FVS(self):
        self._take_self_loop_nodes_as_FVS()
        l_l_SCC = self._function_SCC_decomposition(self.l_t_indexform_links)
        l_l_SCC = [l_SCC for l_SCC in l_l_SCC if len(l_SCC) >=2]#for SCC with one node, no FVS exist
        
        for l_SCC in l_l_SCC: #this part can calculated using multiprocess. order of calculation is not important
            iter_t_FVS = self._find_FVS_in_SCC(l_SCC, self.l_t_indexform_links)
            self.l_iter_t_FVSs.append(iter_t_FVS)
        
        l_t_FVS = self._combine_all_FVSs()
        l_t_FVSs_original_form = self._convert_index_form_FVSs_to_original_form_FVS(l_t_FVS)
        self.check_FVS(l_t_FVSs_original_form)###################### for test this module
        return l_t_FVSs_original_form
    
    def check_FVS(self, l_t_FVSs_original_form):
        for t_FVS in l_t_FVSs_original_form:
            l_t_links = []
            for t_link in self.iter_t_links:
                if (t_link[0] not in t_FVS) and (t_link[-1] not in t_FVS):
                    l_t_links.append((t_link[0],t_link[-1]))
            if not l_t_links:#in this case, t_FVS is FVS
                continue
            l_l_SCC = SCC_decomposition(l_t_links)
            if max(map(len, l_l_SCC)) >1:
                print(t_FVS, " is not FVS! check it again")

        
                
class SA_FVSP_NNS:
    """
    Tang, Zhipeng, Qilong Feng, and Ping Zhong. "Nonuniform neighborhood sampling based simulated annealing for the directed feedback vertex set problem." IEEE Access 5 (2017): 12353-12363.
    논문 내용을 기반으로 하여 약간의 알고리즘 수정을 하여 사용함. 
    
    random.choices 는 python 3.6 이상에만 존재함.
    3.6을 설치하던가 아니면 random.choices에 해당하는 함수를 만들어서 추가할 것."""
    def __init__(self, i_max_move, i_max_fail=30, float_temporature_initial=0.8, float_temperature_decrease=0.99):
        """i_max_move 가 클수록, i_max_fail 이 많을수록 더 많은 계산을 하고 정확도가 올라간다.
        float_temporature_initial 이 크면 클수록 local minimum 을 빠져나갈 가능성이 높아진다.
        float_temperature_decrease는 1보다 작은 양의 실수여야 하며, 이 값이 작을수록 temperature 가 빠르게 감소. 
        즉 local minimum을 빠져나가는 초기 단계를 빨리 지나간다."""
        self.i_max_move = i_max_move
        self.i_max_fail = i_max_fail
        self.i_num_fail = 0
        self.float_temperature = float_temporature_initial
        self.float_temperature_decrease = float_temperature_decrease# 0 < float_temperature_decrease < 1
        
        self.t_nodes = []
        self.dict_index_node_set_regulator_indexes = {}
        self.dict_index_node_set_target_indexes = {}
        
        self.l_seq = []
        self.set_nodes_in_seq = set()#==set(self.l_seq) 계산 속도를 위해 self.l_seq과 함께 업데이트 하기.
        self.set_t_seq_maximals = set()
        self.i_len_maximal_seq = 0
        
        self.dict_index_sampling_ratio = {}
        self.max_passing_failure = 0
    
    def set_nodes(self, iter_nodes):
        self.t_nodes = tuple((node for node in iter_nodes))
        
        self.max_passing_failure = len(self.t_nodes) * 100
        #일단 이렇게 정해 놓는다.
    
    def set_links(self, iter_t_links):
        for t_link in iter_t_links:
            index_regulator = self.t_nodes.index(t_link[0])
            index_target = self.t_nodes.index(t_link[-1])
            self.dict_index_node_set_regulator_indexes.setdefault(index_target, set()).add(index_regulator)
            self.dict_index_node_set_target_indexes.setdefault(index_regulator, set()).add(index_target)
        
        for i_index in range(len(self.t_nodes)):
            #이 dict들이 모든 node 에 대해 값을 가지도록. 함. SCC network를 넣어줬을 때는 안 해도 상관 없지만 보험삼아서.
            self.dict_index_node_set_regulator_indexes.setdefault(index_target, set())
            self.dict_index_node_set_target_indexes.setdefault(index_regulator, set())
    
    class High_quality_Move:
        def __init__(self, index_node, position_type):
            self.index_node = index_node
            self.position_type = position_type#"+" or "-"
            #+라는 것은 이 노드가 이 노드의 target nodes 보다는 seq에서 앞쪽에 위치.
            #-라는 것은 이 노드가 이 노드의 regulator nodes 보다는 seq에서 뒤쪽에 위치
            
            self.i_poition_to_insert = None
            self.set_indexes_to_delete = set()
            self.l_positions_to_delete = []#내림차순으로 정렬해놓을 것.
            #먼저 l_seq의 self.i_poition_to_insert 위치에 self.index_node를 넣은 뒤,
            #l_seq에서 이 self.l_positions_to_delete에 해당하는 positions를 지운다.
        
        def calculate_positions(self, l_seq, 
                                dict_index_node_set_regulator_indexes, 
                                dict_index_node_set_target_indexes):
            if self.position_type == "+":
                set_copied = dict_index_node_set_target_indexes[self.index_node].intersection(l_seq)
                set_to_check = dict_index_node_set_regulator_indexes[self.index_node]
                i_position = len(l_seq)
                while set_copied:
                    for i_position in range(len(l_seq)-1,-1,-1):
                        i_index = l_seq[i_position]
                        set_copied.discard(i_index)
                        if i_index in set_to_check:
                            self.set_indexes_to_delete.add(i_index)
                            self.l_positions_to_delete.append(i_position+1)
                else:
                    self.i_poition_to_insert = i_position
                        
            else:#self.position_type == "-":
                set_copied = dict_index_node_set_regulator_indexes[self.index_node].intersection(l_seq)
                set_to_check = dict_index_node_set_target_indexes[self.index_node]
                i_position = -1
                while set_copied:
                    for i_position, i_index in enumerate(l_seq):
                        set_copied.discard(i_index)
                        if i_index in set_to_check:
                            self.set_indexes_to_delete.add(i_index)
                            self.l_positions_to_delete.append(i_position)
                else:
                    self.i_poition_to_insert = i_position +1
                
                self.l_positions_to_delete.sort(key=lambda x:-x)
        
        def get_score_of_move(self):
            return len(self.l_positions_to_delete) -1
        
        def set_of_indexes_to_delete_in_seq(self):
            return self.set_indexes_to_delete
        
        def __repr__(self):
            return "{} index to {} position".format(self.index_node, self.i_poition_to_insert)
            
        
        
        
    def SA_FVSP_NNS_algorithm(self):
        if not self.dict_index_sampling_ratio:
            dict_index_priority = self.NNS_prirority_function(0.3)
            self.NNS_sampling_function(dict_index_priority, 3)
        #print("i max fail ",self.i_max_fail)
        while self.i_num_fail < self.i_max_fail:
            #print("num of fail", self.i_num_fail)
            #print(self.l_seq)
            i_num_move = 0
            num_move_fail_to_pass_if = 0
            boolean_fail = True
            #print("i max move ",self.i_max_move)
            while (i_num_move < self.i_max_move) and (num_move_fail_to_pass_if < self.max_passing_failure):
                move = self.choose_move()
                #현재의 seq이 매우 optimal 해서 어떤 move 를 골라도 다음의 if 문을 통과하지 못할 경우
                #물론 exponential 항 때문에 통과할 가능성은 언제나 있지만 temperature 가 작게 잡혀서
                #if 문을 통과하기가 매우 어려우면 계산 시간이 매우 길어질 수 있음. 
                #이런 경우를 방지하고자, if 문을 통과하지 못하는 횟수가 얼마 이상으로 늘어나면
                #이 sequence가 충분히 optimal 한 편이라고 하고 넘어가는 코드를
                #내가 임의로 추가하기로 함. num_move_fail_to_pass_if 와  self.max_passing_failure 를 사용.
                
                #print(move)
                i_score_of_move = self.score_function(move)
                #print("i score of move:",i_score_of_move)
                #print("exponential_val ",math.exp(-i_score_of_move/self.float_temperature))
                #print("i num move ", i_num_move)
                if i_score_of_move <=0 or math.exp(-i_score_of_move/self.float_temperature) >= random.uniform(0,1):
                    self.apply_move_to_seq(move)
                    i_num_move += 1
                    num_move_fail_to_pass_if = 0
                    #move가 if 를 pass하면 이것을 초기화 시켜주자.
                    #최대한 원래의 코드에 악영향이 없도록.
                    if len(self.l_seq) > self.i_len_maximal_seq:
                        self.set_t_seq_maximals = set([tuple(sorted(self.l_seq))])
                        self.i_len_maximal_seq = len(self.l_seq)
                        boolean_fail = False
                    elif len(self.l_seq) == self.i_len_maximal_seq:
                        t_tmp_seq = tuple(sorted(self.l_seq))
                        if t_tmp_seq not in self.set_t_seq_maximals:
                            self.set_t_seq_maximals.add(tuple(sorted(self.l_seq)))
                            boolean_fail = False
                else:
                    num_move_fail_to_pass_if += 1
            
            if boolean_fail:#위의 while 문을 돌린 결과 더 향상된 seq을 얻었는가?
                self.i_num_fail += 1
            else:
                self.i_num_fail = 0
            
            self.float_temperature *= self.float_temperature_decrease
        
    
    def choose_move(self):
        """self.dict_index_sampling_ratio 의 score 에 기반해서 node 를 선택한 뒤, 
        그 node 를 끼워넣을 자리를 선택한다. random 요소가 포함됨.
        선택된 node와 자리 정보로 move 객체를 만들어서 return 한다."""
        #use random.choices
        #고민해 볼 것. 매번 seq 에 따라서 random하게 선택 가능한 것을 걸러낼 것인지?
        #아니면 그냥 같은 선택지에서 random하게 뽑은 뒤 이미 seq에 있는 거라면 다시 뽑는 것이 좋을지.
        l_indexes = []
        l_sampling_ratio = []
        for index, sampling_ratio in self.dict_index_sampling_ratio.items():
            if index not in self.set_nodes_in_seq:
                l_indexes.append(index)
                l_sampling_ratio.append(sampling_ratio)
        index_randomly_chosen = random.choices(l_indexes, l_sampling_ratio)[0]
        str_move_type = random.choice(("+","-"))
        
        move = self.High_quality_Move(index_randomly_chosen, str_move_type)
        move.calculate_positions(self.l_seq, self.dict_index_node_set_regulator_indexes, self.dict_index_node_set_target_indexes)
        return move
        
        
    
    def score_function(self, move):
        return move.get_score_of_move()
    
    def get_FVSs_from_set_t_seq(self):
        l_t_FVSs = []
        for t_seq in self.set_t_seq_maximals:
            set_index_of_FVS = set(range(len(self.t_nodes))) - set(t_seq)
            l_t_FVSs.append(tuple(self.t_nodes[i] for i in set_index_of_FVS))
        
        return l_t_FVSs
    
    def apply_move_to_seq(self, move):
        #move 가 가리키는 node를 move가 가리키는 위치에 집어넣는다.
        self.l_seq.insert(move.i_poition_to_insert, move.index_node)
        self.set_nodes_in_seq.add(move.index_node)
        
        #move로 인해 새로 들아간 node와 기존의 nodes 중 seq의 규칙에 맞지 않는 기존의 nodes를 제거한다.
        #제거해야 할 nodes는 move에서 미리 계산해 놓는다.
        for i_to_delete in move.l_positions_to_delete:
            self.l_seq.pop(i_to_delete)
        self.set_nodes_in_seq.difference_update(move.set_of_indexes_to_delete_in_seq())
    
    def calculate_FVSs(self):
        self.SA_FVSP_NNS_algorithm()
        return self.get_FVSs_from_set_t_seq()
    
    def NNS_prirority_function(self, lambda_parameter=0.3):
        """network 의 nodes 에 대해 FVS에 포함되지 않을 경향성을 heuristic하게 계산.
        priority 를 구해서 dict 형태로 return 한다.
        더 높은 priority score 는 move 로 선택했을 때 더 global optimum에 도달할 가능성이 높다는 의미.
        
        이 함수는 network 정보( self.dict_index_node_set_regulator_indexes,self.dict_index_node_set_target_indexes) 가 필요.
        """
        
        dict_index_priority = {}
        
        for i_index in range(len(self.t_nodes)):
            i_num_regulators = len(self.dict_index_node_set_regulator_indexes[i_index])
            i_num_targets = len(self.dict_index_node_set_target_indexes[i_index])
            float_priority = i_num_regulators + i_num_targets - lambda_parameter*abs(i_num_regulators-i_num_targets)
            dict_index_priority[i_index] = float_priority
        
        return dict_index_priority
    
    def NNS_sampling_function(self, dict_index_priority, i_num_of_vertices_in_segmentation_group=3):
        """각 node의 priority 정보를 사용하여 비슷한 priority 끼리 그룹을 만든다.
        i_num_of_vertices_in_segmentation_group 가 클 수록 하나의 그룹에 들어가는 node 수가 많아진다.
        각 node는 그 node가 속한 group 의 score 에 따라 선택될 확률이 증가한다.
        self.dict_index_sampling_ratio 의 key 가 node index이고 value 가 group의 이름 겸 score이다.
        
        논문에서는 moves 단위로 priority 를 계산한 후 sampling function을 적용함.
        그러나 SA_FVSP_NNS 방법에 국한할 경우 nodes 단위로 계산해도 괜찮고, 그것이 계산량이 더 적음."""
        dict_index_priority_ranking = {}
        l_t_index_priority = [(i_index,priority) for i_index, priority in dict_index_priority.items()]
        l_t_index_priority.sort(key=lambda x:x[1])#priority 가 작을수록 앞에 나온다.
        for i_rank, t_index_priority in enumerate(l_t_index_priority):
            dict_index_priority_ranking[t_index_priority[0]] = i_rank
        
        #오로지 priority 순서만을 보고서 이를 적당한 group으로 쪼갠다.
        #priority 가 높은 ndoes 가 group rank 가 더 높은 group에 배정된다.
        dict_group_rank_set_indexes = {}
        for i_index, priority_rank in dict_index_priority_ranking.items():
            group_rank = int(priority_rank/i_num_of_vertices_in_segmentation_group)+2
            dict_group_rank_set_indexes.setdefault(group_rank, set()).add(i_index)
        
        #node 별 group_rank 값이 그 node를 선택하는 비중이 된다.
        #dict_index_sampling_ratio 를 나중에 random.choices 에 사용한다.
        dict_index_sampling_ratio = {}
        for group_rank, set_indexes in dict_group_rank_set_indexes.items():
            for index in set_indexes:
                dict_index_sampling_ratio[index] = group_rank
        
        self.dict_index_sampling_ratio = dict_index_sampling_ratio
        
        
        
        


class FVS_brutal_force_searching:
    """SCC형태의 network로부터 FVS를 brutal force searching 방법으로 구한다.
    self loop는 없다고 가정한다.
    적어도 하나의 feedback은 존재한다고 가정한다.
    multiprocessing 기능을 지원한다.
    모든 가능한 조합을 차례차례 조사할 때, 작은 조합부터 시행하는 방법과 큰 조합부터 시행하는 방법을 선택할 수 있도록 설정할 수 있다."""
    def __init__(self, i_num_of_processes=1):
        self.i_num_of_processes = i_num_of_processes
        
        self.t_nodes = []
        self.l_t_links = []
        self.dict_index_node_set_regulator_indexes = {}
        self.dict_index_node_set_target_indexes = {}
        
        self.matrix_adjacent = None
        self.i_num_of_combination_calculating = 1
        
        
    def set_nodes(self, iter_nodes):
        self.t_nodes = tuple((node for node in iter_nodes))
    
    def set_links(self, iter_t_links):
        """self loop는 전 단계에서 걸러진다. 안 걸러지더라도 큰 문제는 없을 거라고 생각하지만."""
        self.l_t_links = iter_t_links.copy()
        for t_link in iter_t_links:
            index_regulator = self.t_nodes.index(t_link[0])
            index_target = self.t_nodes.index(t_link[-1])
            self.dict_index_node_set_regulator_indexes.setdefault(index_target, set()).add(index_regulator)
            self.dict_index_node_set_target_indexes.setdefault(index_regulator, set()).add(index_target)
        
        for i_index in range(len(self.t_nodes)):
            #이 dict들이 모든 node 에 대해 값을 가지도록. 함. SCC network를 넣어줬을 때는 안 해도 상관 없지만 보험삼아서.
            self.dict_index_node_set_regulator_indexes.setdefault(index_target, set())
            self.dict_index_node_set_target_indexes.setdefault(index_regulator, set())
    
    
    def _make_adjacent_matrix_from_l_t_links(self):
        """self.t_nodes 의 순서를 index 로 하여 matrix 를 만든다.
        (i,j)에 True 값이 있으면 self.t_nodes[j] -> self.t_nodes[i] link 가 존재한다는 의미이다."""
        l_ix_1 = []
        l_ix_2 = []
        
        for t_link in self.l_t_links:
            l_ix_1.append(self.t_nodes.index(t_link[-1]))
            l_ix_2.append(self.t_nodes.index(t_link[0]))
        ix = (np.array(l_ix_1), np.array(l_ix_2))
        matrix = np.matrix(np.zeros((len(self.t_nodes),len(self.t_nodes)), dtype=bool))
        if self.l_t_links:#self.l_t_links == [] 인 경우를 대비하여.
            matrix[ix] = 1
        self.matrix_adjacent = matrix
        
    class FVS_calculator_in_defined_combs:
        """전체 가능한 node combinations (itertools.combinations 로 계산) 을 순서대로 i_all_proc 씩 잘랐을 때,
        i_proc_num 번째로 등장하는 combinations 에 대해 FVS인지 검사하고, 
        FVS일 경우 저장한다."""
        def __init__(self, matrix, i_num_comb, i_proc_num, i_all_proc):
            self.matrix_adjacenct = matrix
            self.iterator_comb = itertools.combinations(list(range(len(matrix))), r=i_num_comb)
            self.i_comb_calculated = 0
            self.i_all_proc = i_all_proc#self.iterator_comb 를 i_all_proc 개로 쪼개서 계산한다.
            self.i_proc_num = i_proc_num#i_all proc 개로 쪼개진 combinations 중 i_proc_num 번째 집단을 맡는다. 0부터 시작.
            self.l_t_FVSs = []
            
            self.backup_address = None
    
        @staticmethod
        def _check_matrix_is_acyclic(matrix):
            """matrix should be square matrix
            if the adjacent matrix has cycle, then return False"""
            matrix_multiplied = np.matrix(matrix, dtype = bool)
            if any(np.diag(matrix_multiplied)):#self loop exist!
                return False 
            matrix_copy = np.matrix(matrix, dtype = bool)
            
            for _ in range(len(matrix_copy)-1):
                matrix_multiplied = np.matmul(matrix_multiplied, matrix_copy)
                if any(np.diag(matrix_multiplied)):#loop exist!
                    break
            else:
                return True#acyclic
            return False#there is an cycle
    
        @staticmethod
        def _get_matrix_without_selected_index(matrix, iter_index_selected):
            """matrix 가 주어졌을 때, iter_index_selected 에 포함되는 indexes 값에 
            해당하는 columns와 rows를 지운 matrix를 return 한다."""
            array_all = np.arange(len(matrix))
            array_withdout_selected = np.setdiff1d(array_all, np.array(iter_index_selected))
            return matrix[array_withdout_selected,:][:, array_withdout_selected]
        
        def find_FVSs_for_given_combinations(self):
            while True:
                try: 
                    t_index_comb = next(self.iterator_comb)
                    #print(t_index_comb)
                    if self.i_comb_calculated % self.i_all_proc == self.i_proc_num:
                        matrix_except_selected_nodes = self._get_matrix_without_selected_index(self.matrix_adjacenct, t_index_comb)
                        #print(matrix_except_selected_nodes)
                        if self._check_matrix_is_acyclic(matrix_except_selected_nodes):
                            #print("acyclic")
                            self.l_t_FVSs.append(t_index_comb)
                    self.i_comb_calculated += 1
                except StopIteration:
                    break
        
        def get_FVSs(self):
            return self.l_t_FVSs
        
        def backup_object(self):
            """후에 중간 저장 기능을 넣을 때를 대비하여 만들어 놓은 함수
            현재의 객체를 저장한다. 후에 FVS_brutal_force_searching 객체에서 
            이 pickle 된 객체들을 불러와서 실행할 수 있도록 한다.
            find_FVSs_for_given_combinations 의 도중 특정 시간 간격에 따라 이 함수를 발동시킬 예정."""
            with open(self.backup_address, 'wb') as f:
                pickle.dump(self, f)
    
    @staticmethod
    def let_object_calculate_FVS(obj_FVS_calculator_in_defined_combs):
        """multiprocess에 계산을 분배하기 위한 함수"""
        obj_FVS_calculator_in_defined_combs.find_FVSs_for_given_combinations()
        l_t_FVSs = obj_FVS_calculator_in_defined_combs.get_FVSs()
        return l_t_FVSs
    
    def calculate_FVSs_for_n_comb(self, i_n):
        """이 부분을 multiprocessing 이 가능하도록 개조할 예정.
        중간 저장 기능을 넣을 경우 l_FVS_calculating_subobjects 를 pickle 된 객체 정보에서 
        복원한 뒤 사용할 수 있도록 하려 한다."""
        l_FVS_calculating_subobjects = [self.FVS_calculator_in_defined_combs(self.matrix_adjacent, i_n, i, self.i_num_of_processes) for i in range(self.i_num_of_processes)]
        iter_l_t_FVSs = map(self.let_object_calculate_FVS, l_FVS_calculating_subobjects)
        l_t_FVSs = functools.reduce(operator.add, iter_l_t_FVSs, [])
        return l_t_FVSs
    
    def calculate_FVSs(self):
        """흠.. 2020 10월쯤 만든 방법은 일단 network 를 matrix 로 만든 다음에,
        그 matrix에서 FVS nodes 만 빼서는, matrix 곱을 수행해서 cycle 이 있는지 확인하는 방법인 모양이다.
        이 방법 말고 SCC 체크할 때의 방법을 사용하는 것이 더 좋을까? 그건 FVS_analysis 모듈에 있다.
        n by n 행렬의 곱과, flow 를 dict 형태의 links data를 사용해서 n 까지 확장하는데 걸리는 시간 (list 말고 array.array나 deque 로 해보자)을 비교해보자."""
        self._make_adjacent_matrix_from_l_t_links()
        #print(self.matrix_adjacent)
        #print(self.t_nodes)
        #print(self.l_t_links)
        while self.i_num_of_combination_calculating <= len(self.t_nodes):
            #print(self.i_num_of_combination_calculating, "combination")
            l_t_FVSs = self.calculate_FVSs_for_n_comb(self.i_num_of_combination_calculating)
            if not l_t_FVSs:#i_num_of_comb 개의 조합으로는 FVS 를 만들 수 없음.
                self.i_num_of_combination_calculating += 1
            else:
                l_t_FVSs_original_index = [tuple(self.t_nodes[i] for i in t_FVS) for t_FVS in l_t_FVSs]
                #print(l_t_FVSs_original_index)
                return l_t_FVSs_original_index
        
        
if __name__ == "__main__":
    links = [('a','f'),('f','c'),('c','b'),('b','a'),('a','e'),('e','d'),('d','c')]
    FVS_finder = FVS_finding(links)
    FVS_nodes_sets = FVS_finder.find_FVS()