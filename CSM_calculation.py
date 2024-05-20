# -*- coding: utf-8 -*-

"""
expanded network 를 기반으로 찾는다.
networkx object 를 links 정보만 따로 분리해서 사용함.


특정 길이까지의 cycles 찾으면서, 그 cycle에 모순되는 nodes 조합이 들어가면 찾지 않는 식으로.
"""
import pickle, os

from Expanded_net_analysis import Expanded_node
import Cycle_analysis

class CSM:
    def __init__(self):
        self.condition_part = {}
        self.stable_part = {}
        self.condition_part_str_form = []
        self.stable_part_str_form = []

    def get_CSM_list_form(self, CSM_list_form):
        """CSM을 이루는 expanded net의 expanded nodes로 이루어진 list를 받음.
        안에 모순되는 expanded nodes는 없는 것을 전제로 함."""
        tuple_forms_single = []
        tuple_forms_composite = []

        for tuple_form_node in CSM_list_form:
            if _is_single_node(tuple_form_node):
                tuple_forms_single.append(tuple_form_node)
            else:
                tuple_forms_composite.append(tuple_form_node)
        
        #single expanded node로 list에 포함되어 있는 것들은 stable part
        for tuple_form_single in tuple_forms_single:
            node_original = tuple_form_single[0]
            node_state = tuple_form_single[1]
            self.stable_part[node_original] = node_state
            self.stable_part_str_form.append(Expanded_node.get_str_form_from_dict_form({node_original:node_state}))
        self.stable_part_str_form = list(set(self.stable_part_str_form))


        #먼저 stable part를 구한 뒤, composite node를 이루는 것 중 stable part 에 포함되지 않은 것들이
        #condition part
        for tuple_form_composite in tuple_forms_composite:
            for tuple_form_single in tuple_form_composite:
                node_original = tuple_form_single[0]
                node_state = tuple_form_single[1]
                if node_original not in self.stable_part:
                    self.condition_part[node_original] = node_state
                    self.condition_part_str_form.append(Expanded_node.get_str_form_from_dict_form({node_original:node_state}))
        self.condition_part_str_form = list(set(self.condition_part_str_form))

    def __repr__(self):
        return str((self.condition_part_str_form, self.stable_part_str_form))
    
    def __add__(self, other_CSM):
        """두 CSM을 합쳐서 생성되는 새로운 CSM 객체를 return
        단 두 개의 CSM에 모순되는 nodes가 있을 경우 error 발생."""
        ex_nodes_in_self = {**self.condition_part,**self.stable_part}
        ex_nodes_in_other = {**other_CSM.condition_part,**other_CSM.stable_part}
        for node, state in ex_nodes_in_self.items():
            if node in ex_nodes_in_other:
                if ex_nodes_in_other[node] != state:
                    raise ValueError("two CSMs are conflicted")

        csm_summed = CSM()
        csm_summed.stable_part = {**self.stable_part,**other_CSM.stable_part}
        csm_summed.stable_part_str_form = list(set(self.stable_part_str_form + other_CSM.stable_part_str_form))
        condition_part_of_two = {**self.condition_part,**other_CSM.condition_part}
        for condition_node, state in condition_part_of_two.items():
            if condition_node not in csm_summed.stable_part:
                csm_summed.condition_part[condition_node] = state
                csm_summed.condition_part_str_form.append(Expanded_node.get_str_form_from_dict_form({condition_node:state}))
        
        return csm_summed

    
###############################################################################################
### nx form expanded network를 받아 CSM 객체들을 return하는 것 관련 함수들.
###############################################################################################

def get_CSMs_from_expanded_network_nxform(expanded_network_nx, 
                                            max_len_of_csm_feedback=5):
    """ExpandedNetwork에서 사용하는 expanded network에서 CSM을 계산한다.
    expanded network에서 max_len_of_csm_feedback 길이 이하의 cycle 중, 
    모순이 발생하지 않는 cycle (CSM)을 탐색한다.
    그 후, 각각의 탐색된 CSM을
    (condition part에 해당하는 single expanded node의 str forms로 이루어진 tuple, 
    stable part에 해당하는 single expanded node의 str forms로 이루어진 tuple)
    로 만든 뒤,    (예시: (['n1_0', 'n6_1', 'n7_0'], ['n2_0', 'n3_1', 'n4_0', 'n8_0', 'n9_0']))
    그것들을 list에 모아서 return한다."""
    expanded_network_tuple_form = _convert_expanded_network_format(expanded_network_nx)
    return get_CSM_metanodes_from_expanded_network(expanded_network_tuple_form, max_len_of_csm_feedback)

def get_CSM_metanodes_from_expanded_network(expanded_network_tuple_forms, 
                                            max_len_of_csm_feedback=5):
    """expanded network 로부터 CSMs를 찾아서 return한다.
    expanded network에서 cycle의 길이가 max_len_of_csm_feedback 이하이면서 (composite node 포함)
    모순이 없는 expanded nodes 로 구성된 것을 찾게 된다.
    
    이 때, stable part 는 동일하지만 condition part 는 다른 경우는 다른 객체로 찾게 됨.
    
    links_of_expanded_network 는 expanded_network_construction 으로 만들어진 
    expanded network 의 links"""
    set_expanded_nodes = _get_nodes_of_links(expanded_network_tuple_forms)
    expanded_network_tuple_forms_copied = expanded_network_tuple_forms.copy()

    CSMs = []
    i_count = 0
    for expanded_node_tuple_form in set_expanded_nodes:
        i_count += 1
        if i_count%10 == 0:
            print("checked_num/all_expanded_nodes = {}/{}".format(i_count, len(set_expanded_nodes)))
        #매 loop마다 node가 지워지면서 새로 만들어진 expaded net을 써서 분석.
        set_expanded_nodes_on_links = _get_nodes_of_links(expanded_network_tuple_forms_copied)
        if expanded_node_tuple_form not in set_expanded_nodes_on_links:
            continue
        #l_t_t_expanded_nodes_link를 쳐내는 과정에서 t_expanded_node와 연결된 links 가 전부 사라질 수 있음. 그 경우 error 발생 가능하므로.
        CSMs.extend(_find_CSM_containing_one_node(expanded_node_tuple_form, expanded_network_tuple_forms_copied, max_len_of_csm_feedback))
        expanded_network_tuple_forms_copied = [t_link for t_link in expanded_network_tuple_forms_copied if expanded_node_tuple_form not in t_link]
    
    return CSMs

def _get_nodes_of_links(links):
    """links에 들어있는 nodes를 모아서 return한다.
    이건 nx form expanded net의 method로 처리해도 좋을 것 같은데..."""
    set_nodes = set()
    for link in links:
        set_nodes.update(link)
    return set_nodes

def _find_CSM_containing_one_node(expanded_node_tuple_form, expanded_net_tuple_forms, i_max_len=None):
    """tuple form의 expanded network와 하나의 expanded node가 주어졌을 때, 
    모순되지 않으면서 cycle 길이가 i_max_len 이하인 cycle (CSM)을 찾아 return한다.
    
    CSM class 의 객체로 return한다."""
    obj_CSM_finder = Find_CSM_containing_node(expanded_node_tuple_form, expanded_net_tuple_forms)
    CSMs = []
    for CSM_list_form in obj_CSM_finder.find_CSM(i_max_len):
        CSM_object = CSM()
        CSM_object.get_CSM_list_form(CSM_list_form)
        CSMs.append(CSM_object)

    return CSMs

class Find_CSM_containing_node(Cycle_analysis.Find_cycles_containing_the_node):
    def __init__(self, t_expanded_node, l_t_t_expanded_nodes_link):
        """ input은 expanded node, link의 tuple form 을 사용할 것."""
        super().__init__( t_expanded_node, l_t_t_expanded_nodes_link)
        self.dict_node_index_set_contradicts = {i:set([]) for i in range(len(self.l_nodes))}
        self._calculate_dict_node_index_set_contradicts()
        self.l_set_of_contradicts = [set([])]#self.l_set_of_contradicts[i] 는 l_flow[:i+1] 까지의 모든 node에 대해 self.dict_node_index_set_contradicts의 값을 합집합 해놓은 것.
    
    def _calculate_dict_node_index_set_contradicts(self):
        for i,t_node in enumerate(self.l_nodes):
            for j, t_node_to_compare in enumerate(self.l_nodes[i+1:]
                                                  ):
                if _two_nodes_are_contradict(t_node, t_node_to_compare):
                    self.dict_node_index_set_contradicts[i].add(i+j+1)
                    self.dict_node_index_set_contradicts[i+j+1].add(i)
                    
    def _extend_flow(self, i_node):
        super()._extend_flow(i_node)
        #print(self.l_flow)##############################
        ###################added in CSM
        self.l_set_of_contradicts.append(self.l_set_of_contradicts[-1].union(self.dict_node_index_set_contradicts[i_node]))
        ###################added in CSM
    
            
    def _go_to_next_node(self, i_next_edge):
        i_node_next = self.dict_i_node_t_i_targets[self.l_flow[-1]][i_next_edge]
        self.dict_i_count_for_each_node[self.l_flow[-1]] -= 1
        
        ###################added in CSM
        if i_node_next in self.l_set_of_contradicts[-1]:
            return
        ###################added in CSM
        
        if i_node_next == self.i_index_of_the_node:#it is a cycle containing 0
            self.l_l_i_cycles.append(self.l_flow.copy())
        elif len(self.l_flow) >= self.i_max_len:
            return
        elif not self.array_blocked[i_node_next]:
            self._extend_flow(i_node_next)
    
    def _back_from_node(self):
        ###################added in CSM
        self.l_set_of_contradicts.pop(-1)
        ###################added in CSM
        super()._back_from_node()
        
            
    def find_CSM(self, max_len=None):
        """max_len 은 expanded network 상에서 최대 얼마 길이까지의 simple cycle까지 탐색하는지 지정.
        None일 경우 제한 없음을 의미함. max_len=1 이면 self-loop 만 탐색하는 식.
        계산량이 너무 많아질 것을 대비하였음.
        
        list 안에 CSM을 이루는 expanded nodes로 이루어진 list가 여러 개 들어가는 형태로 return"""
        super().find_cycles("simple", max_len)#ignore return value of this function
        
        set_t_i_cycles = set([tuple(sorted(l_cycle)) for l_cycle in self.l_l_i_cycles])#CSM 순서 차이에 의한 중복 제거.
        l_l_CSM = [[self.l_nodes[i] for i in l_cycle] for l_cycle in set_t_i_cycles]
        for l_CSM in l_l_CSM:
            l_CSM.sort(key=lambda x:str(x))
        return l_l_CSM

###############################################################################################


###############################################################################################
### nx form expanded network를 tuple form으로 변환하는 것 관련.
###############################################################################################

def _convert_expanded_network_format(expanded_network_nx):
    """하나의 module에서 만들어진 expanded network 객체를
    다른 expanded network 관련 module에서 다룰 수 있도록 변형.
    구체적으로, network 객체에서 links 만 추출한 뒤,
    각 link의 형태를 변형시킨다."""
    links_converted = []
    for link in expanded_network_nx.edges():
        node_from = link[0]
        node_from_info = expanded_network_nx.nodes[node_from]["info"]
        node_to = link[-1]
        node_to_info = expanded_network_nx.nodes[node_to]["info"]
        node_from_tuple_form = _convert_nx_node_to_tuple_form(node_from_info)
        node_to_tuple_form = _convert_nx_node_to_tuple_form(node_to_info)
        links_converted.append((node_from_tuple_form, node_to_tuple_form))

    return links_converted

def _convert_nx_node_to_tuple_form(expanded_node_info):
    """str_form의 expanded node를 받아서 (node name, state) 형태로 변환한다."""
    if expanded_node_info.is_composite():
        return tuple((node,state) for node, state in expanded_node_info.dict_form.items())
    else:
        for node, state in expanded_node_info.dict_form.items():
            #node state 한 쌍 뿐이지만 추출하기 귀찮아서 그냥 for 문 씀.
            return (node, state)

def _is_single_node(t_expanded_node):
    """tuple form expanded node가 single이면 return True
    single 이면 (node, state) 이고
    composite이면 ((node1,state1),(node2,state2))이니까."""
    #return not type(t_expanded_node[1]) == type((0,))
    return not type(t_expanded_node[1]) == tuple

def _two_nodes_are_contradict(t_expanded_node1, t_expanded_node2):
    """두 tuple form expanded node 사이에 서로 모순되는 부분이 있으면 True"""
    if _is_single_node(t_expanded_node1):
        t_expanded_node1 = [t_expanded_node1]
    dict_node_state1 = {node:state for node,state in t_expanded_node1}
    if _is_single_node(t_expanded_node2):
        t_expanded_node2 = [t_expanded_node2]
    dict_node_state2 = {node:state for node,state in t_expanded_node2}
    set_intersection = set(dict_node_state1.keys()).intersection(set(dict_node_state2.keys()))
    if set_intersection:
        for node in set_intersection:
            if dict_node_state1[node] != dict_node_state2[node]:
                return True
    return False

###############################################################################################
    

###############################################################################################
### 결과로 나온 CSMs를 pickle로 저장하고 불러오는 것 관련.
###############################################################################################

def pickling_CSMs(CSMs, save_address, file_name):
    """계산이 오래 걸리는 경우가 많으니 왠만하면 pickling 해둘 것."""
    with open(os.path.join(save_address, file_name), 'wb') as f:
        pickle.dump(CSMs, f)

def get_CSMs_from_pickle(save_address, file_name):
    with open(os.path.join(save_address, file_name), 'rb') as f:
        return pickle.load(f)
    
###############################################################################################


    

if __name__ == "__main__":
    import Model_read
    import Expanded_net_analysis
    add_test_model = os.path.join(r"D:\new canalizing kernel\우정 tumorigenesis 모델", "Fumia_logic_model.bnet")
    dynamics_test_model = Model_read.read_pyboolnet_file(add_test_model)
    expanded_toy = Expanded_net_analysis.make_expanded_net_using_dynamics_pyboolnet(dynamics_test_model,reduction=True, perturbation={})
    #여기까지는 이 module에 넣을 data 예시로 만드는 코드.
    #이 toy model은 CSM 계산이 오래 걸린다.
    print("toy expanded model 객체 생성됨.")

    CSMs = get_CSMs_from_expanded_network_nxform(expanded_toy, max_len_of_csm_feedback=9)
    save_address = r"D:\new canalizing kernel\우정 tumorigenesis 모델"
    pickling_CSMs(CSMs, save_address, "CSM_pickle_test.pickle")
