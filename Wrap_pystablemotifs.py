""""사용법.
bnet file을 사용하여 model에 대한 priems 를 얻는다.
find_attractors_asynchronous_update 함수를 사용하여 Attractor_Repertore 객체를 얻는다.
Attractor_Repertore 객체의 get_attractors method를 사용하여 Attractor 객체를 얻는다.
그 객체로부터 method를 활용하여 정보를 얻는다.

pystbalemotifs의 코드를 더 잘 알게 되면 각각을 내가 편한 형태로 뽑아낼 수 있는 method를 추가할 것."""

import pystablemotifs as sm
from pyboolnet import prime_implicants
#pystablemotifs.__version__ == '3.0.6'
#pyboolnet.version.read_version() == '3.0.11'

import networkx as nx
#nx.__version__ == '3.1'

class Attractor_Repertore:
    #pystablemotifs의 attractorrepertore 객체를 담고,
    #그 객체에서 내가 원하는 정보를 추출해주는 methods로 구성됨.
    def __init__(self, att_rep_obj_in_pystablemotifs):
        self.att_rep_obj_in_pystablemotifs = att_rep_obj_in_pystablemotifs

    def get_attractors(self):
        return [Attractor(att_obj) for att_obj in self.att_rep_obj_in_pystablemotifs.attractors]


class Attractor:
    #pystablemotifs의 attractor 객체를 담고,
    #그 객체에서 내가 원하는 정보를 추출해주는 methods로 구성됨.
    def __init__(self, att_obj_in_pystablemotifs):
        self.att_obj_in_pystablemotifs = att_obj_in_pystablemotifs
            
    def is_point_attractor(self):
        """이것으로 point attractor인지 판별이 안되는 경우가 있으면 수정할 것."""
        return self.att_obj_in_pystablemotifs.n_unfixed == 0
    
    def get_states_in_att(self):
        """attractor에 포함되는 states를 일단 순서에 무관하게 return하기."""
        if self.is_point_attractor():
            state_dict_form = self.att_obj_in_pystablemotifs.attractor_dict.copy()
            return [state_dict_form]
        
        else:
            stg_of_states = self.att_obj_in_pystablemotifs.stg
            #여기서 state는 그냥 1과 0으로 이루어진 string으로 되어 있음.
            #각각의 의미하는 바가, sorted(reduced_primes) 라고 된 것 같은데,
            #틀릴 경우 수정 필요.
            node_order = sorted(self.att_obj_in_pystablemotifs.reduced_primes)
            att_state_dict_forms = []
            for state in stg_of_states.nodes:
                state_dict_form = dict(zip(node_order, (int(i) for i in state)))
                #stg의 nodes에 기록된 state는 fluctuating nodes만 고려한 것으로 보임.
                #fixed nodes의 state 정보를 따로 추가.
                state_dict_form = {**state_dict_form, **self.att_obj_in_pystablemotifs.fixed_nodes}
                att_state_dict_forms.append(state_dict_form)
            
            return att_state_dict_forms
    
    def get_stg_of_att_states(self):
        """nx digraph 형태로 return. 각각의 node에는 번호가 매겨지고,
        그 번호는 get_states_in_att 로 얻어지는 list의 index를 가리키도록 한다."""
        if self.is_point_attractor():
            #point attractor의 경우는 self loop를 지닌 single node graph
            nx_stg = nx.DiGraph()
            nx_stg.add_edge(0,0)
        
        else:
            state_index_map = {}
            for i, state in enumerate(self.att_obj_in_pystablemotifs.stg.nodes):
                state_index_map[state] = i
            
            nx_stg = nx.DiGraph()
            for source, target in self.att_obj_in_pystablemotifs.stg.edges:
                source_i = state_index_map[source]
                target_i = state_index_map[target]
                nx_stg.add_edge(source_i,target_i)
            
        return nx_stg


    





def find_attractors_asynchronous_update(primes, node_fixation:dict["node","state"]):
    """어떤 boolean model에 대해, pyboolnet의 primes가 주어지면, 
    그것에 node_fixation을 처리한 뒤에 asynchronous update의 모든 attractors를 계산한다.
    
    pystablemotifs 패키지의 attractors 객체의 많은 정보 중, 내가 사용하는 것을 추출해 사용.
    쓰다가 잘 안 맞으면 좀 더 조사해서 개조해 쓸 것."""
    primes_with_node_fixation = prime_implicants.create_constants(primes, node_fixation, True)
    attractor_repertore = sm.AttractorRepertoire.from_primes(primes_with_node_fixation)
     
    return Attractor_Repertore(attractor_repertore)


if __name__ == "__main__":
    pass