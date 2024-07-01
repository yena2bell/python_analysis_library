class Acyclic_form:
    """어떤 network에 대해, edge의 tuple form (source node, ... , target node)으로 된 list가 주어지고,
    그 network의 FVS nodes가 주어졌을 때,
    acyclic form을 만들어서 return한다.
    
    edge의 tuple form에서 source node와 target node는 string으로 주어질 것."""
    def __init__(self, edges_tuple_form:"[(source node:str, ..., target node:str),...]"):
        self.edges = edges_tuple_form
        self.FVS_nodes = None

        self.input_nodes = []
        #처음부터 mutated network를 넣을 경우, 이 input nodes는 original network의 input nodes와 다를 수 있음.
        self.output_nodes = []
        self._find_input_nodes_and_output_nodes()


        self.__FVS_connecting_paths = {}
        #FVS pair에 대해, 그 사이를 잇는 path는 list of tuple forms으로 저장한다.
        #하나의 FVS pair에 대해 두 개 이상의 path가 존재할 수 있기 때문에 list of path를 담는다.

    def _find_input_nodes_and_output_nodes(self):
        """주어진 tuple form edges에서, 
        target node에만 등장하는 node가 output nodes
        source_node에만 등장하거나, 유일한 incoming link가 자기 자신일 경우 input nodes"""
        source_nodes = set()
        target_nodes = set()

        input_node_candidates = set()
        for edge in self.edges:
            source_node = edge[0]
            target_node = edge[-1]
            if source_node == target_node:
                #이것들은 input node일 수 있지만 output node는 될 수 없다.
                input_node_candidates.add(source_node)
            else:
                source_nodes.add(source_node)
                target_nodes.add(target_node)

            
            
        
        for node in target_nodes.difference(source_nodes):
            self.output_nodes.append(node)

        for node in source_nodes.difference(target_nodes):
            self.input_nodes.append(node)
        for node in input_node_candidates.difference(target_nodes):
            #자기자신 self loop 외에, 다른 incoming links가 없는것들만
            self.input_nodes.append(node)



    def set_FVS_nodes(self, FVS_nodes):
        self.FVS_nodes = FVS_nodes

    def get_acyclic_form_edges(self, source_suffix="_source", sink_suffix="_sink"):
        """이 객체에 입력된 tuple form edges에 대해, FVS nodes를 source node와 sink node로 쪼개서,
        source node와 sink node에 각각 suffix를 붙인 뒤, return한다."""
        edges_acyclic_form = []
        for edge_tuple_form in self.edges:
            source_node = edge_tuple_form[0]
            target_node = edge_tuple_form[-1]
            if source_node in self.FVS_nodes:
                source_node += source_suffix
            if target_node in self.FVS_nodes:
                target_node += sink_suffix
            
            edge_for_acyclic_form = (source_node, *edge_tuple_form[1:-1], target_node)
            edges_acyclic_form.append(edge_for_acyclic_form)
        
        return edges_acyclic_form
    
    def get_FVS_connecting_paths(self, nodes_form=False):
        """이 객체에 입력된 tuple form edges에 대해, 주어진 FVS nodes로 만들어지는
        acyclic form에서, input node나 source FVS node를 시작점으로 하여,
        target FVS node나 이미 있던 sink node에서 끝나는 모든 path를 찾아서
        return한다.
        
        nodes_form이 True이면, path의 nodes만 뽑아서 return"""
        if self.__FVS_connecting_paths:
            FVS_connecting_paths_edges = self.__FVS_connecting_paths.copy()
        else:
            FVS_connecting_paths_edges = {}
            source_node_to_edges_map = self._get_source_node_to_edges_map(self.edges)

            def _make_path_recursively(path, source_node_to_edges_map):
                """하나의 node에 대해, 그 node의 downstream으로 이어지는 path를
                recursive하게 다 찾는 함수"""
                all_paths = []

                if path[-1][-1] in self.output_nodes:
                    all_paths.append(path)
                    return all_paths
                if path[-1][-1] in self.FVS_nodes:
                    all_paths.append(path)
                    return all_paths

                
                for next_edge in source_node_to_edges_map.get(path[-1][-1],[]):
                    path_extended = path + [next_edge]                    
                    all_paths.extend(_make_path_recursively(path_extended, source_node_to_edges_map))
                return all_paths
            
            for source_node_in_acyclic_form in self.input_nodes + list(self.FVS_nodes):
                for edge_start in source_node_to_edges_map.get(source_node_in_acyclic_form,[]):
                    path_start = [edge_start]
                    paths_starting_form_this_node = _make_path_recursively(path_start, source_node_to_edges_map)

                    for path in paths_starting_form_this_node:
                        start_node = path[0][0]
                        end_node = path[-1][-1]
                        FVS_connecting_paths_edges.setdefault((start_node, end_node), []).append(path)
                
        if nodes_form:
            FVS_connecting_paths_nodes = {}
            for start_end, conneting_paths_edge_form in FVS_connecting_paths_edges.items():
                connecting_paths_node_form = []
                for connecting_path_edge_form in conneting_paths_edge_form:
                    connecting_path_node_form = []
                    for edge in connecting_path_edge_form:
                        connecting_path_node_form.append(edge[0])
                    else:
                        connecting_path_node_form.append(edge[-1])
                    connecting_paths_node_form.append(connecting_path_node_form)
                
                FVS_connecting_paths_nodes[start_end] = connecting_paths_node_form
            return FVS_connecting_paths_nodes
        else:
            return FVS_connecting_paths_edges
    
    def get_source_node_to_target_nodes_map(self, edges_tuple_form):
        """주어진 edges tuple forms를 
        {source node: set(target node1, target node2,,,)}
        형태로 바꿔서 return"""
        source_node_to_target_noets_map = {}
        for edge_tuple_form in edges_tuple_form:
            source_node = edge_tuple_form[0]
            target_node = edge_tuple_form[-1]
            source_node_to_target_noets_map.setdefault(source_node, set()).add(target_node)
        
        return source_node_to_target_noets_map
    
    def _get_source_node_to_edges_map(self, edges_tuple_form):
        """주어진 edges tuple forms를 
        {source node: [edge1, edge2,,,]}
        형태로 바꿔서 return
        여기서 edge1,2,,, 는 source node가 key 값인 edges"""
        source_node_to_edges_map = {}
        for edge_tuple_form in edges_tuple_form:
            source_node = edge_tuple_form[0]
            source_node_to_edges_map.setdefault(source_node, []).append(edge_tuple_form)
        
        return source_node_to_edges_map


if __name__ == "__main__":
    from Model_read_using_pyboolnet import read_pyboolnet_file
    import os
    library_folder = os.path.abspath(os.path.dirname(__file__))
    toy_model_dynamics_pyboolnet = read_pyboolnet_file(os.path.join(library_folder,"toy_data","toy_model1.bnet"))
    edges_tuple_form = toy_model_dynamics_pyboolnet.get_edges_with_modalities(node_cut=[], ignore_self_loop_on_source_nodes=True)

    acylic_form_obj = Acyclic_form(edges_tuple_form)
    acylic_form_obj.set_FVS_nodes(('n4', 'n5', 'n9'))
    