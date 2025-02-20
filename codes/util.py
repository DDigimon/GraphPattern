from networkx.algorithms import isomorphism
import networkx as nx
from graph_patterns import direct_judge

def iso_list_check(g,g_list):
    flag=True
    for g_in in g_list:
        GM = isomorphism.GraphMatcher(g_in, g)
        if GM.is_isomorphic():
            # print(g_in,g)
            flag=False
    return flag

def graph_iso_check(g1,g2):
    if len(g1.edges())!=len(g2.edges()):
        return False
    else:
        if g1.is_directed():
            in_degree_sequence_1 = sorted([d for n, d in g1.in_degree()], reverse=True)
            out_degree_sequence_1 = sorted([d for n, d in g2.out_degree()], reverse=True)

            in_degree_sequence_2 = sorted([d for n, d in g1.in_degree()], reverse=True)
            out_degree_sequence_2 = sorted([d for n, d in g2.out_degree()], reverse=True)
            # print(in_degree_sequence_1,in_degree_sequence_2)
            if in_degree_sequence_1!=in_degree_sequence_2 or out_degree_sequence_1!=out_degree_sequence_2:
                return False
        else:
            degree_sequence_1 = sorted([d for n, d in g1.degree()], reverse=True)
            degree_sequence_2 = sorted([d for n, d in g1.degree()], reverse=True)
            if degree_sequence_1!=degree_sequence_2:
                return False
        
        GM = isomorphism.GraphMatcher(g1, g2)
        if GM.is_isomorphic():
            return True
        else:
            return False
