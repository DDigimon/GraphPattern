import networkx as nx
from networkx.algorithms import isomorphism
from graph_nlp import graph_txt

pattern_description={'triangle':'triangle (A motif consisting of three nodes where each node is connected to the other two, forming a triangle)',
                     'diamond':'diamond (A four-node motif with five edges)',
                     'tailed-triangle':'tailed triangle (A triangle with an additional node connected to one of the vertices of the triangle)',
                     'square':'square (A 4-node cycle where each node is connected to exactly two other nodes)',
                     'house':'house (A motif resembling the shape of a house with 5 nodes and 6 edges. The vertices and edges are arranged such that there is a triangular "roof" on top of a square or rectangular "base."',
                     'FFL':'3-node Feed-Forward Loop (A three-node directed motif in which one source node influences a target node through two distinct pathways)',
                     'FBL':'3-node Feedback loop (A directed cycle where the nodes form a loop)',
                     'vs':'V-structure (Two nodes have directed edges pointing toward a common target node)',
                     'd-diamond':'direceted diamond (A 4-node motif in a directed graph where one node has directed edges to two intermediate nodes, and both of those intermediate nodes have directed edges to a common target node.)',
                     }

pattern_output_form={'diamond':'The detected patterns are: [(#1, #2, #3, #4), ...]',
            'square':'The detected patterns are: [(#1, #2, #3, #4), ...]',
            'tailed-triangle':'The detected patterns are: [(#1, #2, #3, #4), ...]',
            'house':'The detected patterns are: [(#1, #2, #3, #4, #5), ...]',
            'triangle':'The detected patterns are: [(#1, #2, #3), ...]',
            'FFL':'The detected patterns are: [(#1, #2, #3), ...]',
            'FBL':'The detected patterns are: [(#1, #2, #3), ...]',
            'vs':'The detected patterns are: [(#1, #2, #3), ...]',
            'd-diamond':'The detected patterns are: [(#1, #2, #3, #4), ...]',
            'diamond_a':'The detected patterns are: [(#1, #2, #3, #4, #5), ...]'}


def direct_judge(patterns):
    if 'FFL' in patterns or 'FBL' in patterns or 'd-diamond' in patterns or 'vs' in patterns:
        direction=True
    else:
        direction=False
    return direction

def _triangle():
    G = nx.Graph()
    # Add edges to form a triangle
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
    return G

def _triangle_tailed():
    G = nx.Graph()
    # Add edges to form a triangle
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A'),('D','C')])
    return G

def _square():
    G = nx.Graph()
    # Add edges to form a triangle
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'),('D','A')])
    return G

def _house():
    G = nx.Graph()
    # Add edges to form a triangle
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'),('D','A'),('C','E'),('E','D')])
    return G

def _hexagon():
    G = nx.Graph()
    # Add edges to form a triangle
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'),('D','E'),('E','F'),('F','A')])
    # features={'A':'C'}
    return G

def _diamond():
    G = nx.Graph()
    # Add edges to form a triangle
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A'), ('A', 'D'),('B', 'D')])
    return G

def _diamond_a():
    G = nx.Graph()
    # Add edges to form a triangle
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A'), ('A', 'D'),('B', 'D'),('D','E')])
    return G

def _FFL():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C'])
    G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C')])
    return G

def _FBL():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C'])
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
    return G

def _vs():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C'])
    G.add_edges_from([('A', 'B'), ('C', 'B')])
    return G

def _ddamond():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C','D'])
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'D'),('D','C')])
    return G



def pattern_generation(name):
    if name=='triangle':
        return _triangle()
    if name=='diamond':
        return _diamond()
    if name=='diamond_a':
        return _diamond_a()
    if name=='FFL':
        return _FFL()
    if name=='tailed-triangle':
        return _triangle_tailed()
    if name=='square':
        return _square()
    if name=='house':
        return _house()
    if name=='FBL':
        return _FBL()
    if name=='vs':
        return _vs()
    if name=='d-diamond':
        return _ddamond()
    if name=='hexagon':
        return _hexagon()
    

def find_pattern_list(target_graph,pattern_name):
    if 'claim' in pattern_name:
        pattern_name=pattern_name.split('_')[1]
    directed=direct_judge(pattern_name)
    pattern_graph=pattern_generation(pattern_name)
    if directed==False:
        GM = isomorphism.GraphMatcher(target_graph, pattern_graph)
    else:
        GM = isomorphism.DiGraphMatcher(target_graph, pattern_graph)
    matches = list(GM.subgraph_isomorphisms_iter())
    triangles_networkx=set()
    
    for m in matches:
        pattern=tuple(sorted(m.keys()))
        triangles_networkx.add(pattern)
    return triangles_networkx
