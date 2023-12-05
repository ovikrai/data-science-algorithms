from algos.bag import Bag
from algos.matrix import Matrix

from networkx import Graph, DiGraph


# TODO: CHECK IMPLEMENTATION
class UndirectedGraph:
    graph: Graph

    # Create the structure
    def __init__(self):
        self.graph = Graph()

    # Number of nodes or vertices
    def vertices(self):
        return self.graph.number_of_nodes()

    # Number of edges
    def edges(self):
        return self.graph.number_of_edges()

    # Add edge <v, w> to this graph
    def add_edge(self, v: int, w: int):
        self.graph.add_edge(v, w)

    # vertices adjacent to v
    def adj(self, v: int):
        self.graph.neighbors(v)


class DirectedGraph:
    graph: Graph

    # Create the structure
    def __init__(self):
        self.graph = DiGraph()

    # Number of nodes or vertices
    def vertices(self):
        return self.graph.number_of_nodes()

    # Number of edges
    def edges(self):
        return self.graph.number_of_edges()

    # Add edge <v, w> to this graph
    def add_edge(self, v: int, w: int):
        self.graph.add_edge(v, w)

    # vertices adjacent to v
    def adj(self, v: int):
        self.graph.neighbors(v)


class WeightedEdge:
    v: int
    u: int
    weight: float

    def __init__(self, v: int, u: int, weight: float):
        self.v = v  # one vertex
        self.u = u  # the other vertex
        self.weight = weight

    def get_weight(self):
        return self.weight

    def either(self):
        return self.v

    def other(self, vertex: int):
        if vertex == self.v:
            return self.u
        elif vertex == self.u:
            return self.v
        else:
            raise RuntimeError('Inconsistent Edge')

    def compare_to(self, that):
        if self.get_weight() < that.get_weight():
            return -1
        elif self.get_weight() > that.get_weight():
            return 1
        else:
            return 0


class WeightedGraph:
    v: int
    u: int
    adj: list

    def __init__(self, v: int):
        self.v = v
        self.u = 0

        # adjacency list of lists
        self.adj = [list] * v

    # Number of nodes or vertices
    def vertices(self):
        return self.v

    # Number of edges
    def edges(self):
        return self.u

    # Add edge <v, w> to this graph
    def add_edge(self, edge: WeightedEdge):
        v = edge.either()
        u = edge.other(v)
        self.adj[v].append(edge)
        self.adj[u].append(edge)
        self.u = self.u + 1

    # list of vertices adjacent to v
    def adj(self, v: int):
        return self.adj[v]

    def get_edges(self):
        bag = []
        for v in range(0, self.v):
            for edge in self.adj[v]:
                if edge.other(v) > v:
                    bag.append(edge)
        return bag


class WeightedDirectedEdge:
    weight: float
    v: int
    u: int

    def __init__(self, v: int, u: int, weight: float):
        self.v = v
        self.u = u
        self.weight = weight

    def get_weight(self):
        return self.weight

    def vertex_from(self):
        return self.v

    def vertex_to(self):
        return self.u


class WeightedDirectedGraph:
    v: int
    u: int
    adjacency: list

    def __init__(self, v: int):
        self.v = v
        self.u = 0
        self.adj = [list] * v

    def vertices(self):
        return self.v

    # Number of edges
    def edges(self):
        return self.u

    def add_edge(self, edge: WeightedDirectedEdge):
        self.adjacency[edge.vertex_from()].append(edge)
        self.u = self.u + 1

    def adj(self, v: int):
        return self.adjacency[v]

    def get_edges(self):
        edges = []
        n = self.v
        for i in range(0, n):
            for e in self.adjacency[i]:
                edges.append(e)

        return edges
