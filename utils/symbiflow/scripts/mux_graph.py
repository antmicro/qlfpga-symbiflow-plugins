#!/usr/bin/env python3
"""
A set of classes for building OpenFPGA compatible mux graphs and deriving their
encoding bit patterns.
"""

from enum import Enum
import math

# =============================================================================


class NodeType(Enum):
    """
    Mux graph node type
    """
    INTERNAL = 0
    INPUT = 1
    OUTPUT = 2


class Node:
    """
    Mux graph node. Stores id and optionally pin index
    """
    def __init__(self, node_id, node_type):
        self.id = node_id
        self.type = node_type
        self.pin = None


class Edge:
    """
    Mux graph edge. Stores source and destination node ids plus a tuple
    (bit, value) indicating which bit needs to be set/cleared to enable it.
    """
    def __init__(self, src_id, dst_id):
        self.src_id = src_id
        self.dst_id = dst_id
        self.bit = None

# =============================================================================


class Graph:
    """
    Multiplexer graph
    """

    def __init__(self, num_inputs):
        """
        Basic constructor
        """
        self.num_inputs = num_inputs
        self.total_bits = 0
        self.nodes = {}
        self.edges = set()

        self.input_node_ids = {}
        self.output_node_id = None

        self.next_node_id = 0

    def add_node(self, node_type):
        """
        Adds a node to the graph. Automatically sets its id. Returns the node
        object.
        """
        node = Node(self.next_node_id, node_type)

        self.nodes[node.id] = node
        self.next_node_id += 1

        return node

    def add_edge(self, src_id, dst_id):
        """
        Adds a new edge to the graph. Checks if the nodes that it refers to
        are present in the graph. Returns the edge object
        """
        assert src_id in self.nodes
        assert dst_id in self.nodes

        edge = Edge(src_id, dst_id)

        self.edges.add(edge)
        return edge

    @staticmethod
    def build_multi_level(num_inputs, num_levels, num_inputs_per_branch):
        """
        Builds a multi-level multiplexer graph.
        """
        assert num_inputs_per_branch > 1, num_inputs_per_branch

        # Either a single bit or 1 bit per input
        if num_inputs_per_branch == 2:
            bits_per_level = 1
        else:
            bits_per_level = num_inputs_per_branch

        # Create the graph
        graph = Graph(
            num_inputs = num_inputs
        )

        # Set the total number of bits
        num_bits_per_level = num_inputs_per_branch
        if num_inputs_per_branch == 2:
            num_bits_per_level = 1

        graph.total_bits = num_bits_per_level * num_levels

        # Create the output node
        output_node = graph.add_node(NodeType.OUTPUT)
        output_node.pin = 0

        # Create input node ID list. Initialize it with the output one as there
        # are no levels yet.
        input_node_ids = {output_node.id}

        # Create the node lookup
        node_lookup = [[] for i in range(num_levels + 1)]
        node_lookup[num_levels].append(output_node.id)

        # Build all levels
        def build_levels():
            for lvl in range(num_levels)[::-1]:
                # Build one level
                for seed_node_id in node_lookup[lvl + 1]:

                    # Remove the seed from the input list
                    input_node_ids.remove(seed_node_id)

                    # Build each mux on this level
                    for i in range(num_inputs_per_branch):

                        # Add a node
                        node = graph.add_node(NodeType.INTERNAL)
                        # Connect with the seed node
                        edge = graph.add_edge(node.id, seed_node_id)

                        # A 2:1 mux
                        if num_inputs_per_branch == 2:
                            bit = lvl 
                            val = (i == 0)
                        # A N:1 mux
                        else:
                            bit = lvl * num_inputs_per_branch + i
                            val = True

                        assert bit < graph.total_bits, (graph.total_bits, bit)
                        edge.bit = (bit, val)

                        # Add the node to the lookup
                        node_lookup[lvl].append(node.id)
                        # Add to the input list
                        input_node_ids.add(node.id)

                        # The demand is met
                        if len(input_node_ids) >= num_inputs:
                            return

        build_levels()

        # Update input node types
        for node_id in input_node_ids:
            graph.nodes[node_id].type = NodeType.INPUT

        # Rearrange input nodes
        input_node_ids = []
        input_id = 0

        for lvl in range(num_levels):
            for node_id in node_lookup[lvl]:
                node = graph.nodes[node_id]

                if node.type == NodeType.INPUT:

                    node.pin = input_id
                    input_id += 1

                    input_node_ids.append(node.id)

        # Store node maps
        graph.input_node_ids = input_node_ids
        graph.output_node_id = output_node.id

        return graph

    @staticmethod
    def build(num_inputs, impl_structure, num_levels=None, add_const_input=False):
        """
        Builds a mux graph basing on the given parameters.
        """

        # Get the size of the mux implementation
        if add_const_input:
            num_inputs += 1

        # Build the mux graph for the appropriate structure.
        if impl_structure == "tree":
            assert num_levels is None

            # Determine level count
            num_levels = (num_inputs - 1).bit_length()

            # Build
            graph = Graph.build_multi_level(num_inputs, num_levels, 2)

        elif impl_structure == "multi_level":
            assert num_levels is not None

            # Special cases
            if num_levels in [1, 2]:
                num_inputs_per_branch = num_inputs

            elif num_levels == 2:
                num_inputs_per_branch = int(math.ceil(math.sqrt(num_inputs)))

            else:
                num_inputs_per_branch = 2
                while num_inputs_per_branch ** num_levels < num_inputs:
                    num_inputs_per_branc += 1

            # Build
            graph = Graph.build_multi_level(num_inputs, num_levels, num_inputs_per_branch)

        elif impl_structure == "one_level":
            assert num_levels is None

            # Build
            graph = Graph.build_multi_level(num_inputs, 1, num_inputs)

        else:
            assert False, impl_structure

        return graph

    def get_input_encoding(self, pin):
        """
        Discovers and returns bit pattern required to select the given input.
        The bit pattern is returned as a list containing True for 1, False for
        0 and None for don't care.
        """
        assert pin < self.num_inputs, pin

        # Initialize encoding bits
        encoding = [None] * self.total_bits

        # Walk from the input node to the output node
        node_id = self.input_node_ids[pin]
        while self.nodes[node_id].type != NodeType.OUTPUT:

            # Find edge
            edges = [e for e in self.edges if e.src_id == node_id]
            assert len(edges) == 1, (node_id, edges)

            # Set bit
            bit, val = edges[0].bit

            assert encoding[bit] is None
            encoding[bit] = val

            # Jump to the next node
            node_id = edges[0].dst_id

        return encoding

    def get_all_input_encoding(self):
        """
        Returns encoding for all inputs as a dict indexed by input pin indices.
        """
        encoding = {i: self.get_input_encoding(i) \
                    for i in range(self.total_inputs)}

        return encoding

    @staticmethod
    def encoding_to_string(encoding):
        """
        An utility function that converts an input encoding to an user-friendly
        string.
        """

        # Convert
        encoding_str = ""
        for bit in encoding:
            if bit is True:
                encoding_str += "1"
            elif bit is False:
                encoding_str += "0"
            elif bit is None:
                encoding_str += "-"
            else:
                assert False, bit

        return encoding_str

    def dump_dot(self):
        """
        Dumps the graph into graphviz .dot format. Returns a list of lines.
        """

        dot = []

        # Add header
        dot.append("digraph g {")
        dot.append(" rankdir=LR;")
        dot.append(" ratio=0.5;")
        dot.append(" splines=false;")
        dot.append(" node [style=filled];")

        # Build nodes
        nodes = {}
        for node in self.nodes.values():

            label = "{}".format(node.id)
            xlabel = "{}".format(node.pin)
            color = "#D0D0D0"
            rank = 0

            if node.type == NodeType.INPUT:
                shape = "diamond"
            elif node.type == NodeType.OUTPUT:
                shape = "octagon"
            else:
                shape = "ellipse"

            if rank not in nodes:
                nodes[rank] = []

            nodes[rank].append({
                "id": node.id,
                "label": label,
                "xlabel": xlabel,
                "color": color,
                "shape": shape
            })

        # Add nodes
        for rank, nodes in nodes.items():
            dot.append(" {")
            #dot.append("  rank=same;")

            for node in nodes:
                dot.append("  node_{} [label=\"{}\",xlabel=\"{}\",fillcolor=\"{}\",shape={}];".format(
                    node["id"],
                    node["label"],
                    node["xlabel"],
                    node["color"],
                    node["shape"],
                ))

            dot.append(" }")

        # Add edges
        for edge in self.edges:

            if edge.bit is not None:
                label = "{}{}".format("" if edge.bit[1] else "!", edge.bit[0])
            else:
                label = ""

            color = "#000000"

            dot.append(" node_{} -> node_{} [label=\"{}\",color=\"{}\"];".format(
                edge.src_id,
                edge.dst_id,
                label,
                color
            ))

        # Footer
        dot.append("}")
        return "\n".join(dot)
