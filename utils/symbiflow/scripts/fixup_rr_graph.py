#!/usr/bin/env python3
"""
"""
import argparse
import os

from enum import Enum

import lxml.etree as ET

import capnp
import capnp.lib.capnp
capnp.remove_import_hook()

from rr_graph import tracks
import rr_graph.graph2 as rr
import rr_graph_xml.graph2 as rr_xml
import rr_graph_capnp.graph2 as rr_capnp
from rr_graph.lib import progressbar_utils

from gsb_data import load_gsb_data
from mux_graph import Graph as MuxGraph
from openfpga_arch_utils import load_circuit_models, MuxCircuitModel

# =============================================================================

class RoutingMuxType(Enum):
    """
    Type of a routing mux instance
    """
    SB = 0
    CBX = 1
    CBY = 2


class RoutingMux:
    """
    A class for describing a routing mux instance
    """

    def __init__(self, type, side, index, node_id, num_inputs):
        self.type = type
        self.side = side
        self.index = index
        self.node_id = node_id
        self.edges = {i: None for i in range(num_inputs)}

        self.model = None
        self.node_side = None

    @property
    def num_inputs(self):
        return len(self.edges)

    def input_node_id(self, index):
        return self.edges[i].src_node

    def get_feature_name(self, bit):
        """
        Returns a FASM feature name corresponding to the i-th memory bit
        There is no encoding done here. Which bits has to be set to enable
        an input has to be known upfront.
        """

        # Type and side string
        if self.type == RoutingMuxType.SB:
            type_str = "track"
            side_str = self.side
        elif self.type in [RoutingMuxType.CBX, RoutingMuxType.CBY]:
            type_str = "ipin"
            side_str = self.node_side

        # Final feature name
        return "mem_{}_{}_{}.mem_out[{}]".format(
            side_str,
            type_str,
            self.index,
            bit
        )

    def get_features_for_input(self, index):
        """
        Returns a list of FASM features needed to enable the specific input
        """
        assert self.model is not None

        # Get bit encoding
        encoding = self.model.graph.get_input_encoding(index)

        # Convert to a series of FASM features
        features = []
        for i, bit in enumerate(encoding):
            if bit is True:
                features.append(self.get_feature_name(i))

        return features

    def __str__(self):
        return "{} node_id={} N={}".format(
            self.type.name,
            self.node_id,
            self.num_inputs,
        )

    def __repr__(self):
        return str(self)


class RoutingMuxModel:
    """
    A model of a routing mux type
    """

    def __init__(self, circuit_model, num_inputs):
        assert isinstance(circuit_model, MuxCircuitModel), circuit_model
        assert num_inputs > 1, num_inputs

        # Build the mux graph
        self.graph = MuxGraph.build(
            num_inputs,
            circuit_model.structure,
            circuit_model.num_levels,
            circuit_model.add_const_input
        )

# =============================================================================


def rr_node_beg_loc(node):
    """
    Returns location of the "entry" to a node
    """
    if node.direction == rr.NodeDirection.INC_DIR:
        return (node.loc.x_low, node.loc.y_low)
    elif node.direction == rr.NodeDirection.DEC_DIR:
        return (node.loc.x_high, node.loc.y_high)
    elif node.direction == rr.NodeDirection.NO_DIR:
        return (node.loc.x_low, node.loc.y_low)
    else:
        assert False, node.direction

def rr_node_end_loc(node):
    """
    Returns location of the "exit" of a node
    """
    if node.direction == rr.NodeDirection.DEC_DIR:
        return (node.loc.x_low, node.loc.y_low)
    elif node.direction == rr.NodeDirection.INC_DIR:
        return (node.loc.x_high, node.loc.y_high)
    elif node.direction == rr.NodeDirection.NO_DIR:
        return (node.loc.x_low, node.loc.y_low)
    else:
        assert False, node.direction

def rr_node_to_string(node):
    """
    Returns a string with node information. For debugging.
    """

    beg_loc = rr_node_beg_loc(node)
    end_loc = rr_node_end_loc(node)

    return "{} id={} ({}, {}) -> ({}, {}) ptc={}".format(
        node.type.name,
        node.id,
        *beg_loc,
        *end_loc,
        node.loc.ptc
    )

# =============================================================================

def offset_loc(loc, ofs):
    """
    Adds offset to a location represented as a tuple of two ints.
    """
    return (loc[0] + ofs[0], loc[1] + ofs[1])

# =============================================================================


def append_edge_metadata(edge, meta_name, meta_value):
    """
    Appends metadata to an rr graph edge. Returns a new Edge object.
    """

    edge_d = edge._asdict()

    if edge_d["metadata"] is None:
        edge_d["metadata"] = []

    edge_d["metadata"].append(rr.NodeMetadata(
        name=meta_name,
        x_offset=0,
        y_offset=0,
        z_offset=0,
        value=meta_value
    ))

    return rr.Edge(**edge_d)

# =============================================================================


def identify_routing_muxes(graph, gsb_data):
    """
    Identify routing muxes by comparing GSB data agains the routing graph.
    Creates a dict indexed by grid coordinates containing lists of routing
    muxes present at those locations.
    """

    # Create a node by id lookup
    nodes_by_id = {node.id: node for node in graph.nodes}

    # Sort edges by their sink nodes
    edges_by_sink = {}
    for edge in graph.edges:

        if edge.sink_node not in edges_by_sink:
            edges_by_sink[edge.sink_node] = set()

        edges_by_sink[edge.sink_node].add(edge)

    # Set of nodes not assigned to any routing mux
    unassigned_nodes = set([
        node.id for node in graph.nodes if \
        node.type in [rr.NodeType.CHANX, rr.NodeType.CHANY, rr.NodeType.IPIN, rr.NodeType.OPIN]
    ])

    undriven_nodes = set([
        node.id for node in graph.nodes if \
        node.type in [rr.NodeType.CHANX, rr.NodeType.CHANY, rr.NodeType.IPIN]
    ])

    # Process GSBs
    routing_muxes = {}
    for gsb_loc, gsbs in progressbar_utils.progressbar(gsb_data.items()):

        # Process each GSB mux
        for gsb in gsbs:

            # Get the rr node
            assert gsb.node.id in nodes_by_id, gsb.node
            node = nodes_by_id[gsb.node.id]

            # Verify node type
            assert node.type.name.upper() == gsb.node.type, \
                (node.type.name.upper(), gsb.node.type)

            # A GSB must have at least one driver
            if not gsb.drivers:
                print("WARNING: A GSB mux for {} must have a driver".format(
                    rr_node_to_string(node)
                ))
                continue

            # This is a passthrough connection, do not create a mux for it
            if len(gsb.drivers) == 1:

                # Mark the output node as driven.
                if gsb.node.id != gsb.drivers[0].id:
                    undriven_nodes.discard(gsb.node.id)
                    unassigned_nodes.discard(gsb.node.id)
                    unassigned_nodes.discard(gsb.drivers[0].id)

                continue

            # The node must be undriven
            assert node.id in undriven_nodes, rr_node_to_string(node)

            # Check if there are any edges
            if node.id not in edges_by_sink:
                print("WARNING: No edges to {}".format(
                    rr_node_to_string(node)
                ))
                continue

            # Incoming rr edge count must match the GSB mux input count
            assert len(edges_by_sink[node.id]) == len(gsb.drivers), \
                (len(edges_by_sink[node.id]), len(gsb.drivers), gsb.mux_size)

            # Determine mux type
            driver_types = set([drv.type for drv in gsb.drivers])
            assert "IPIN" not in driver_types, (gsb.node, driver_types)

            # Switchbox
            if gsb.node.type in ["CHANX", "CHANY"]:
                mux_type = RoutingMuxType.SB

            # Connection box
            elif gsb.node.type == "IPIN":

                if "CHANX" in driver_types and not "CHANY" in driver_types:
                    mux_type = RoutingMuxType.CBX
                elif "CHANY" in driver_types and not "CHANX" in driver_types:
                    mux_type = RoutingMuxType.CBY
                else:
                    assert False, (gsb.node, driver_types)

            else:
                assert False, gsb.node

            # Create the mux
            mux = RoutingMux(
                mux_type,
                gsb.node.side,
                gsb.node.index,
                node.id,
                len(gsb.drivers)
            )

            if node.type == rr.NodeType.IPIN:
                mux.node_side = node.loc.side.name.lower()

            # Create its edges
            for i, gsb_node in enumerate(gsb.drivers):

                # Cannot loopback
                assert gsb_node.id != gsb.node.id, \
                    rr_node_to_string(nodes_by_id[gsb.node.id])

                # Get the rr node
                assert gsb_node.id in nodes_by_id, gsb_node
                node = nodes_by_id[gsb_node.id]

                # Verify node type
                assert node.type.name.upper() == gsb_node.type, \
                    (node.type.name.upper(), gsb_node.type)

                # Find the edge
                for e in edges_by_sink[mux.node_id]:
                    if e.src_node == node.id:
                        edge = e
                        break
                else:
                    print("WARNING: No rr edge from {} to {}".format(
                        rr_node_to_string(nodes_by_id[gsb_node.id]),
                        rr_node_to_string(nodes_by_id[mux.node_id]),
                    ))
                    continue

                # Store the edge
                mux.edges[i] = edge

            # Check if we found any edges. If no then skip adding the mux
            if not any([e is not None for e in mux.edges.values()]):
                print("WARNING: No edges to {} found".format(
                    rr_node_to_string(nodes_by_id[mux.node_id])
                ))
                continue

            # Add the mux
            loc = gsb_loc
            if loc not in routing_muxes:
                routing_muxes[loc] = []
            routing_muxes[loc].append(mux)

            # Mark nodes used by the mux
            undriven_nodes.discard(mux.node_id)
            unassigned_nodes.discard(mux.node_id)
            for edge in mux.edges.values():
                unassigned_nodes.discard(edge.src_node)

    # DEBUG - compute and print some statistics #
    mux_count = 0
    num_sb = 0
    num_cbx = 0
    num_cby = 0
    for muxes in routing_muxes.values():
        mux_count += len(muxes)
        for mux in muxes:
            if mux.type == RoutingMuxType.SB:
                num_sb += 1
            if mux.type == RoutingMuxType.CBX:
                num_cbx += 1
            if mux.type == RoutingMuxType.CBY:
                num_cby += 1

    print(" {:>6} SB routing muxes".format(num_sb))
    print(" {:>6} CBX routing muxes".format(num_cbx))
    print(" {:>6} CBY routing muxes".format(num_cby))
    print(" {:>6} total routing muxes".format(mux_count))

    num_undriven_chan = len([n for n in undriven_nodes \
        if nodes_by_id[n].type in [rr.NodeType.CHANX, rr.NodeType.CHANY]])
    num_undriven_ipin = len([n for n in undriven_nodes \
        if nodes_by_id[n].type in [rr.NodeType.IPIN]])

    print(" {:>6} undriven CHAN rr nodes".format(num_undriven_chan))
    print(" {:>6} undriven IPIN rr nodes".format(num_undriven_ipin))
    print(" {:>6} total undriven rr nodes".format(len(undriven_nodes)))

    num_unassigned_chan = len([n for n in unassigned_nodes \
        if nodes_by_id[n].type in [rr.NodeType.CHANX, rr.NodeType.CHANY]])
    num_unassigned_ipin = len([n for n in unassigned_nodes \
        if nodes_by_id[n].type in [rr.NodeType.IPIN]])
    num_unassigned_opin = len([n for n in unassigned_nodes \
        if nodes_by_id[n].type in [rr.NodeType.OPIN]])

    print(" {:>6} unassigned CHAN rr nodes".format(num_unassigned_chan))
    print(" {:>6} unassigned IPIN rr nodes".format(num_unassigned_ipin))
    print(" {:>6} unassigned OPIN rr nodes".format(num_unassigned_opin))
    print(" {:>6} total unassigned rr nodes".format(len(unassigned_nodes)))

    return routing_muxes

# =============================================================================


def build_routing_muxes_models(circuit_models, routing_muxes):
    """
    Identifies common routing mux types and builds their models (with encoding)
    """

    # Get the default mux circuit model
    circuit_model = None
    for circuit in circuit_models.values():
        if circuit.type == "mux" and circuit.is_default:
            circuit_model = circuit
            break
    else:
        print(" ERROR: No default mux circuit model!")
        exit(-1)

    # Build mux models
    mux_sizes = {}
    mux_models = {}
    for loc, muxes in progressbar_utils.progressbar(routing_muxes.items()):
        for mux in muxes:

            # Create or use an existing one
            key = mux.num_inputs
            if key not in mux_models:
                model = RoutingMuxModel(circuit_model, mux.num_inputs)
                mux_models[key] = model
            else:
                model = mux_models[key]

            # Set the model
            mux.model = model

            # Statistics
            size = mux.num_inputs
            if size not in mux_sizes:
                mux_sizes[size] = 0
            mux_sizes[size] += 1

    # Print size statistics
    print(" mux size statistics:")
    for size in sorted(list(mux_sizes.keys())):
        print(" {:>6} of size {}".format(mux_sizes[size], size))

    return mux_models


def annotate_edges(graph, routing_muxes):
    """
    Injects FASM annotation for edges that correspont to routing muxes.
    """

    # First remove all the existing mux edges from the graph. They will be
    # replaced with annotated ones.
    mux_edges = set()
    for loc, muxes in routing_muxes.items():
        for mux in muxes:
            mux_edges |= set(mux.edges.values())

    graph.edges = list(set(graph.edges) - mux_edges)

    # Annotate and add edges for routing muxes
    new_edges = []
    for loc, muxes in progressbar_utils.progressbar(routing_muxes.items()):
        for mux in muxes:

            # Determine correct location for this mux
            if mux.type == RoutingMuxType.CBY:
                mux_loc = offset_loc(loc, (0, 1))
            else:
                mux_loc = loc

            # Emmit edges
            for i, edge in mux.edges.items():
                index = i

                # Get FASM features required to activate the given input
                features = mux.get_features_for_input(index)

                # Prefix the features
                for i in range(len(features)):
                    features[i] = "fpga_top.{}_{}__{}_.{}".format(
                        mux.type.name.lower(),
                        mux_loc[0],
                        mux_loc[1],
                        features[i]
                    )

                # Add a new edge if there are any FASM features. Otherwise
                # re-use the old one
                if features:
                    metadata = "\n".join(features)
                    new_edge = append_edge_metadata(edge, "fasm_features", metadata)
                    new_edges.append(new_edge)

                else:
                    new_edges.append(edge)

    # Append all the edges
    graph.edges.extend(new_edges)
    print(" {:>6} edges annotated".format(len(new_edges)))

# =============================================================================


def compute_rr_graph_stats(graph):
    """
    Compute some rr graph statistics useful for inspection / debugging
    """
    nodes_by_id = {n.id: n for n in graph.nodes}

    edges_by_src = {}
    edges_by_dst = {}

    for e in graph.edges:

        if e.src_node not in edges_by_src:
            edges_by_src[e.src_node] = set()
        edges_by_src[e.src_node].add(e)

        if e.sink_node not in edges_by_dst:
            edges_by_dst[e.sink_node] = set()
        edges_by_dst[e.sink_node].add(e)

    ipin_count = len([n for n in nodes_by_id.values() if \
                      n.type == rr.NodeType.IPIN])

    opin_count = len([n for n in nodes_by_id.values() if \
                      n.type == rr.NodeType.OPIN])

    chan_count = len([n for n in nodes_by_id.values() if \
                      n.type in [rr.NodeType.CHANX, rr.NodeType.CHANY]])

    direct_count = len([e for e in graph.edges if \
                        nodes_by_id[e.src_node].type == rr.NodeType.OPIN and \
                        nodes_by_id[e.sink_node].type == rr.NodeType.IPIN])

    unroutable_ipins = 0
    unroutable_opins = 0
    undriven_chans = 0
    dangling_chans = 0

    for node in nodes_by_id.values():

        if node.type == rr.NodeType.IPIN:
            for edge in edges_by_dst.get(node.id, []):
                other = nodes_by_id[edge.src_node]
                if other.type in [rr.NodeType.CHANX, rr.NodeType.CHANY]:
                    break
            else:
                unroutable_ipins += 1

        elif node.type == rr.NodeType.OPIN:

            for edge in edges_by_src.get(node.id, []):
                other = nodes_by_id[edge.sink_node]
                if other.type in [rr.NodeType.CHANX, rr.NodeType.CHANY]:
                    break
            else:
                unroutable_opins += 1

        elif node.type in [rr.NodeType.CHANX, rr.NodeType.CHANY]:

            for edge in edges_by_dst.get(node.id, []):
                other = nodes_by_id[edge.src_node]
                if other.type in [rr.NodeType.CHANX, rr.NodeType.CHANY, rr.NodeType.OPIN]:
                    break
            else:
                undriven_chans += 1

            for edge in edges_by_src.get(node.id, []):
                other = nodes_by_id[edge.sink_node]
                if other.type in [rr.NodeType.CHANX, rr.NodeType.CHANY, rr.NodeType.IPIN]:
                    break
            else:
                dangling_chans += 1

    print("Graph stats:")
    print(" {:>6} IPIN nodes".format(ipin_count))
    print(" {:>6} OPIN nodes".format(opin_count))
    print(" {:>6} CHAN nodes".format(chan_count))
    print(" {:>6} direct connections".format(direct_count))
    print(" {:>6} unroutable IPINs".format(unroutable_ipins))
    print(" {:>6} unroutable OPINs".format(unroutable_opins))
    print(" {:>6} undriven CHANs".format(undriven_chans))
    print(" {:>6} dangling CHANs".format(dangling_chans))


def yield_edges(edges):
    """
    Yields edges in a format acceptable by the graph serializer.
    """
    conns = set()

    # Process edges
    for edge in edges:

        # Reformat metadata
        if edge.metadata:
            metadata = [(meta.name, meta.value) for meta in edge.metadata]
        else:
            metadata = None

        # Check for repetition
        if (edge.src_node, edge.sink_node) in conns:
            print(
                "WARNING: Removing duplicated edge from {} to {}, metadata='{}'"
                .format(edge.src_node, edge.sink_node, metadata)
            )
            continue

        conns.add((edge.src_node, edge.sink_node))

        # Yield the edge
        yield (edge.src_node, edge.sink_node, edge.switch_id, metadata)

# =============================================================================


class BinaryGraphWriter(rr_capnp.Graph):
    """
    A proxy class for binary graph writing that leaverages the existing code
    from SymbiFlow.

    This class overloads the rr_graph_capnp.graph2 class to substitute a
    constructor that does not require loading the graph from a binary file.
    This way an externally loaded graph can be used.
    """

    def __init__(
            self,
            graph,
            root_attrib,
            capnp_schema_file_name,
            output_file_name,
            progressbar=None
    ):

        if progressbar is None:
            progressbar = lambda x: x  # noqa: E731

        self.progressbar = progressbar

        self.input_file_name = None
        self.output_file_name = output_file_name

        self.graph = graph
        self.root_attrib = root_attrib

        # Handle multiple paths for the capnp library
        search_path = [os.path.dirname(os.path.dirname(capnp.__file__))]
        if 'CAPNP_PATH' in os.environ:
            search_path.append(os.environ['CAPNP_PATH'])

        for path in ['/usr/local/include', '/usr/include']:
            if os.path.exists(path):
                search_path.append(path)

        # Load the Cap'n'proto schema
        self.rr_graph_schema = capnp.load(
            capnp_schema_file_name,
            imports=search_path
        )

# =============================================================================

def main():

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rr-graph-in",
        required=True,
        type=str,
        help="Input RR graph XML"
    )
    parser.add_argument(
        "--rr-graph-out",
        required=True,
        type=str,
        help="Output RR graph XML"
    )
    parser.add_argument(
        "--gsb",
        default=None,
        type=str,
        help="Path to GSB data files"
    )
    parser.add_argument(
        "--openfpga-arch-in",
        type=str,
        required=True,
        help="OpenFPGA arch.xml input"
    )
    parser.add_argument(
        "--capnp-schema",
        default=None,
        type=str,
        help="Path to rr graph cap'n'proto schema (for binary rr graph writing)"
    )

    args = parser.parse_args()

    # Verify the output rr graph extension
    rr_graph_out_ext = os.path.splitext(args.rr_graph_out)[1].lower()

    if rr_graph_out_ext == ".xml":
        pass

    if rr_graph_out_ext == ".bin":

        # Check if we have Cap'n'proto schema provided
        if args.capnp_schema is None:
            print("ERROR: Binary rr graph writeout requires Cap'n'proto schema")
            exit(-1)

    else:
        print("ERROR: Unsupported rr graph extension '{}'".format(rr_graph_out_ext))
        exit(-1)

    # Read and parse the OpenFPGA arch XML file
    xml_tree = ET.parse(args.openfpga_arch_in, ET.XMLParser(remove_blank_text=True))
    xml_openfpga_arch = xml_tree.getroot()
    assert xml_openfpga_arch is not None and \
        xml_openfpga_arch.tag == "openfpga_architecture"

    # Get circuit models
    circuit_models = load_circuit_models(xml_openfpga_arch)

    # Load the routing graph
    print("Loading rr graph...")
    xml_graph = rr_xml.Graph(
        input_file_name=args.rr_graph_in,
        output_file_name=args.rr_graph_out,
        rebase_nodes=False,
        filter_nodes=False,
        load_edges=True,
        build_pin_edges=False,
        progressbar=progressbar_utils.progressbar
    )

    compute_rr_graph_stats(xml_graph.graph)

    # Load GSB data
    if args.gsb is not None:
        print("Loading GSB data...")
        gsb_data = load_gsb_data(args.gsb, progressbar_utils.progressbar)
        assert len(gsb_data), "No GSB data found!"

        # DEBUG
        total_mux_count = 0
        conf_mux_count = 0
        for loc, gsbs in gsb_data.items():
            total_mux_count += len(gsbs)
            for gsb in gsbs:
                if len(gsb.drivers) > 1:
                    conf_mux_count += 1

        print(" {:>6} locs".format(len(gsb_data)))
        print(" {:>6} all muxes".format(total_mux_count))
        print(" {:>6} configurable muxes".format(conf_mux_count))

        # Identify routing muxes
        print("Identifying routing muxes...")
        routing_muxes = identify_routing_muxes(xml_graph.graph, gsb_data)

        # Build routing mux models
        print("Building routing muxes models...")
        routing_muxes_models = build_routing_muxes_models(circuit_models, routing_muxes)

        # Annotate edges
        print("Annotating rr graph edges...")
        annotate_edges(xml_graph.graph, routing_muxes)

    # Re-add all channel nodes as tracks and clear their PTCs. This allows
    # the Graph2 class code to re-assign PTCs which makes the graph usable
    # for SymbiFlow VPR.
    for i, node in enumerate(xml_graph.graph.nodes):

        # Check endpoint locations
        assert node.loc.x_low <= node.loc.x_high, node
        assert node.loc.y_low <= node.loc.y_high, node

        # Handle CHAN node
        if node.type in [rr.NodeType.CHANX, rr.NodeType.CHANY]:

            # Verify direction
            assert node.direction in [
                rr.NodeDirection.INC_DIR,
                rr.NodeDirection.DEC_DIR,
                rr.NodeDirection.BI_DIR
            ], node

            # Named tuples to dicts
            node = node._asdict()
            loc = node["loc"]._asdict()

            # Clear PTC
            loc["ptc"] = None

            # Not sure why the segment id has its own NamedTuple object but
            # the Graph2 code requires it.
            node["segment"] = rr.NodeSegment(segment_id=node["segment"])

            # Dicts to named tuples
            loc = rr.NodeLoc(**loc)
            node["loc"] = loc
            node = rr.Node(**node)

            # Replace the node
            xml_graph.graph.nodes[i] = node

            # Add it as a track
            xml_graph.graph.tracks.append(node.id)

        # Handle non-CHAN node
        else:

            # Verify direction
            assert node.direction == rr.NodeDirection.NO_DIR, node

            # Set direction to None
            node = node._asdict()
            node["direction"] = None
            node = rr.Node(**node)

            # Replace the node
            xml_graph.graph.nodes[i] = node

    # Create channels from tracks
    channels_obj = xml_graph.graph.create_channels(pad_segment=0)

    # Remove padding channels
    print("Removing padding nodes...")
    xml_graph.graph.nodes = [
        n for n in xml_graph.graph.nodes if n.capacity > 0
    ]

    # Write the routing graph
    nodes_obj = xml_graph.graph.nodes
    edges_obj = xml_graph.graph.edges
    node_remap = lambda x: x

    print("Serializing the rr graph...")

    if rr_graph_out_ext == ".xml":
        xml_graph.serialize_to_xml(
            channels_obj=channels_obj,
            nodes_obj=nodes_obj,
            edges_obj=yield_edges(edges_obj),
            node_remap=node_remap,
        )

    elif rr_graph_out_ext == ".bin":

        # Build the writer
        writer = BinaryGraphWriter(
            graph=xml_graph.graph,
            root_attrib=xml_graph.root_attrib,
            capnp_schema_file_name=args.capnp_schema,
            output_file_name=args.rr_graph_out,
            progressbar=progressbar_utils.progressbar,
        )

        # Write
        writer.serialize_to_capnp(
            channels_obj=channels_obj,
            num_nodes=len(nodes_obj),
            nodes_obj=nodes_obj,
            num_edges=len(edges_obj),
            edges_obj=yield_edges(edges_obj),
        )

    else:
        assert False, rr_graph_out_ext


# =============================================================================


if __name__ == "__main__":
    main()
