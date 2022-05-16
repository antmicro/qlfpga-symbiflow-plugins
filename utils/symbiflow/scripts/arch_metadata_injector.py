#!/usr/bin/env Python3
"""
This script contains method for patchin VPR arch.xml in order to include
pb_type medatata which mostly include FASM annotations compatible with
OpenFPGA bistream entities naming scheme.
"""
import re
import json

import lxml.etree as ET

from openfpga_arch_utils import CircuitModel, MuxCircuitModel, LutCircuitModel
from openfpga_arch_utils import load_circuit_models

from vpr_arch_utils import is_leaf_pbtype
from vpr_arch_utils import append_metadata
from vpr_arch_utils import yield_pins
from vpr_arch_utils import get_pb_and_port

from mux_graph import Graph

# =============================================================================

class PhysicalAnnotation:
    """
    This class represents a physical annotation of a pb_type which binds it
    to a physical circuit model.
    """
    def __init__(self, path):
        self.path = path
        self.circuit_model = None
        self.mode_bits = set()

    def __str__(self):
        return "{}: circuit={} mode_bits={}".format(
            self.path,
            self.circuit_model,
            self.mode_bits
        )

    def __repr__(self):
        return str(self)


class ConfigProtocol:
    """
    Represents OpenFPGA configuration protocol settings
    """
    def __init__(self, type, circuit_model):
        self.type = type
        self.circuit_model = circuit_model

# =============================================================================


def load_physical_modes(xml_root):
    """
    Gets physical modes
    """

    # Get "pb_type_annotation"
    xml_rules = xml_root.find("pb_type_annotations")
    assert xml_rules is not None

    # Examine all annotations
    physical_modes = {}
    for xml_rule in xml_rules.findall("pb_type"):
        if "name" in xml_rule.attrib and \
           "physical_mode_name" in xml_rule.attrib:

            path = xml_rule.attrib["name"]
            mode = xml_rule.attrib["physical_mode_name"]

            physical_modes[path] = mode

    # DEBUG
    print("Physical modes:")
    for k, v in physical_modes.items():
        print(" {}[{}]".format(k, v))

    return physical_modes


def load_physical_annotations(xml_root):
    """
    Gets physical annotations from OpenFPGA arch XML
    """

    # Get "pb_type_annotation"
    xml_rules = xml_root.find("pb_type_annotations")
    assert xml_rules is not None

    # Examine all annotations
    annotations = {}

    for xml_annotation in xml_rules.findall("pb_type"):
        assert "name" in xml_annotation.attrib

        # If this is logical -> physical pb map then use the physical pb_type
        # as the path
        physical_pb_type_name = xml_annotation.get("physical_pb_type_name", None)
        if physical_pb_type_name is not None:
            path = physical_pb_type_name
        else:
            path = xml_annotation.attrib["name"]

        mode_bits = xml_annotation.get("mode_bits", None)

        # Got a circuit model binding
        circuit_model = xml_annotation.get("circuit_model_name", None)
        if circuit_model:

            if path not in annotations:
                annotations[path] = PhysicalAnnotation(path)

            annotations[path].circuit_model = circuit_model

            if mode_bits:
                annotations[path].mode_bits.add(mode_bits)

        # Got a logical -> physical pb_type map
        elif physical_pb_type_name is not None:

            if path not in annotations:
                annotations[path] = PhysicalAnnotation(path)

            if mode_bits:
                annotations[path].mode_bits.add(mode_bits)

        # Check interconnect annotations
        for xml_ic in xml_annotation.findall("interconnect"):
            ic_path = path + "." + xml_ic.attrib["name"]

            # Got a circuit model binding
            circuit_model = xml_ic.get("circuit_model_name", None)
            if circuit_model:

                if ic_path not in annotations:
                    annotations[ic_path] = PhysicalAnnotation(ic_path)

                annotations[ic_path].circuit_model = circuit_model

    # DEBUG
    print("Physical annotations:")
    keys = sorted(list(annotations.keys()))
    for k in keys:
        print("", annotations[k])

    return annotations


def load_config_protocol(xml_root):
    """
    Parses the configuration_protocol section
    """

    # Get "configuration_protocol"
    xml_config = xml_root.find("configuration_protocol")
    assert xml_config is not None

    # Get "organization"
    xml_org = xml_config.find("organization")
    assert xml_org is not None

    # Get relevant info
    return ConfigProtocol(
        type = xml_org.attrib["type"],
        circuit_model = xml_org.attrib["circuit_model_name"]
    )

# =============================================================================


def identify_physical_entities(xml_arch, physical_modes):
    """
    Identifies all VPR arch XML entities that require FASM annotation because
    they belong to a physical pb_type mode. The algorithm implemented below
    takes into account all pb_types (and interconnects) above a physical
    pb_type.
    """

    # Format full physical mode paths
    phy_paths = set(["{}[{}]".format(k, v) for k, v in physical_modes.items()])

    # Get the complexblocklist section
    xml_cplxblocklist = xml_arch.find("complexblocklist")
    assert xml_cplxblocklist is not None


    physical_pb_types = set()
    physical_entities = set()

    # Walks backwards from a leaf pb_type towards the root pb_type. Adds
    # pb_types on the path and their interconnect elements to the set of
    # physical entities
    def walk_bwd(xml_block, path):

        while True:

            # Add this pb_type
            if xml_block.tag == "pb_type":
                physical_entities.add(path)

            # Add this pb_type interconnect elements
            xml_ic = xml_block.find("interconnect")
            if xml_ic is not None:
                for xml_conn in xml_ic:

                    # Check if it is an XML element (not a comment)
                    if not isinstance(xml_conn.tag, str):
                        continue

                    ic_name = xml_conn.attrib["name"]
                    ic_path = path + "." + ic_name
                    physical_entities.add(ic_path)

            # Step back
            # FIXME: This path parsing is crude. Maybe use regex here?
            if xml_block.tag == "pb_type":
                path = ".".join(path.split(".")[:-1])
            elif xml_block.tag == "mode":
                path = path.rsplit("[", maxsplit=1)[0]

            # Get parent
            xml_parent = xml_block.getparent()

            # We've hit the complexblocklist. Stop
            if xml_parent is None or xml_parent.tag == "complexblocklist":
                break

            assert path
            xml_block = xml_parent

    # Recursive walk starting from a root pb_type. Identifies all leaf pb_types
    # locaded under physical modes
    def walk_fwd(xml_pbtype, path=None):

        # Initialize hierarchical path
        if path is None:
            path = ""

        # Append self name
        path += "{}".format(xml_pbtype.attrib["name"])

        # Identify all modes and store their XML roots
        xml_modes = {m.attrib["name"]: m for m in xml_pbtype.findall("mode")}

        # No explicit modes, insert the default mode
        if not xml_modes:
            xml_modes = {"default": xml_pbtype}

        # Process each mode or a pb_type if the mode is implicit
        for mode, xml_mode in xml_modes.items():

            # Append the mode
            if mode != "default":
                curr_path = path + "[{}]".format(mode)
            else:
                curr_path = path

            # Check if the current path is a part of any physical path
            # If not then do not add annotation.
            is_under_physical_mode = False

            for phy_path in phy_paths:
                if curr_path.startswith(phy_path):
                    is_under_physical_mode = True
                    break

            # Process children
            for xml_pb in xml_mode.findall("pb_type"):
                child_name = xml_pb.attrib["name"]
                child_path = curr_path + "." + child_name

                # If under a physical mode then store and process leaf children
                if is_leaf_pbtype(xml_pb) and is_under_physical_mode:
                    physical_pb_types.add(child_path)
                    walk_bwd(xml_pb, child_path)

                # Recurse
                walk_fwd(xml_pb, curr_path + ".")

    # Recurse for each CLB
    for xml_pbtype in xml_cplxblocklist.findall("pb_type"):
        walk_fwd(xml_pbtype)

    # DEBUG
    print("Physical pb_types (leaves):")
    for p in sorted(physical_pb_types):
        print("", p)

    print("Physical enitites:")
    for p in sorted(physical_entities):
        print("", p)

    return physical_entities

# =============================================================================


def build_hierarchical_name(xml_item, name=None, is_last=True):
    """
    Builds a OpenFPGA compatible hierarchical name for a pb_type by traversing
    the hierarchy up to the root.
    """

    # Initialize the name if not given
    if name is None:
        name = ""

    # This is a pb_type
    if xml_item.tag == "pb_type":
        part = "{}_".format(xml_item.attrib["name"])

        # Not last
        if not is_last:
            # No explicit modes and not a leaf
            if not xml_item.findall("mode") and not is_leaf_pbtype(xml_item):
                part += "mode_default__"

    # This is a mode
    elif xml_item.tag == "mode":
        part = "mode_{}__".format(xml_item.attrib["name"])

    # We've hit complexblocklist
    elif xml_item.tag == "complexblocklist":
        return "logical_tile_" + name

    else:
        assert False, xml_item.tag

    # Recurse
    parent = xml_item.getparent()
    return build_hierarchical_name(parent, part + name, is_last=False)

# =============================================================================


INDEXED_NAME_RE = re.compile(r"(?P<name>[A-Za-z0-9_]+)(\[(?P<index>[0-9:]+)\])?")

def convert_indices(port_spec, force_pb_index=False):
    """
    Convert port specification given as "<pb_type>[<pb_index>].<port>[<bit>]"
    (indices are optional!) into "<pb_type>_<pb_index>_<port>_<bit>". The bit
    index is always present, the pb_index is omitted if not given in the
    original string.
    """

    # FIXME: Double check if the naming scheme generated by this function is
    # always compatible with OpenFPGA.

    # Separate pb_type and port
    pb_type, port = port_spec.split(".")

    parts = []
    for part in [pb_type, port]:

        # Match the name
        match = INDEXED_NAME_RE.fullmatch(part)
        assert match is not None, part

        # Reformat
        name = match.group("name")
        indx = match.group("index")

        # Enforce index for the port poart
        if indx is None and (part == port or force_pb_index):
            indx = "0"

        # Append name with index
        if indx is not None:

            # Make sure that even in case of a range expression we are
            # referencing a single bit
            if ":" in indx:
                i0, i1 = indx.split(":")
                assert i0 == i1, indx
                indx = i0

            parts.append("{}_{}".format(name, indx))

        # Append just the name
        else:
            parts.append(name)


    return "_".join(parts)


# =============================================================================


def annotate_leaf_pbtype(xml_pbtype, path, circuit_models, physical_annotations, suffix):
    """
    Annotates a leaf pb_type with FASM features / parameters. The pb_type must
    correspond to a physical annotation and have a circuit model associated.
    """

    # FIXME
    if path not in physical_annotations:
        return

    # Get physical annotation
    assert path in physical_annotations, path
    annotation = physical_annotations[path]

    # Get circuit model
    assert annotation.circuit_model in circuit_models, annotation.circuit_model
    circuit = circuit_models[annotation.circuit_model]

    # Initial offset
    mem_offset = 0

    # FIXME: The assumption is that mode bits always follow non-mode bits.
    # and that both refer to the same "mem_out" feature.

    # Check if mode bit count from physical pb_type annotation matches the
    # count from the circuit model.
    for mode_bits in annotation.mode_bits:
        len_mode_bits = len(mode_bits)
        assert len_mode_bits == circuit.mode_bits, \
            (circuit.name, circuit.mode_bits, annotation.mode_bits, mode_bits)

    # This is a lut
    if circuit.type == "lut":
        width = (circuit.conf_bits - 1).bit_length()
        size = (1 << width)

        feature = "{}_{}_mem.mem_out[{}:{}]".format(
            circuit.name,
            suffix,
            mem_offset + size - 1,
            mem_offset
        )
        fasm_params = "{} = LUT".format(feature)

        print(" {}: fasm_params=\"{}\"".format(path, fasm_params))
        append_metadata(xml_pbtype, "fasm_params", fasm_params)

        mem_offset += size

        # Append metadata describing the LUT and its ports
        assert isinstance(circuit, LutCircuitModel), type(circuit)
        append_metadata(xml_pbtype, "class", "lut")

        port_data = {p: {k: v for k, v in a.items() if k != "size"} for p, a in circuit.ports.items()}
        append_metadata(xml_pbtype, "lut_ports", json.dumps(port_data, sort_keys=True))

    # We have a single mode bit possibility, emit a feature
    if len(annotation.mode_bits) == 1:
        mode_bit = next(iter(annotation.mode_bits))
        index_str = "[{}]".format(mem_offset)

        # FIXME: Correct feature name !
        feature = "{}_{}_mem.mem_out{}=1'b{}".format(
            circuit.name,
            suffix,
            index_str,
            mode_bit
        )

        print(" {}: fasm_feature=\"{}\"".format(path, feature))
        append_metadata(xml_pbtype, "fasm_features", feature)

        mem_offset += 1

    # We have multiple mode bit possibilities, make a FASM parameter
    elif len(annotation.mode_bits) > 1:
        width = (circuit.mode_bits - 1).bit_length()
        size = (1 << width)

        if size == 1:
            index_str = "[{}]".format(mem_offset)
        else:
            index_str = "[{}:{}]".format(mem_offset + size - 1, mem_offset)

        # FIXME: Correct feature name !
        feature = "{}_{}_mem.mem_out{}".format(
            circuit.name,
            suffix,
            index_str
        )
        fasm_params = "{} = MODE".format(feature)

        print(" {}: fasm_feature=\"{}\"".format(path, fasm_params))
        append_metadata(xml_pbtype, "fasm_params", fasm_params)

        mem_offset += size


def annotate_interconnect(xml_conn, path, circuit_models, physical_annotations, suffix):
    """
    Annotates an interconnect with FASM featurees. The interconnect element
    must have a physical annotation and a circuit model associated.
    """

    # Explicit annotation
    if path in physical_annotations:

        annotation = physical_annotations[path]

        # Get circuit model
        assert annotation.circuit_model in circuit_models, annotation.circuit_model
        circuit = circuit_models[annotation.circuit_model]

    # Use default circuit model
    else:

        # Determine appropriate circuit type
        assert xml_conn.tag in ["mux", "direct"], xml_conn.tag
        circuit_type = {"mux": "mux", "direct": "wire"}[xml_conn.tag]

        # Find the default circuit
        for circuit in circuit_models.values():
            if circuit.type == circuit_type and circuit.is_default:
                break
        else:
            assert False, path

    # This is a mux
    if xml_conn.tag == "mux":

        xml_ic = xml_conn.getparent()

        # Get ports
        inp_names = xml_conn.attrib["input"].split()
        out_name  = xml_conn.attrib["output"]

        # Get the port XML element. If the interconnect drives a pb_type input
        # we need an extra prefix in the FASM feature name.
        _, xml_port = get_pb_and_port(xml_ic, out_name)
        is_pb_input = xml_port.tag in ["input", "clock"]

        # Reformat input names so that there are no range expressions
        for i in range(len(inp_names)):
            names = list(yield_pins(xml_ic, inp_names[i]))
            assert len(names) == 1, (inp_names[i], names)

            inp_names[i] = names[0]

        xml_conn.set("input", " ".join(inp_names))

        # Get mux width
        width = len(inp_names)

        # Build the mux graph
        assert isinstance(circuit, MuxCircuitModel), type(circuit)
        mux_graph = Graph.build(
            num_inputs = width,
            impl_structure = circuit.structure,
            num_levels = circuit.num_levels,
            add_const_input = circuit.add_const_input
        )

        # Build a set of features for each input
        features = []
        for i, inp_name in enumerate(inp_names):

            # Get the input encoding
            index = i
            #if circuit.add_const_input:
            #    index += 1 # FIXME: Not sure if this is what OpenFPGA does

            encoding = mux_graph.get_input_encoding(index)

            # Convert to a series of FASM features
            feature = []
            for i, bit in enumerate(encoding):
                if bit is True:
                    feature.append("mem_{}.mem_out[{}]".format(
                        convert_indices(out_name, is_pb_input), i
                    ))

            # When no features should be set use the special NULL word
            if not feature:
                feature = "NULL"
            else:
                feature = ",".join(feature)

            features.append("{} : {}".format(inp_name, feature))

        # Append the mux metadata
        append_metadata(xml_conn, "fasm_mux", "\n" + "\n".join(features))

        for f in features:
            print("  {}: fasm_mux=\"{}\"".format(path, f))

# =============================================================================


def append_delays(xml_conn, inputs, output, delays):
    """
    This function appends delays to a mux or direct, specifically during the translation
    of a complete crossbar into a series of muxes.
    """

    for in_port in inputs:
        generic_in_port = re.sub(r"\[[0-9:]+\]", "", in_port)
        generic_out_port = re.sub(r"\[[0-9:]+\]", "", output)

        if generic_in_port not in delays:
            continue

        max_delay, min_delay, out_port = delays[generic_in_port]

        assert out_port == generic_out_port, (output, generic_out_port, delays)

        if max_delay is None and min_delay is not None:
            max_delay = min_delay
        if max_delay is not None and min_delay is None:
            min_delay = max_delay

        if max_delay is None and min_delay is None:
            continue

        ET.SubElement(xml_conn, "delay_constant", {
            "max": max_delay,
            "min": min_delay,
            "in_port": in_port,
            "out_port": output
        })

# =============================================================================


def process_interconnect(xml_ic, path, physical_entities, circuit_models, physical_annotations, suffix):
    """
    Processes an interconnect. Checks each child connection tag if it requires
    FASM annotation and if one does invokes annotate_interconnect() for it.
    """

    # Process all interconnect elements
    for xml_conn in xml_ic:

        # Check if it is an XML element (not a comment)
        if not isinstance(xml_conn.tag, str):
            continue

        # Check if this element should be annotated
        ic_name = xml_conn.attrib["name"]
        ic_path = path + "." + ic_name

        if ic_path not in physical_entities:
            continue

        # If this is a "complete" interconnect then split it into individual
        # "mux" interconnects.
        if xml_conn.tag == "complete":

            delays = dict()
            for delay in xml_conn.findall(".//delay_constant"):
                attrs = delay.attrib
                max_delay = attrs.get("max", None)
                min_delay = attrs.get("min", None)
                out_port = attrs["out_port"]
                in_port = attrs["in_port"]

                in_port = re.sub(r"\[[0-9:]+\]", "", in_port)
                out_port = re.sub(r"\[[0-9:]+\]", "", out_port)

                delays[in_port] = (max_delay, min_delay, out_port)

            print(" Processing <complete> interconnect '{}' ...".format(ic_name))
            xml_conns = []

            # Collect inputs and outputs
            inputs = []
            for port_spec in xml_conn.attrib["input"].split():
                inputs.extend(list(yield_pins(xml_ic, port_spec)))

            outputs = []
            for port_spec in xml_conn.attrib["output"].split():
                outputs.extend(list(yield_pins(xml_ic, port_spec)))

            # Only one input, make directs
            if len(inputs) == 1:
                inp = inputs[0]

                for out in outputs:
                    conn = ET.Element("direct", {
                        "name": "{}_{}_to_{}".format(ic_name, inp, out),
                        "input": inp,
                        "output": out
                    })

                    append_delays(conn, inputs, out, delays)

                    xml_ic.append(conn)
                    xml_conns.append(conn)

            # Multiple inputs, make a mux for each output
            else:
                inp = " ".join(inputs)
                for out in outputs:
                    conn = ET.Element("mux", {
                        "name": "{}_{}".format(ic_name, out),
                        "input": inp,
                        "output": out
                    })

                    append_delays(conn, inputs, out, delays)

                    xml_ic.append(conn)
                    xml_conns.append(conn)

            # Remove the complete inrectonnect
            xml_ic.remove(xml_conn)

        # Do not split
        else:
            xml_conns = [xml_conn]

        # Process interconnect elements after the split
        for xml_conn in xml_conns:
            name = xml_conn.attrib["name"]

            # Do not annotate directs
            if xml_conn.tag == "direct":
                continue

            print(" Annotating <{}> interconnect '{}'".format(xml_conn.tag, name))
            annotate_interconnect(xml_conn, ic_path, circuit_models, physical_annotations, suffix)


def process_pb_types(xml_arch, circuit_models, physical_entities, physical_annotations, config_protocol):
    """
    Recurisvely processes each root pb_type (complex block) in the architecture.
    For each root pb_type walks through its children and injects FASM prefixes
    and features.
    """

    # Determine the FASM suffix dependent on the configuration protocol
    suffix = config_protocol.circuit_model

    print("Annotating pb_types ...")

    # Get the complexblocklist section
    xml_cplxblocklist = xml_arch.find("complexblocklist")
    assert xml_cplxblocklist is not None

    # Recursive walk function
    def walk(xml_pbtype, path=None):

        # Initialize hierarchical path
        if path is None:
            path = ""

        # Append self name
        path += "{}".format(xml_pbtype.attrib["name"])

        # Identify all modes and store their XML roots
        xml_modes = {m.attrib["name"]: m for m in xml_pbtype.findall("mode")}
        # No explicit modes, insert the default mode
        if not xml_modes:
            xml_modes = {"default": xml_pbtype}

        # Process each mode or a pb_type if the mode is implicit
        for mode, xml_mode in xml_modes.items():

            # Append the mode
            if mode != "default":
                curr_path = path + "[{}]".format(mode)
            else:
                curr_path = path

            # Process / annotate interconnects
            xml_ic = xml_mode.find("interconnect")
            if xml_ic is not None:
                process_interconnect(xml_ic, curr_path, physical_entities, circuit_models, physical_annotations, suffix)

            # Annotate child pb_types and/or recurse
            for xml_pb in xml_mode.findall("pb_type"):

                child_path = curr_path + ".{}".format(xml_pb.attrib["name"])

                # Add hierarchical name as FASM prefix
                if child_path in physical_entities:
                    prefix = build_hierarchical_name(xml_pb)
                    num_pb = int(xml_pb.get("num_pb", "1"))

                    final_prefix = []
                    for i in range(num_pb):
                        final_prefix.append("{}{}".format(prefix, i))
                    final_prefix = " ".join(final_prefix)

                    append_metadata(xml_pb, "fasm_prefix", final_prefix)
                    print(" {}: fasm_prefix=\"{}\"".format(child_path, final_prefix))

                # This is a leaf pb_type
                if is_leaf_pbtype(xml_pb):
                    annotate_leaf_pbtype(xml_pb, child_path, circuit_models, physical_annotations, suffix)

                # Recurse
                walk(xml_pb, curr_path + ".")

    # Recurse for each CLB
    for xml_pbtype in xml_cplxblocklist.findall("pb_type"):
        walk(xml_pbtype)


# =============================================================================


def inject_metadata(xml_arch, xml_openfpga_arch):
    """
    Injects metadata to complex blocks. FASM feature names are generated to be
    conformant to OpenFPGA bitstream features naming convention.
    """

    # Get circuit models
    circuit_models = load_circuit_models(xml_openfpga_arch)
    # Get physical annotations
    physical_annotations = load_physical_annotations(xml_openfpga_arch)
    # Get physical modes
    physical_modes = load_physical_modes(xml_openfpga_arch)
    # Get configuration protocol
    config_protocol = load_config_protocol(xml_openfpga_arch)

    # Identify physical entities
    physical_entities = identify_physical_entities(xml_arch, physical_modes)

    # Annotate physical modes
    process_pb_types(xml_arch, circuit_models, physical_entities, physical_annotations, config_protocol)

    return xml_arch
