#!/usr/bin/env python3
"""
This script is responsible for extracting all the data required by repacker
from OpenFPGA architecture file so that it can be redistributed.
"""

import argparse
import json

import lxml.etree as ET

# =============================================================================


def load_repacking_rules(xml_root):
    """
    Gets repacking rules from the OpenFPGA arch
    """

    def handle_phys_io_rule(src_pbtype, dst_pbtype):
        """
        Handles the destination pb type to take into account the new
        physical IO structure in the VPR architecture.

        If the pb_type is an IO kind, the destination rule is changed
        accordingly.
        """
        io_type = src_pbtype.split(".")[-1]

        if io_type in ["inpad", "outpad"]:
            return "{orig}[{pad}].{pad}".format(orig=dst_pbtype, pad=io_type)
        else:
            return dst_pbtype

    def parse_index(s):
        """
        Tries to parse a string as integer and if it fails retries as float
        """
        try:
            return int(s)
        except ValueError:
            return float(s)

    # Get "pb_type_annotation"
    xml_rules = xml_root.find("pb_type_annotations")
    assert xml_rules is not None

    # Examine all annotations
    rules = []

    for xml_rule in xml_rules.findall("pb_type"):
        if "name" in xml_rule.attrib and \
           "physical_pb_type_name" in xml_rule.attrib:

            # Index map - linear function (ax + b) where index_map = (a, b)
            index_map = (
                parse_index(xml_rule.get("physical_pb_type_index_factor", "1")),
                parse_index(xml_rule.get("physical_pb_type_index_offset", "0")),
            )

            # Mode bits
            mode_bits = xml_rule.get("mode_bits", None)

            # Get port map
            port_map = {}
            for xml_port in xml_rule.findall("port"):
                src = xml_port.attrib["name"]
                dst = xml_port.attrib["physical_mode_port"]
                port_map[src] = dst

            src_pbtype = xml_rule.attrib["name"]
            dst_pbtype = xml_rule.attrib["physical_pb_type_name"]

            dst_pbtype = handle_phys_io_rule(src_pbtype, dst_pbtype)

            # Create the rule
            rule = {
                "src_pbtype": src_pbtype,
                "dst_pbtype": dst_pbtype,
                "index_map": index_map,
                "mode_bits": mode_bits,
                "port_map": port_map,
            }
            rules.append(rule)

    return rules


# =============================================================================


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "-o",
        type=str,
        default="repacking_rules.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--openfpga-arch",
        type=str,
        required=True,
        help="OpenFPGA arch.xml input"
    )

    args = parser.parse_args()

    # Read and parse the OpenFPGA arch XML file
    openfpga_xml_tree = ET.parse(args.openfpga_arch, ET.XMLParser(remove_blank_text=True))
    xml_openfpga_arch = openfpga_xml_tree.getroot()
    assert xml_openfpga_arch is not None and \
        xml_openfpga_arch.tag == "openfpga_architecture"

    # Get the repacking rules
    repacking_rules = load_repacking_rules(xml_openfpga_arch)

    # Write JSON file
    json_root = {
        "repacking_rules": repacking_rules
    }

    with open(args.o, "w") as fp:
        json.dump(json_root, fp, indent=2)

# =============================================================================

if __name__ == "__main__":
    main()

