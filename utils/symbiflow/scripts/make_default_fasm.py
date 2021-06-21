#!/usr/bin/env python3
"""
"""

import argparse
import re
import itertools
import sys

import lxml.etree as ET

# =============================================================================


def load_physical_and_idle_modes(xml_root):
    """
    Gets physical modes and idle modes of pb_types
    """

    # Get "pb_type_annotation"
    xml_rules = xml_root.find("pb_type_annotations")
    assert xml_rules is not None

    # Examine all annotations
    physical_modes = {}
    idle_modes = {}

    for xml_rule in xml_rules.findall("pb_type"):
        if "name" in xml_rule.attrib:
            path = xml_rule.attrib["name"]

            if "physical_mode_name" in xml_rule.attrib:
                mode = xml_rule.attrib["physical_mode_name"]
                physical_modes[path] = mode

            if "idle_mode_name" in xml_rule.attrib:
                mode = xml_rule.attrib["idle_mode_name"]
                idle_modes[path] = mode

    # DEBUG
    print("Physical modes:")
    for k, v in physical_modes.items():
        print(" {}[{}]".format(k, v))

    print("Idle modes:")
    for k, v in idle_modes.items():
        print(" {}[{}]".format(k, v))

    return physical_modes, idle_modes

# =============================================================================


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--arch-in",
        type=str,
        required=True,
        help="VPR arch.xml input"
    )
    parser.add_argument(
        "--openfpga-arch-in",
        type=str,
        required=True,
        help="OpenFPGA arch.xml input"
    )
    parser.add_argument(
        "--layout",
        type=str,
        required=True,
        help="Device layout name"
    )

    args = parser.parse_args()

    # Read and parse the VPR arch XML file
    xml_tree = ET.parse(args.arch_in,
        ET.XMLParser(remove_blank_text=True, remove_comments=args.strip_comments)
    )
    xml_arch = xml_tree.getroot()
    assert xml_arch is not None and xml_arch.tag == "architecture"

    # Read and parse the OpenFPGA arch XML file
    xml_tree = ET.parse(args.openfpga_arch_in,
        ET.XMLParser(remove_blank_text=True, remove_comments=True)
    )
    xml_openfpga_arch = xml_tree.getroot()
    assert xml_openfpga_arch is not None and \
        xml_openfpga_arch.tag == "openfpga_architecture"


# =============================================================================

if __name__ == "__main__":
    main()
