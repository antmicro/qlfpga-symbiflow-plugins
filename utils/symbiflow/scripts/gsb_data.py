#!/usr/bin/env python3
"""
A set of utilities for handling rr graph GSB data.
"""
import os
from collections import namedtuple

import lxml.etree as ET

# =============================================================================


class GsbEntry:
    """
    Stores a single GSB entry which corresponds to a single N:1 mux
    """

    Node = namedtuple("Node", "id type side segment_id grid_side index")

    def __init__(self, xml_entry):
        """
        Builds a GsbEntry object from an XML ElementTree object
        """

        # Check type
        assert xml_entry.tag in ["CHANX", "CHANY", "IPIN"], xml_entry.tag

        # Sink node info
        segment_id = xml_entry.get("segment_id", None)
        if segment_id is not None:
            segment_id = int(segment_id)

        self.node = GsbEntry.Node(
            id = int(xml_entry.attrib["node_id"]),
            type = xml_entry.tag,
            side = xml_entry.attrib["side"],
            segment_id = segment_id,
            grid_side = None,
            index = int(xml_entry.attrib["index"]),
        )
        GsbEntry.check_node(self.node)

        self.mux_size = int(xml_entry.attrib["mux_size"])

        # Driver nodes
        self.drivers = []
        for xml_driver in xml_entry.findall("driver_node"):

            segment_id = xml_driver.get("segment_id", None)
            if segment_id is not None:
                segment_id = int(segment_id)

            node = GsbEntry.Node(
                id = int(xml_driver.attrib["node_id"]),
                type = xml_driver.attrib["type"],
                side = xml_driver.attrib["side"],
                segment_id = segment_id,
                grid_side = xml_driver.get("grid_side", None),
                index = int(xml_driver.attrib["index"]),
            )

            GsbEntry.check_node(node)
            self.drivers.append(node)

    @staticmethod
    def check_node(node):
        """
        Throws an assertion if some node data is incorrect
        """
        assert node.type in ["CHANX", "CHANY", "IPIN", "OPIN"], node
        assert node.side in ["left", "top", "right", "bottom"], node

    def dump(self):
        """
        Dumps GSB data.
        """
        print("GSB: {}".format(self.node))
        for i, driver in enumerate(self.drivers):
            print(" {:2d} {}".format(i, driver))

# =============================================================================


def load_gsb_data(path, pbar=lambda x: x):
    """
    Loads GSB data for a routing graph stored in files under the given path
    """

    gsb_data = {}

    # Loop over all XML files found and read them
    for fname in pbar(os.listdir(path)):

        # Check if this looks like an XML file
        _, ext = os.path.splitext(fname)
        if ext.lower() != ".xml":
            continue

        # Must be a file
        fname = os.path.join(path, fname)
        if not os.path.isfile(fname):
            continue

        # Read and parse the XML
        xml_tree = ET.parse(fname, ET.XMLParser(remove_blank_text=True))
        xml_root = xml_tree.getroot()

        # Check if this is a GSB
        if xml_root.tag != "rr_gsb":
            continue

        # Read and parse GSB entries
        gsbs = []
        for xml_element in xml_root:
            gsbs.append(GsbEntry(xml_element))

        # Store them
        loc = (
            int(xml_root.attrib["x"]),
            int(xml_root.attrib["y"]),
        )

        gsb_data[loc] = gsbs

    return gsb_data
