#!/usr/bin/env python3
"""
This script is intended for processing an arch.xml file used in the OpenFPGA
project so that it can be used with the VPR used in SymbiFlow.
"""

import argparse
import re
import itertools
import sys

import lxml.etree as ET

import arch_fasm_injector

# =============================================================================


def fixup_tiles(xml_arch):
    """
    This function convert non-heterogeneous tiles into heterogeneous with only
    one sub-tile. This is required to match with the syntax supported by the
    VPR version used in SymbiFlow.
    """

    # Legal attributes for a tile tag
    TILE_ATTRIB = ["name", "width", "height", "area"]
    # Legal attributes for a sub-tile tag
    SUB_TILE_ATTRIB = ["name", "capacity"]

    # Get the tiles section
    xml_tiles = xml_arch.find("tiles")
    assert xml_tiles is not None

    # List all tiles
    elements = xml_tiles.findall("tile")
    for xml_org_tile in elements:

        # Check if this one is heterogeneous, skip if so.
        if xml_org_tile.find("sub_tile"):
            continue

        # Detach the tile node
        xml_tiles.remove(xml_org_tile)

        # Make a new sub-tile node. Copy legal attributes and children
        xml_sub_tile = ET.Element("sub_tile",
            {k: v for k, v in xml_org_tile.attrib.items() if k in SUB_TILE_ATTRIB}
        )
        for element in xml_org_tile:
            xml_sub_tile.append(element)

        # Make a new tile node, copy legal attributes
        xml_new_tile = ET.Element("tile",
            {k: v for k, v in xml_org_tile.attrib.items() if k in TILE_ATTRIB}
        )

        # Attach nodes
        xml_new_tile.append(xml_sub_tile)
        xml_tiles.append(xml_new_tile)

    return xml_arch


def fixup_attributes(xml_arch):
    """
    Removes OpenFPGA-specific attributes from the arch.xml that are not
    accepted by the SymbiFlow VPR version.
    """

    # Remove all attributes from the layout section
    xml_old_layout = xml_arch.find("layout")
    assert xml_old_layout is not None

    xml_new_layout = ET.Element("layout")
    for element in xml_old_layout:
        xml_new_layout.append(element)

    xml_arch.remove(xml_old_layout)
    xml_arch.append(xml_new_layout)

    # Remove not supported attributes from "device/switch_block" elements
    xml_device = xml_arch.find("device")
    if xml_device is not None:

        # Legal switch_block tags
        SWITCH_BLOCK_TAGS = ["type", "fs"]

        # Remove illegal tags
        xml_switch_blocks = xml_device.findall("switch_block")
        for xml_old_switch_block in xml_switch_blocks:

            attrib = {k: v for k, v in xml_old_switch_block.attrib.items() \
                      if k in SWITCH_BLOCK_TAGS}

            xml_new_switch_block = ET.Element("switch_block", attrib)
            for element in xml_old_switch_block:
                xml_new_switch_block.append(element)

            xml_device.remove(xml_old_switch_block)
            xml_device.append(xml_new_switch_block)

    return xml_arch


def fixup_models(xml_arch):
    """
    Set "never_prune" to "true" for all models in the architecture.

    FIXME: The attribute should actually be set only for IO models used by
    physical modes. For that there will be a need to read and parse OpenFPGA
    arch XML to identify which modes are physical. For now let's assume that
    all models are non-prunable.
    """

    # Find the models section
    xml_models = xml_arch.find("models")
    assert xml_models is not None

    # Process them
    for xml_model in xml_models.findall("model"):
        if xml_model.get("never_prune", None) != "true":

            xml_model_new = ET.Element("model", {
                "name": xml_model.get("name"),
                "never_prune": "true"
            })

            for element in xml_model:
                xml_model_new.append(element)

            xml_models.remove(xml_model)
            xml_models.append(xml_model_new)

    return xml_arch


def fixup_phys_io(xml_arch):
    """
    Adds input and output blif to the physical IOs.

    Being a special handling case, this step needs to happen after the XML has
    been generated, and as a final step.

    It re-writes the physical IO pb_type to allow having top level input ports
    in the design using the generated architecture in symbiflow.

    It is assumed that the input direction will have the bit set to 1.
    """

    def add_io_mode(xml_parent, dir, fasm_feature=None):
        if dir == "out":
            other_dir = "in"
        else:
            assert dir == "in"
            other_dir = "out"

        pad = "{}pad".format(dir)
        blif = ".{}put".format(dir)
        other_pin = "{}put".format(other_dir)
        pin = "{}put".format(dir)

        xml_mode = ET.SubElement(xml_parent, "mode", {
            "name": pad
        })

        xml_pb_type = ET.SubElement(xml_mode, "pb_type", {
            "name": pad,
            "blif_model": blif,
            "num_pb": "1"
        })

        ET.SubElement(xml_pb_type, other_pin, {
            "name": pad,
            "num_pins": "1"
        })

        xml_ic = ET.SubElement(xml_mode, "interconnect")
        ET.SubElement(xml_ic, "direct", {
            "name": "{pad}_blif_to_{pad}_pad".format(pad=pad),
            pin: "{pad}.{pad}".format(pad=pad),
            other_pin: "pad.{pad}".format(pad=pad),
        })

        if fasm_feature is not None:
            xml_metadata = ET.SubElement(xml_mode, "metadata")
            xml_fasm_feature = ET.SubElement(xml_metadata, "meta", {
                "name": "fasm_features"
            })

            xml_fasm_feature.text = fasm_feature

        return xml_mode


    # Remove IO model
    xml_models = xml_arch.find("models")
    assert xml_models is not None

    for xml_model in xml_models.findall("model"):
        if xml_model.get("name", None) == "io":
            xml_models.remove(xml_model)


    xml_blocks = xml_arch.find("complexblocklist")
    for xml_pb_type in xml_blocks.xpath("//pb_type"):
        name = xml_pb_type.get("name", None)
        blif_model = xml_pb_type.get("blif_model", None)

        if name != "pad" or blif_model != ".subckt io":
            continue

        del xml_pb_type.attrib["blif_model"]

        fasm_feature = None

        xml_metadata = xml_pb_type.find("metadata")
        for meta in xml_metadata.findall("meta"):
            if meta.get("name", None) == "fasm_params":
                fasm_feature = meta.text.split("=")[0].strip()
                xml_metadata.remove(meta)

        assert fasm_feature is not None

        parts = fasm_feature.split(".")
        parts[-1] = "NOT_" + parts[-1]
        inv_fasm_feature = ".".join(parts)

        add_io_mode(xml_pb_type, "in", fasm_feature)
        add_io_mode(xml_pb_type, "out", inv_fasm_feature)

    return xml_arch


def make_all_modes_packable(xml_arch):
    """
    Strips all attributes that disable packing for a mode
    """

    # Find the complexblocklist section
    xml_complexblocklist = xml_arch.find("complexblocklist")
    assert xml_complexblocklist is not None

    # Find all modes
    xml_modes = xml_complexblocklist.xpath("//mode")

    # Enable packing (by stripping all attributes except "name")
    for xml_mode in xml_modes:

        if xml_mode.get("disabled_in_pack", None) == "true" or \
           xml_mode.get("disable_packing", None) == "true" or \
           xml_mode.get("packable", None) == "false":

            xml_mode_new = ET.Element("mode", {"name": xml_mode.get("name")})

            for element in xml_mode:
                xml_mode_new.append(element)

            xml_parent = xml_mode.getparent()

            xml_parent.remove(xml_mode)
            xml_parent.append(xml_mode_new)

    return xml_arch

# =============================================================================


def pick_layout(xml_arch, layout_spec, corner=None):
    """
    This function processes the <layout> section. It allows to pick the
    specified fixed layout and optionally change its name. This is required
    for SymbiFlow.
    """

    # Get layout names
    if "=" in layout_spec:
        layout_name, new_name = layout_spec.split("=", maxsplit=1)
    else:
        layout_name = layout_spec
        new_name = layout_spec

    if corner:
        new_name = "{}_{}".format(new_name, corner)

    # Get the layout section
    xml_layout = xml_arch.find("layout")
    assert xml_layout is not None

    # Find the specified layout name
    found = False
    for element in list(xml_layout):

        # This one is fixed and name matches
        if element.tag == "fixed_layout" and \
           element.attrib["name"] == layout_name:

            # Copy the layout with a new name
            attrib = dict(element.attrib)
            attrib["name"] = new_name

            new_element = ET.Element("fixed_layout", attrib)
            for sub_element in element:
                new_element.append(sub_element)

            xml_layout.append(new_element)
            found = True

        # Remove the element
        xml_layout.remove(element)

    # Not found
    if not found:
        print("ERROR: Fixed layout '{}' not found".format(layout_name))
        exit(-1)

    return xml_arch

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
        "--arch-out",
        type=str,
        default=None,
        help="VPR arch.xml output"
    )
    parser.add_argument(
        "--pick-layout",
        type=str,
        default=None,
        help="Pick the given layout name. Optionally re-name it (<old_name>=<new_name>)"
    )
    parser.add_argument(
        "--corner",
        type=str,
        default=None,
        help="Pick the given corner."
    )
    parser.add_argument(
        "--strip-comments",
        action="store_true",
        help="Strips all comments from the VPR arch.xml"
    )
    parser.add_argument(
        "--openfpga-arch-in",
        type=str,
        required=True,
        help="OpenFPGA arch.xml input"
    )
    parser.add_argument(
        "--enable-all-modes",
        action="store_true",
        help="Enable VPR packing for all physical modes"
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

    # Fixup models
    fixup_models(xml_arch)

    # Fixup tiles.
    fixup_tiles(xml_arch)

    # Fixup non-packable modes
    if args.enable_all_modes:
        make_all_modes_packable(xml_arch)

    # Fixup OpenFPGA specific attributes
    fixup_attributes(xml_arch)

    # Pick layout
    if args.pick_layout:
        pick_layout(xml_arch, args.pick_layout, args.corner)

    # Inject FASM annotation
    xml_arch = arch_fasm_injector.inject_fasm_annotation(
        xml_arch,
        xml_openfpga_arch
    )

    # Fixup Physical IOs
    fixup_phys_io(xml_arch)

    # Write the modified architecture file back
    xml_tree = ET.ElementTree(xml_arch)
    xml_tree.write(
        args.arch_out,
        pretty_print=True,
        encoding="utf-8"
    )

# =============================================================================

if __name__ == "__main__":
    main()
