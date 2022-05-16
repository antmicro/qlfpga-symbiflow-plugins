#!/usr/bin/env Python3
"""
Utilities for reading and representing data from OpenFPGA architecture XML
"""

# =============================================================================


class CircuitModel:
    """
    This class holds all relevant data of a circuit model necessary for
    FASM annotation of VPR arch XML.
    """
    def __init__(self, type, name):
        self.type = type
        self.name = name
        self.conf_bits = 0
        self.mode_bits = 0
        self.is_default = False

    def __str__(self):
        return "{} ({}): conf_bits={} mode_bits={} is_default={}".format(
            self.name,
            self.type,
            self.conf_bits,
            self.mode_bits,
            self.is_default,
        )

    def __repr__(self):
        return str(self)


class MuxCircuitModel(CircuitModel):
    """
    Subclass for representing mux circuit models. These mux model are generic
    (i.e. without the explicit width). They are specialized when found
    instanced.
    """

    def __init__(self, type, name, structure, num_levels=1, add_const_input=False):
        assert type == "mux"
        super().__init__(type, name)

        self.structure = structure
        self.num_levels = num_levels
        self.add_const_input = add_const_input

    def __str__(self):
        return "{} structure={} num_levels={} add_const_input={}".format(
            super().__str__(),
            self.structure,
            self.num_levels,
            self.add_const_input
        )

    def __repr__(self):
        return str(self)


class LutCircuitModel(CircuitModel):
    """
    Subclass for representing LUT subcircuits. Stores information about ports
    and fracturing.
    """

    def __init__(self, type, name, ports):
        assert type == "lut"
        super().__init__(type, name)

        self.ports = ports

    def __str__(self):
        string = super().__str__() + " "
        for port, attrib in self.ports.items():
            string += "port={}[{}:0] {} ".format(
                port,
                attrib["size"],
                {k:v for k, v in attrib.items() if k != "size"},
            )

        return string

    def __repr__(self):
        return str(self)

# =============================================================================


def load_circuit_models(xml_root):
    """
    Loads circuit models from OpenFPGA arch XML.
    """

    # Get "circuit_library"
    xml_circuit_lib = xml_root.find("circuit_library")
    assert xml_circuit_lib is not None

    # Collect circuit models
    circuit_models = {}
    default_circuit_models = {}

    for xml_circuit in xml_circuit_lib.findall("circuit_model"):

        # Get the "desgin_technology" tag
        xml_tech = xml_circuit.find("design_technology")
        assert xml_tech is not None

        # Add
        type = xml_circuit.attrib["type"]
        name = xml_circuit.attrib["name"]

        # We have a mux
        if type == "mux":

            tech = xml_tech.get("type", None)
            assert tech == "cmos", tech

            structure = xml_tech.get("structure", None)
            num_levels = xml_tech.get("num_level", None)
            add_const_input = xml_tech.get("add_const_input", "false")

            if num_levels is not None:
                num_levels = int(num_levels)

            add_const_input = (add_const_input == "true")

            circuit_models[name] = MuxCircuitModel(
                type,
                name,
                structure,
                num_levels,
                add_const_input
            )

        # We have a LUT
        elif type == "lut":

            PORT_ATTRIB = {"type", "size", "lut_frac_level", "lut_output_mask"}

            ports = {}
            for xml_tag in xml_circuit.findall("port"):

                if xml_tag.attrib["type"] not in ["input", "output"]:
                    continue

                key = xml_tag.attrib["prefix"]
                val = {k: v for k, v in xml_tag.attrib.items() if k in PORT_ATTRIB}
                ports[key] = val

            circuit_models[name] = LutCircuitModel(
                type,
                name,
                ports
            )

        # We have something else
        else:
            circuit_models[name] = CircuitModel(type, name)

        # Check if this one is configurable by the bitstream
        for xml_item in xml_circuit:

            # Port of type "sram"
            if xml_item.tag == "port" and xml_item.attrib["type"] == "sram":

                bits = int(xml_item.attrib["size"])

                if xml_item.get("mode_select", "false") == "true":
                    circuit_models[name].mode_bits += bits
                else:
                    circuit_models[name].conf_bits += bits

        # Add as a default
        if xml_circuit.get("is_default", "false") == "true":
            circuit_models[name].is_default = True

            # Check if there is only one default circuit of a given type
            assert type not in default_circuit_models, circuit_models[name]
            default_circuit_models[type] = name

    # DEBUG
    print("Circuit models (configurable):")
    keys = sorted(list(circuit_models.keys()))
    for k in keys:
        if circuit_models[k].mode_bits + circuit_models[k].conf_bits:
            print("", circuit_models[k])

    return circuit_models

