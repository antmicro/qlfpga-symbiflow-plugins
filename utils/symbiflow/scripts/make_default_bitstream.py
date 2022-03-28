#!/usr/bin/env python3
"""
An utility script for generating default bitstream for QLF-based architectures.

The script uses qlf_fasm as Python package to first generate a FASM file
representing the default bitstream and then assembling it to its binary
form. It also updated the 'device.json' file in the FASM database so that
it contains a section informing of the default bitstream presence.
"""
import argparse
import sys
import os
import json
import logging

import qlf_fasm.make_default_fasm
import qlf_fasm.qlf_fasm

# =============================================================================


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--db-root",
        type=str,
        required=True,
        help="FASM database root"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Log level (def. \"WARNING\")"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, args.log_level.upper()),
    )

    # Prepare a set of FASM features that need to be set in the bitstream
    # Build a set of IO mode features for one tile. Each tile contains 16 IOs
    # FIXME: The feature name and IO count per IO tile is hard-coded.
    features = []
    for i in range(16):
        features.append("logical_tile_io_mode_io__{}.logical_tile_io_mode_physical__iopad_0.logical_tile_io_mode_physical__iopad_mode_default__pad_0.IO_QL_CCFF_mem.mem_out".format(i))

    # Run a script from the qlf_fasm package to generate default FASM file
    logging.info("Generating FASM for the default bitstream...")
    argv = [
        sys.argv[0],
        "--features", ",".join(features),
        "--db-root", args.db_root,
        "-o", os.path.join(args.db_root, "default_bitstream.fasm"),
        "--log-level", args.log_level,
    ]

    sys.argv = argv
    qlf_fasm.make_default_fasm.main()

    # Run qlf_fasm to generate the final default bitstream
    logging.info("Assembling default bitstream...")
    argv = [
        sys.argv[0],
        "-a",
        os.path.join(args.db_root, "default_bitstream.fasm"),
        os.path.join(args.db_root, "default_bitstream.hex"),
        "--no-default-bitstream",
        "-f", "4byte",
        "--db-root", args.db_root,
        "--log-level", args.log_level,
    ]
    sys.argv = argv
    qlf_fasm.qlf_fasm.main()

    # Append information to "device.json" that a default bitstream is present
    logging.info("Updating 'device.json'...")
    device_json = os.path.join(args.db_root, "device.json")
    with open(device_json, "r") as fp:
        json_root = json.load(fp)

    json_root["default_bitstream"] = {
        "file": "default_bitstream.hex",
        "format": "4byte",
    }

    fname = os.path.join(device_json)
    with open(fname, "w") as fp:
        json.dump(json_root, fp, indent=2, sort_keys=True)

    logging.info("Done.")


if __name__ == "__main__":
    main()
