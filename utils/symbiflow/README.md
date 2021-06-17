# Utilities for SymbiFlow

These are utilities required to prepare VPR arch.xml and rr_graph, repacker
rules and FASM database to be used with the SymbiFlow toolchain.

## Installation

1. Install Python3 prerequisities
```
pip install -r requirements.txt
```

## Running

Run `make all` to build all the data needed by SymbiFlow.

The files to be used with SymbiFlow are:
 - `arch.final.xml`
 - `rr_graph.final.bin.gz`
 - `repacking_rules.json`
 - `fasm_database`
