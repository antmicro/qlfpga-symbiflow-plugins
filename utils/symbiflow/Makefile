BUILD_PATH ?= ./build
PYTHON ?= python3

# The path to an OpenFPGA tash must be provided
ifeq ($(OPENFPGA_TASK_PATH),)
$(error ERROR: The variable OPENFPGA_TASK_PATH has to be specified)
endif

# Device process
PROCESS_PREFIX ?= UMC22nm_

# Input OpenFPGA files
OPENFPGA_VPR_ARCH = $(abspath $(OPENFPGA_TASK_PATH)/latest/$(PROCESS_PREFIX)vpr/top/MIN_ROUTE_CHAN_WIDTH/arch/$(PROCESS_PREFIX)vpr.xml)
OPENFPGA_ARCH = $(abspath $(OPENFPGA_TASK_PATH)/arch/$(PROCESS_PREFIX)openfpga.xml)
OPENFPGA_RR_GRAPH = $(abspath $(OPENFPGA_TASK_PATH)/../rr_data/rr_graph.openfpga.xml.obj)
OPENFPGA_GSB_PATH = $(abspath $(OPENFPGA_TASK_PATH)/../rr_data/gsb)
OPENFPGA_FABRIC_BITSTREAM = $(abspath $(shell find $(OPENFPGA_TASK_PATH)/../ -name fabric_bitstream.xml))

# Device layout to use
LAYOUT ?= 32x32

# Corner delay model to use. E.g. slow, fast, typical, etc.
CORNER ?= ""

# Full name of the target SymbiFlow device
DEVICE ?= qlf_k4n8-qlf_k4n8_umc22

# Output rr graph format (xml or bin)
GRAPH_EXT ?= bin

# FASM prefix template for VPR tiles (do not change)
FASM_PREFIX_TEMPLATE ?= \
    "fpga_top.grid_{tile}{side}_{x}__{y}_.logical_tile_{site[0]}_mode_{site[0]}__{i}"

# Cap'n'proto rr graph schema URL. The GIT SHA is fixed to the same as the VPR
# version used currently in SymbiFlow
CAPNP_SCHEMA_URL ?= https://raw.githubusercontent.com/verilog-to-routing/vtr-verilog-to-routing/b3b34e77aa1bad7371bfe05087e01e04e2495179/libs/libvtrcapnproto/gen/rr_graph_uxsdcxx.capnp

# =============================================================================

$(BUILD_PATH):
	mkdir -p $(BUILD_PATH)/tmp

# Downloads VPR Cap'n'proto schema for binary rr graph storage
$(BUILD_PATH)/tmp/rr_graph.capnp: | $(BUILD_PATH)
	wget $(CAPNP_SCHEMA_URL) -O $@

# Patches the routing graph so that it can be used in SymbiFlow
$(BUILD_PATH)/rr_graph.final.$(GRAPH_EXT): $(OPENFPGA_RR_GRAPH) $(OPENFPGA_GSB_PATH) $(BUILD_PATH)/tmp/rr_graph.capnp | $(BUILD_PATH)
	$(PYTHON) scripts/fixup_rr_graph.py \
		--rr-graph-in $< \
		--rr-graph-out $@ \
		--capnp-schema $(BUILD_PATH)/tmp/rr_graph.capnp \
		--openfpga-arch-in $(OPENFPGA_ARCH) \
		--gsb $(OPENFPGA_GSB_PATH)

# Compresses the binary rr graph using gzip to reduce its storage size
$(BUILD_PATH)/rr_graph.final.$(GRAPH_EXT).gz: $(BUILD_PATH)/rr_graph.final.$(GRAPH_EXT) | $(BUILD_PATH)
	cp $< $(subst .gz,.tmp,$@)
	gzip -f -9 -q $(subst .gz,.tmp,$@)
	mv $(subst .gz,.tmp.gz,$@) $@

# Processes VPR architecture suitable for OpenFPGA and converts it so that it
# is suitable for VPR used in SymbiFlow.
$(BUILD_PATH)/tmp/arch.fixup.xml: $(OPENFPGA_VPR_ARCH) $(OPENFPGA_ARCH) | $(BUILD_PATH)
	$(PYTHON) scripts/fixup_arch.py \
		--arch-in $(OPENFPGA_VPR_ARCH) \
		--openfpga-arch-in $(OPENFPGA_ARCH) \
		--pick-layout $(LAYOUT)=$(DEVICE) \
		--strip-comments \
		--corner $(CORNER) \
		--arch-out $@

# Flattens the grid layout definition and injects FASM prefixes for tiles.
$(BUILD_PATH)/arch.final.xml: $(BUILD_PATH)/tmp/arch.fixup.xml | $(BUILD_PATH)
	$(PYTHON) scripts/flatten_layout.py \
		--arch-in $< \
		--arch-out $@ \
		--fasm_prefix $(FASM_PREFIX_TEMPLATE) \

# Extract data needed for the repacker in SymbiFlow and write it to a file
# that can be redistributed
$(BUILD_PATH)/repacking_rules.json: $(OPENFPGA_ARCH) | $(BUILD_PATH)
	$(PYTHON) scripts/extract_repacking_rules.py \
		--openfpga-arch $(OPENFPGA_ARCH) \
		-o $@

# Build FASM database
$(BUILD_PATH)/fasm_database: $(OPENFPGA_FABRIC_BITSTREAM) | $(BUILD_PATH)
	$(PYTHON) scripts/fasm_database_builder.py \
		--fabric-bitstream $< \
		-o $@ \
		--log-level DEBUG || (rm -rf $@ && false)
ifndef NO_DEFAULT_BITSTREAM
	$(PYTHON) scripts/make_default_bitstream.py \
		--db-root $@ \
		--log-level DEBUG || (rm -rf $@ && false)
endif

all: $(BUILD_PATH)/arch.final.xml $(BUILD_PATH)/rr_graph.final.$(GRAPH_EXT).gz $(BUILD_PATH)/repacking_rules.json $(BUILD_PATH)/fasm_database

clean:
	rm -rf build

.PHONY: all clean
