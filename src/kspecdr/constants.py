"""
Instrument constants and global definitions.
"""

INST_GENERIC = 0
INST_2DF = 1
INST_6DF = 2
INST_AAOMEGA_2DF = 3
INST_HERMES = 4
INST_AAOMEGA_SAMI = 5
INST_TAIPAN = 6
INST_AAOMEGA_KOALA = 7
INST_AAOMEGA_IFU = 8
INST_SPECTOR_HECTOR = 9
INST_AAOMEGA_HECTOR = 10
INST_ISOPLANE = 99

# Maximum number of fibres
MAX__NFIBRES = 1000

# Fiber type codes
FIBER_TYPE_PROGRAM = "P"       # Science target (star or galaxy)
FIBER_TYPE_SKY = "S"           # Sky fiber
FIBER_TYPE_FIDUCIAL = "F"      # Fiducial (guide) fiber
FIBER_TYPE_CALIBRATION = "C"   # Spectrophotometric standard star
FIBER_TYPE_NONE = "N"          # Unallocated / unused
FIBER_TYPE_UNKNOWN = "U"       # Unknown
