# `kspecdr`
Reduction pipeline for KSPEC instrument based on 2drdr software.

# Progress Tracker

| Category | Function/Component | Related 2dFDR Command(s) | Priority | Completeness (%) | Notes |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **I. Pipeline Structure** | Overall Pipeline Flow/Wrapper Implementation | `reduce_run` (potentially) | Final | 0% | Defining the main script/class structure. |
| **II. Frame Organization & Integrity** | Check Frame Integrity | `get class`, `get kywd`, `get size`, `compare` | Low | 0% | Implement integrity checks and file classification. |
|  | Check Run Integrity | `list` | Low | 0% | Ensure all required calibration frames are present for a run. |
|  | Ordered List/Table Generation | `list` | Low | 0% | Generate a structured input list for reduction steps. |
|  | Quick-look Functions | N/A (External plotting) | Middle | 0% | Quick visualization of raw frames. |
| **III. Preprocessing CCD Frames** | General Preprocessing | `make_im` | URGENT | 90% | Make intermediate preprocessed images for further reductions. |
|  | Reduce Bias Frames | `reduce_bias`, `combine_image` | Done | 100% | Implementation of bias reduction/combination. |
|  | Reduce Dark Frames | `reduce_dark`, `combine_image` | Done | 100% | Implementation of dark reduction/combination. |
|  | Reduce Detector Flat Frames (LFLAT) | `reduce_lflat`, `combine_image` | High | 0% | Implementation of LFLAT reduction/combination. |
| **IV. Extraction** | Generate Tramline Map Frames | `make_tlm` | URGENT | 90% | Wrapping tramline map generation. |
|  | Extract Spectra | `make_ex` | High | 0% | Extract spectra from the predefined tramline map |
| **V. Wavelength Calibration** | Reduce Arc Frames and Generate Calibration Data (from Arc) | `reduce_arc` | High | 0% | Implementation of arc frame reduction. |
|  | Interactive Wavelength Calibration | N/A (Requires GUI/interactive mode) | Low | 0% | Define requirements for interactive mode, if needed. |
| **VI. Final Reduction** | Reduce Fiber Flat (FFLAT) Frames | `reduce_fflat` | High | 0% | Implementation of FFLAT reduction. |
|  | Flat-fielding, Scrunching (RED frame) | `make_red` | Middle | 0% | Final reduction step before combination/flux calibration. |
|  | Throughput Correction | `correct_throughput` | Middle | 0% |  |
|  | Sky Subtraction | `subtract_sky` | Middle | 0% |  |
|  | Combine Spectra | `combine_spectra` | Middle | 0% | Combining multiple reduced science frames. |
|  | Splice (if applicable) | `splice` | Middle | 0% | For two-channel data (e.g., Red/Blue). |
|  | Transfer Function Correction | `transfunc` | High | 0% |  |
|  | Reduce Sky Frames | `reduce_sky` | High | 0% | Implementation of sky frame reduction. |
|  | Reduce Science/Object Frames | `reduce_object` | High | 0% | Implementation of science frame reduction. |
|  | Reduce Standard Star Frames (Flux) | `reduce_fflux` | High | 0% | Implementation of standard star/flux reduction. |
| **VII. Helper Functions** | Print Available Commands/Steps | `help`, `examples` | Low | 0% | Wrapper command listing canonical steps. |
|  | Logger Implementation | N/A (Standard Python logging) | Low | 0% | Setting up robust logging. |
| **VIII. Configuration/Error** | Keyword Argument Passing | N/A | Low | 0% | Strategy for passing config (e.g., idxFile). |
|  | Error Handling (I/O, Order, Arguments, etc.) | N/A | Low | 0% | Implement internal checks for reduction logic. |
| **IX. Documentation** | Docstrings/Sphinx Docs | N/A | Low | 0% | Generating documentation and examples. |
