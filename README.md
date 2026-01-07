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
| **III. Preprocessing CCD Frames** | General Preprocessing | `make_im` | URGENT | 90% | Make intermediate preprocessed images. |
|  | Reduce Bias Frames | `reduce_bias`, `combine_image` | Done | 100% | Full implementation with sigma-clipped combination. |
|  | Reduce Dark Frames | `reduce_dark`, `combine_image` | Done | 100% | Full implementation with exposure time grouping. |
|  | Reduce Detector Flat Frames (LFLAT) | `reduce_lflat`, `combine_image` | High | 100% | Implementation in `preproc.py`. |
| **IV. Extraction** | Generate Tramline Map Frames | `make_tlm` | URGENT | 90% | Implemented using wavelet/peak-finding algorithms. |
|  | Extract Spectra | `make_ex` | High | 90% | `SUM` extraction implemented. `OPTEX`/`GAUSS` are placeholders. |
| **V. Wavelength Calibration** | Reduce Arc Frames and Generate Calibration Data (from Arc) | `reduce_arc` | High | 90% | Logic implemented in `reduce_arc.py` calling calibration routines. |
|  | Interactive Wavelength Calibration | N/A (Requires GUI/interactive mode) | Low | 0% | Define requirements for interactive mode, if needed. |
| **VI. Final Reduction** | Reduce Fiber Flat (FFLAT) Frames | `reduce_fflat` | High | 50% | Image combination implemented; full reduction (extraction) pending integration. |
|  | Flat-fielding, Scrunching (RED frame) | `make_red` | Middle | 50% | Basic flow exists; flat-fielding and scrunching logic are placeholders. |
|  | Throughput Correction | `correct_throughput` | Middle | 0% |  |
|  | Sky Subtraction | `subtract_sky` | Middle | 0% |  |
|  | Combine Spectra | `combine_spectra` | Middle | 20% | `combine_image` exists, but specific spectral combination logic (RED files) is missing. |
|  | Splice (if applicable) | `splice` | Middle | 0% | For two-channel data (e.g., Red/Blue). |
|  | Transfer Function Correction | `transfunc` | High | 0% |  |
|  | Reduce Sky Frames | `reduce_sky` | High | 0% | Implementation of sky frame reduction. |
|  | Reduce Science/Object Frames | `reduce_object` | High | 0% | Implementation of science frame reduction. |
|  | Reduce Standard Star Frames (Flux) | `reduce_fflux` | High | 0% | Implementation of standard star/flux reduction. |
| **VII. Helper Functions** | Print Available Commands/Steps | `help`, `examples` | Low | 0% | Wrapper command listing canonical steps. |
|  | Logger Implementation | N/A (Standard Python logging) | Low | 80% | Basic logging configured in modules. |
| **VIII. Configuration/Error** | Keyword Argument Passing | N/A | Low | 50% | `args` dictionary passed around; needs more formal structure. |
|  | Error Handling (I/O, Order, Arguments, etc.) | N/A | Low | 30% | Basic checks in place; needs comprehensive error management. |
| **IX. Documentation** | Docstrings/Sphinx Docs | N/A | Low | 20% | Docstrings added to implemented modules. Sphinx setup pending. |
