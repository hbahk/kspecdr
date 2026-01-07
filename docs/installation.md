# Installation

`kspecdr` is a Python-based reduction pipeline for the K-SPEC instrument.

## Prerequisites

*   Python 3.8+
*   `numpy`
*   `scipy`
*   `astropy`
*   `matplotlib` (optional, for visualization)
*   `pywt` (optional, for wavelet-based peak finding)
*   `scikit-learn` (optional, for some calibration routines)

## Installation from Source

1.  Clone the repository:

    ```bash
    git clone https://github.com/your-org/kspecdr.git
    cd kspecdr
    ```

2.  Install the package:

    ```bash
    pip install -e .
    ```

    We recommend using `-e` (editable mode) if you are developing or testing the latest changes.

3.  (Optional) Install documentation dependencies:

    If you wish to build the documentation locally:
    ```bash
    pip install -r docs/requirements.txt
    ```

```{note}
The project is currently under active development. Some dependencies might change. Please check `requirements.txt` for the latest list.
```
