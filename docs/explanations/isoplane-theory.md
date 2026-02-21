# Isoplane Reduction Theory and Strategy (End-to-End)

This document explains what is considered at each reduction stage, why each step exists, and what mathematical model is used in the current `kspecdr` implementation for Isoplane data.

It complements the procedure-oriented guide in [Isoplane Data Reduction Workflow](../tutorials/isoplane-end-to-end.md).

## 1. End-to-End View

```{mermaid}
flowchart TD
    A["Raw Isoplane FITS"] --> B["Header Conversion + Orientation Normalization"]
    B --> C["IM Preprocessing (Bias/Dark/Variance/Masking)"]
    C --> D["Calibration Frame Combination"]
    D --> E["Fiber Flat IM -> Tramline Map (TLM)"]
    E --> F["Arc IM -> Extraction (EX)"]
    F --> G["Arc Wavelength Calibration -> RED (WAVELA, SHIFTS)"]
    E --> H["Science IM -> Extraction (EX)"]
    G --> I["Science Rebin/Scrunch using Arc Solution"]
    H --> I
    I --> J["Final Calibrated Science Product"]
```

## 2. Mathematical Notation

We use the following symbols throughout:

$$
I(x,y): \text{image intensity at spatial pixel }x\text{ and spectral pixel }y
$$

$$
V(x,y): \text{variance of }I(x,y)
$$

$$
g: \text{detector gain (e}^- / \text{ADU}), \quad \sigma_r: \text{read noise (e}^-)
$$

$$
\lambda(y): \text{wavelength as a function of spectral pixel}
$$

$$
T_f(y): \text{tramline center of fiber }f\text{ at spectral pixel }y
$$

## 3. Stage 1: Isoplane Header Conversion and Normalization

### 3.1 What is considered

- Raw Isoplane metadata is instrument-specific and not fully compatible with downstream modules.
- Downstream logic requires standardized keys such as `INSTRUME`, `RO_GAIN`, `RO_NOISE`, `LAMBDAC`, `DISPERS`, and `DISPAXIS`.
- Orientation inconsistencies must be corrected before trace detection and extraction.

### 3.2 Strategy

- Sanitize invalid FITS cards.
- Map readout mode keywords to calibrated gain/noise values.
- Convert timing to standard seconds and UTC fields.
- Parse grating settings and compute dispersion estimate.
- Flip image axes when orientation keywords indicate inversion.
- Attach a valid `FIBRES` table (dummy or user-supplied).

### 3.3 Core equations

Exposure time:

$$
t_{\mathrm{exp}} = \frac{t_{\mathrm{ms}}}{1000}
$$

Reciprocal linear dispersion model:

$$
\mathrm{RLD}\ [\text{\AA/mm}] = 23.0 \times \frac{1200}{L_{\mathrm{mm}}}
$$

Pixel-size conversion:

$$
p_{\mathrm{mm}} = p_{\mu \mathrm{m}} \times 10^{-3}
$$

Pixel dispersion:

$$
\mathrm{DISPERS}\ [\text{\AA/pixel}] = \mathrm{RLD} \times p_{\mathrm{mm}}
$$

Central wavelength:

$$
\lambda_c\ [\text{\AA}] = 10 \times \lambda_c\ [\text{nm}]
$$

## 4. Stage 2: IM Preprocessing (`make_im`)

### 4.1 What is considered

- Hot/bad/saturated pixels can destabilize trace and extraction.
- Bias and dark are additive detector/system backgrounds.
- Variance must be explicit for statistically meaningful extraction/calibration.
- Flat-field and cosmic-ray operations should be optional and controlled.

### 4.2 Strategy

- Apply bad-pixel masks and mark saturated regions.
- Subtract bias using either:
  - scalar median level, or
  - full 2D master bias.
- Subtract dark after exposure-time scaling.
- Initialize or update variance.
- Optionally divide by long-slit flat and remove cosmic rays.

### 4.3 Equations

Bias subtraction:

$$
I_b(x,y) = I_{\mathrm{raw}}(x,y) - B(x,y)
$$

Dark scaling and subtraction:

$$
s_d = \frac{t_{\mathrm{obj}}}{t_{\mathrm{dark}}}, \qquad
I_{bd}(x,y) = I_b(x,y) - s_d D(x,y)
$$

Initial variance model:

$$
V_0(x,y) = \sigma_r^2 + \frac{\max(I_{bd}(x,y), 0)}{g}
$$

Flat-field correction (`F` = flat image):

$$
I'(x,y) = \frac{I_{bd}(x,y)}{F(x,y)}
$$

Implemented variance update for flat division:

$$
V'(x,y) = \frac{V_0(x,y)}{F(x,y)^2}
          + \frac{I'(x,y)^2\,V_F(x,y)}{F(x,y)^2}
$$

### 4.4 Flow

```{mermaid}
flowchart TD
    A["Converted FITS"] --> B["Create IM (float image + VARIANCE HDU)"]
    B --> C["Apply bad/saturated pixel masking"]
    C --> D["Bias correction"]
    D --> E["Dark correction"]
    E --> F["Variance initialization/update"]
    F --> G["Optional: LFLAT division"]
    G --> H["Optional: CR cleaning (LACOSMIC)"]
    H --> I["Output *_im.fits"]
```

## 5. Stage 3: Calibration-Frame Combination (`combine_image`, `reduce_*`)

### 5.1 What is considered

- Multiple exposures reduce random noise and reject outliers.
- Different frame classes require different combination policies.
- Darks may need separate masters by exposure time.

### 5.2 Strategy

- Build stack over input IM frames.
- Optionally normalize level per frame (central median) before combining.
- Combine using `MEAN`, `MEDIAN`, or `SIGMA_CLIP`.
- Propagate or estimate variance for combined frame.

### 5.3 Equations

Mean combine:

$$
I_{\mathrm{mean}} = \frac{1}{N}\sum_{i=1}^{N} I_i
$$

$$
V_{\mathrm{mean}} = \frac{1}{N^2}\sum_{i=1}^{N} V_i
$$

Median combine variance approximation:

$$
V_{\mathrm{median}} \approx \frac{\pi}{2}\,V_{\mathrm{mean}}
$$

Sigma-clipped variance on surviving samples:

$$
V_{\mathrm{clip}} = \frac{\sum V_i^{(\mathrm{valid})}}{N_{\mathrm{valid}}^2}
$$

## 6. Stage 4: Tramline Map Generation (`make_tlm`)

### 6.1 What is considered

- Fiber traces are curved and may shift with optical distortion.
- Peak finding must be robust against noise and local background variation.
- Traces must be linked consistently across sampled columns.

### 6.2 Strategy

- Sample image in spectral direction (`STEP=50` in current code).
- Build spatial profile in each sampled column window.
- Detect candidate peaks (Isoplane default uses wavelet-based detection).
- Link peaks with multi-target tracking (Hungarian assignment with gating).
- Fit polynomial trace model and interpolate/extrapolate missing fibers.
- Write TLM and predicted wavelength extension.

### 6.3 Peak detection mathematics

The wavelet threshold uses a robust scale:

$$
\sigma_{\mathrm{robust}} = 1.4826 \cdot \mathrm{MAD}(R)
$$

$$
z_{\mathrm{tol}} = k_{\mathrm{MAD}} \cdot \sigma_{\mathrm{robust}}
$$

where \(R\) is the wavelet response and \(k_{\mathrm{MAD}}\) is a tunable multiplier.

Sub-pixel peak center from zero crossings:

$$
x_{\mathrm{peak}} = \frac{x_{\mathrm{L0}} + x_{\mathrm{R0}}}{2}
$$

### 6.4 Tracking mathematics

Assignment cost between existing track and current peak:

$$
C = \sqrt{(\Delta x)^2 + (\Delta s)^2}
$$

with gating constraints:

$$
|\Delta x| \le d_{\max}, \qquad \Delta s \le s_{\max}
$$

### 6.5 Predicted wavelength in TLM

For spectral pixel center \(y\):

$$
\lambda_{\mathrm{pred}}(y) = \lambda_c + d_\lambda\,(y-y_{\mathrm{mid}})
$$

where \(d_\lambda = \mathrm{DISPERS}\), \(y_{\mathrm{mid}} = 0.5\,N_{\mathrm{spec}}\).

### 6.6 Flow

```{mermaid}
flowchart TD
    A["Input flat IM"] --> B["Column sampling + spatial profile"]
    B --> C["Wavelet peak detection"]
    C --> D["Track linking (MTT + Hungarian)"]
    D --> E["Polynomial trace fit per track"]
    E --> F["Trace-to-fiber matching"]
    F --> G["Interpolate/extrapolate missing fibers"]
    G --> H["Write TLM + predicted WAVELA"]
```

## 7. Stage 5: Extraction (`make_ex`, SUM path)

### 7.1 What is considered

- Extraction aperture must follow tramline center for each fiber and spectral pixel.
- Partial-pixel overlap is needed for sub-pixel center locations.
- Invalid pixels and disabled fiber classes should be handled explicitly.

### 7.2 Strategy

- For each \((y,f)\), define aperture centered at \(T_f(y)\) with width \(W\).
- Integrate all touched spatial pixels by overlap fraction.
- Accumulate flux and variance over aperture.
- Zero fibers of types `F`, `N`, `U`.

### 7.3 Equations

Aperture bounds:

$$
x_{\mathrm{low}} = T_f(y) - \frac{W}{2}, \qquad
x_{\mathrm{high}} = T_f(y) + \frac{W}{2}
$$

Flux (implemented):

$$
F_f(y) = \sum_{x \in \mathcal{A}(y,f)} I(x,y)\,w_x
$$

Variance (implemented):

$$
V_f(y) = \sum_{x \in \mathcal{A}(y,f)} V(x,y)\,w_x
$$

where \(w_x \in [0,1]\) is the overlap fraction for pixel \(x\).

## 8. Stage 6: Arc Wavelength Calibration (`reduce_arc`, `reduce_arcs`)

### 8.1 What is considered

- The predicted wavelength from header is approximate.
- Arc line matching needs robust handling of blends, saturation, and outliers.
- Fiber-to-fiber distortion requires landmark synchronization.

### 8.2 Strategy

- Select a robust reference fiber near detector center.
- Build a high-S/N template by fiber synchronization and averaging.
- Read lamp line table (`*.arc`) and mask problematic blends.
- Generate synthetic arc model and estimate shift field by cross-correlation.
- Refine line centroids and fit global polynomial wavelength model robustly.
- Propagate calibration to all fibers and write `WAVELA`/`SHIFTS`.

### 8.3 Key equations

Line-width estimate from wavelet zero-crossing gap \(w\):

$$
\sigma_{\mathrm{line}} = \sqrt{\left(\frac{w}{2}\right)^2 - a^2}
$$

Resonance scale:

$$
a_{\mathrm{res}} = \sqrt{5}\,\sigma_{\mathrm{line}}
$$

Adaptive wavelet threshold:

$$
z_{\mathrm{tol}} =
\left(3\,\sigma_{\mathrm{noise}}\right)
\frac{\max(\mathrm{CWT}_{a_{\mathrm{res}}})}{\max(S)}
$$

Synthetic arc model:

$$
M(\lambda)=\sum_i A_i
\exp\!\left(-\frac{(\lambda-\mu_i)^2}{2\sigma_\lambda^2}\right)
$$

Blend criterion:

$$
\Delta\lambda < 3\,\sigma_\lambda
$$

Robust residual rejection:

$$
\left|r_i-\mathrm{median}(r)\right| \ge 3\cdot \mathrm{MAD}(r)
\;\Rightarrow\; \text{outlier}
$$

### 8.4 Flow

```{mermaid}
flowchart TD
    A["Arc EX + predicted WAVELA + line list"] --> B["Reference fiber selection"]
    B --> C["Wavelet landmark detection per fiber"]
    C --> D["Landmark tracking across fibers"]
    D --> E["Fiber synchronization + template construction"]
    E --> F["Synthetic model from lamp table"]
    F --> G["Cross-correlation shift path estimation"]
    G --> H["Peak refinement + robust polynomial fit"]
    H --> I["Propagate solution to all fibers"]
    I --> J["Write RED with calibrated WAVELA and SHIFTS"]
```

## 9. Stage 7: Science Path in Current Implementation

### 9.1 What is considered

- Full `reduce_object` wrapper is not yet complete.
- Reliable current approach is staged processing using implemented modules.

### 9.2 Strategy

1. Convert raw science frames.
2. Run `make_im`.
3. Run `make_ex` with the same TLM used for arcs/flats.
4. Rebin/scrunch science EX using calibrated arc RED solution.

### 9.3 Rebinning equation (current interpolation approach)

For each fiber \(f\):

$$
S^{\mathrm{out}}_f(\lambda_j) =
\mathcal{I}\!\left(S^{\mathrm{in}}_f(\lambda), \lambda_j\right)
$$

where \(\mathcal{I}\) is linear interpolation on wavelength coordinates.

### 9.4 Flow

```{mermaid}
flowchart TD
    A["Science converted FITS"] --> B["Science IM"]
    B --> C["Science EX (SUM extraction)"]
    C --> D["Apply arc RED wavelength solution"]
    D --> E["Scrunched/calibrated science output"]
```

## 10. Cross-Stage Quality Control Logic

Recommended checks:

- Conversion:
  - required header keys and orientation validity.
  - `FIBRES` extension count.
- IM:
  - expected bias/dark subtraction levels.
  - saturation fraction and variance scale sanity.
- TLM:
  - trace count and continuity vs expected fibers.
- Extraction:
  - NaN/bad fraction and fiber consistency.
- Arc calibration:
  - matched-line count, residual median/MAD/RMS.
  - wavelength monotonicity and smoothness.
- Science:
  - skyline placement and repeatability across frames/fibers.

## 11. Assumptions and Limits

Current assumptions:

- Read-noise + Poisson-like variance model is adequate for commissioning.
- Isoplane matching can be approximated by ordered 1-to-1 trace mapping.
- SUM extraction is acceptable for current use.
- Interpolation-based scrunch is acceptable for current goals.

Current limits:

- No production optimal extraction path yet.
- `reduce_object` full chain is not complete.
- Some heuristics (thresholds, blend masks, width criteria) may require setup-specific tuning.

## 12. Practical Interpretation

The current architecture is strongest as a transparent staged reduction framework:

- each intermediate product can be inspected,
- statistical logic is explicit,
- and the mathematical structure is already suitable for incremental upgrades (optimal extraction, complete object wrapper, stricter flux-conserving rebinning).
