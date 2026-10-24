import os
import sys
import tempfile
import logging
import warnings

import numpy as np
from astropy.io import fits

# Add src to path
sys.path.append(os.path.abspath('src'))

from kspecdr.inst.isoplane import (
    convert_isoplane_header,
    ics_frame_info,
    write_isoplane_converted_image,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def lightfield_header(ny=1330, nx=1340):
    """The LightField cards convert_isoplane_header reads (as in the 2026 PIXIS frames)."""
    h = fits.Header()
    h["DATE-OBS"] = "2026-11-01T08:45:07"
    h["EXPTIME"] = "1800000"
    h["HIERARCH PI CAMERA SHUTTERTIMING EXPOSURETIME"] = "1800000"
    # the 2026 orientation cards: the converter flips the rows, not the columns
    h["HIERARCH PI ACQUISITION ORIENTATION LIGHTPATHORIENTATIONUNCORRECTED"] = "FlippedHorizontally"
    h["HIERARCH PI CAMERA EXPERIMENT ACQUISITION ORIENTATION RESULT"] = "Normal"
    h["HIERARCH PI SPECTROMETER GRATING SELECTED"] = "[500nm,600][3][0]"
    h["HIERARCH PI SPECTROMETER GRATING CENTERWAVELENGTH"] = "430"
    h["HIERARCH PI CAMERA SENSOR INFORMATION PIXEL WIDTH"] = "20"
    h["HIERARCH PI CAMERA SENSOR INFORMATION SENSORNAME"] = "PIXIS 1300BX"
    h["HIERARCH PI CAMERA ADC SPEED"] = "0.1"
    h["HIERARCH PI CAMERA ADC ANALOGGAIN"] = "High"
    h["HIERARCH PI CAMERA ADC QUALITY"] = "LowNoise"
    return h


def ics_cards(h, **over):
    """The cards the ICS IsoPlane server adds (KSPEC_ICS SPECTRO_ISOPLANE/fitsio.py)."""
    cards = {"OBSTYPE": "Object", "ICSFUNC": "getobj", "OBSDATE": "20261101", "PROJID": "P1",
             "TILEID": "0123", "OBJECT": "NGC 1", "TILERA": "10:00:00", "TILEDEC": "-30:00:00",
             "EXPREQ": 1800.0, "EXPNUM": 2, "NEXP": 3,
             "OBSERVAT": "KMTNet SSO", "SITELAT": -31.273361, "SITELONG": 149.061194,
             "SITEELEV": 1149.0, "UTSTART": "2026-11-01T08:15:02.123+00:00",
             "UTEND": "2026-11-01T08:45:02.125+00:00", "SPECBACK": "ISOPLANE",
             "DATASEC": "[1:1340,1:1299]", "BIASSEC": "[1:1340,1300:1330]", **over}
    h = h.copy()
    for key, value in cards.items():
        if value is not None:
            h[key] = value
    return h


def check(label, cond):
    print(f"{'SUCCESS' if cond else 'FAIL'} [{label}]")
    return bool(cond)


def test_ics_frame():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = convert_isoplane_header(ics_cards(lightfield_header()))
    return all([
        check("class from getobj + OBSTYPE", h["NDFCLASS"] == "MFOBJECT"),
        check("times from the SPE stamps",
              (h["UTSTART"], h["UTEND"], h["TIMESRC"]) == ("08:15:02", "08:45:02", "SPE")),
        check("full stamps kept", h["DATE-BEG"] == "2026-11-01T08:15:02.123"),
        check("UTMJD at the start", abs(h["UTMJD"] - (61345 + (8 * 3600 + 15 * 60 + 2.123) / 86400))
              < 1e-8),
        check("ICS cards kept", (h["EXPNUM"], h["NEXP"], h["TILEID"], h["OBJECT"])
              == (2, 3, "0123", "NGC 1")),
    ])


def test_classes_and_fallbacks():
    lf = lightfield_header()
    ok = []
    for func, obstype, want in (("getbias", "Bias", "BIAS"), ("getdark", "Dark", "DARK"),
                                ("getflat", "Flat", "MFFFF"), ("getarc", "Arc", "MFARC"),
                                ("getbias", "Object", "BIAS"),   # the command wins
                                ("getobj", "Bias", "BIAS")):     # old GUI, shutter open
        info = ics_frame_info(ics_cards(lf, ICSFUNC=func, OBSTYPE=obstype))
        ok.append(check(f"{func}/{obstype} -> {want}", info["ndfclass"] == want))
    h = convert_isoplane_header(ics_cards(lf, UTSTART="None", UTEND="None"))
    ok.append(check("no stamp: times from DATE-OBS",
                    (h["UTSTART"], h["TIMESRC"]) == ("08:45:07", "DATE-OBS")))
    h = convert_isoplane_header(ics_cards(lf), ndfclass="MFARC")
    ok.append(check("explicit class overrides", h["NDFCLASS"] == "MFARC"))
    info = ics_frame_info(ics_cards(lf))
    ok.append(check("site", abs(info["site"].geodetic.lat.deg + 31.273361) < 1e-9))
    return all(ok)


def test_frame_without_ics_cards():
    lf = lightfield_header()
    h = convert_isoplane_header(lf, "MFFFF")
    ok = [check("January frame: class from the caller", h["NDFCLASS"] == "MFFFF"),
          check("January frame: no ICS info", ics_frame_info(lf) is None),
          check("January frame: UTSTART = DATE-OBS", h["UTSTART"] == "08:45:07"),
          check("January frame: no TIMESRC", "TIMESRC" not in h)]
    try:
        convert_isoplane_header(lf)
        ok.append(check("January frame without a class raises", False))
    except ValueError:
        ok.append(check("January frame without a class raises", True))
    return all(ok)


def test_sections_follow_the_flip():
    ok = []
    with tempfile.TemporaryDirectory() as d:
        raw, out = os.path.join(d, "raw.fits"), os.path.join(d, "conv.fits")
        data = np.zeros((1330, 1340), dtype=np.uint16)
        data[1299:] = 1  # raw bias rows 1300-1330
        hdu = fits.PrimaryHDU(data, ics_cards(lightfield_header()))
        hdu.header.set("EXTEND", True, after="NAXIS2")  # as in the LightField frames
        hdu.writeto(raw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_isoplane_converted_image(raw, out, n_fibers=3)
        with fits.open(out) as h:
            conv, hdr = h[0].data, h[0].header
        ok.append(check("BIASSEC on the converted rows", hdr["BIASSEC"] == "[1:1340,1:31]"
                        and np.all(conv[:31] == 1) and np.all(conv[31:] == 0)))
        ok.append(check("raw section kept", hdr["RBIASSEC"] == "[1:1340,1300:1330]"))
    return ok and all(ok)


if __name__ == "__main__":
    results = [
        test_ics_frame(),
        test_classes_and_fallbacks(),
        test_frame_without_ics_cards(),
        test_sections_follow_the_flip(),
    ]
    sys.exit(0 if all(results) else 1)
