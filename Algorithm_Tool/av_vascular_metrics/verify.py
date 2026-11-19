"""Synthetic geometry fixtures, not clinical validation."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from Algorithm_Tool.av_vascular_metrics.runner import compute, fractal, read, tortuosity


def run(root):
    root.mkdir()
    n = 256
    yy,xx = np.indices((n,n))
    disc = (xx-128)**2+(yy-128)**2 <= 16**2
    artery,vein = np.zeros_like(disc),np.zeros_like(disc)
    artery[50:55,20:230] = True
    vein[20:230,70:76] = True
    arrays = {"artery_mask":artery,"vein_mask":vein,"disc_mask":disc}
    inputs,bindings = {},{}
    for key,a in arrays.items():
        path = root/(key+".png")
        Image.fromarray(a.astype(np.uint8)*255).save(path)
        inputs[key],bindings[key] = read(path)
    r = compute(*inputs.values(),bindings)
    checks = {}
    checks["deterministic"] = r == compute(*inputs.values(),bindings)
    checks["overlap_excluded"] = r["overlap_pixels"]==30 and r["classes"]["artery"]["whole_image"]["vessel_pixels"]==1020
    checks["density_exact"] = r["classes"]["vein"]["whole_image"]["area_density"]==(1260-30)/(n*n)
    checks["separate_classes"] = r["classes"]["artery"] != r["classes"]["vein"]
    checks["annulus_explicit"] = r["disc"]["annulus_pixels"]>0 and .99<r["disc"]["annulus_fraction_in_image"]<=1
    sk = np.zeros_like(disc)
    sk[128,32:224] = True
    straight = tortuosity(sk,32)
    checks["straight_ratio_one"] = straight["count"]==1 and straight["mean"]==1
    checks["rotated_straight"] = tortuosity(np.rot90(sk),32)["mean"]==1
    diagonal = np.eye(n,dtype=bool)
    checks["diagonal_euclidean"] = math.isclose(tortuosity(diagonal,32)["mean"],1,abs_tol=1e-12)
    bend = np.zeros_like(disc)
    bend[100,30:150]=True
    bend[100:220,149]=True
    checks["bent_greater_one"] = tortuosity(bend,32)["mean"]>1.3
    tiny = np.zeros_like(disc)
    tiny[10,10:14] = True
    checks["short_excluded"] = tortuosity(tiny,32)["count"]==0
    checks["line_fractal_one"] = .8<fractal(sk)["value"]<1.2
    checks["plane_fractal_two"] = math.isclose(fractal(np.ones_like(sk))["value"],2,abs_tol=1e-12)
    checks["empty_fractal_null"] = fractal(np.zeros_like(sk))["value"] is None
    empty = compute(np.zeros_like(sk),np.zeros_like(sk),disc,bindings)
    checks["empty_width_null"] = empty["classes"]["artery"]["whole_image"]["width_proxy_pixels"] is None
    checks["empty_tortuosity_null"] = empty["classes"]["artery"]["tortuosity"]["mean"] is None
    checks["width_normalized"] = r["classes"]["artery"]["whole_image"]["width_proxy_disc_ratio"] == r["classes"]["artery"]["whole_image"]["width_proxy_pixels"]/r["disc"]["diameter_pixels"]
    for name,data in [("nonbinary",np.ones((20,20),dtype=np.uint8)),("rgb",np.zeros((20,20,3),dtype=np.uint8))]:
        path = root/(name+".png")
        Image.fromarray(data).save(path)
        try:
            read(path)
        except ValueError:
            checks[name]=True
        else:
            checks[name]=False
    for name,aa,vv,dd in [("mismatch",artery[:20],vein,disc),("empty_disc",artery,vein,np.zeros_like(disc)),("split_disc",artery,vein,disc|(xx<2))]:
        try:
            compute(aa,vv,dd,bindings)
        except ValueError:
            checks[name]=True
        else:
            checks[name]=False
    clipped = compute(artery,vein,(xx-2)**2+(yy-2)**2<=16**2,bindings)
    checks["clipped_annulus_reported"] = clipped["disc"]["annulus_fraction_in_image"]<.5
    (root/"result.json").write_text(json.dumps(r))
    (root/"checks.json").write_text(json.dumps(checks))
    if not all(checks.values()):
        raise ValueError(str([k for k,v in checks.items() if not v]))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args().output)
