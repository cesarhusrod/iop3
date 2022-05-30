from iop3_pipeline import read_blazar_file, closest_blazar, has_near_calibrators
import os
from astropy.io import fits

import numpy as np
import pandas as pd

from mcFits import *
from mcReduction import *
import math 

from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
import astropy.units as u


config_dir="./conf"
input_path="/home/users/dreg/misabelber/GitHub/data/raw/MAPCAT/"

blazar_path=os.path.join(config_dir, 'blazar_photo_calib_last.csv')
blazar_data = read_blazar_file(blazar_path)

missing_blazars=np.array([])

for path, subdirs, files in os.walk(input_path):
    subdirs.sort()
    for name in files:
        if ".fits" in name:
            full_filepath=os.path.join(path,name)
            try:
                hdul=fits.open(full_filepath)
            except:
                continue
            if 'IMAGETYP' in hdul[0].header:
                if 'science' in hdul[0].header['IMAGETYP']:
                    obj=hdul[0].header['OBJECT']
                    if 'Flat' in obj or 'Dome' in obj or 'bias' in obj or 'flat' in obj \
                            or 'check' in obj or 'Test' in obj:
                        continue
                    try:
                        is_missing=not has_near_calibrators(full_filepath, blazar_data)
                    except:
                        continue
                    if is_missing:
                        if 'deg' in obj:
                            obj=obj[:-8]
                        if obj.split(" ")[0] in missing_blazars or obj in missing_blazars:
                            continue
                        print(obj)
                        missing_blazars=np.append(missing_blazars, obj)
                            
                        
