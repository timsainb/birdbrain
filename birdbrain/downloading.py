import urllib
from tqdm import tqdm
from birdbrain.utils import ensure_dir
from pyunpack import Archive
from glob import glob
import numpy as np
import shutil
import os

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize
        
def tqdm_download(url, save_loc):
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
              desc=url.split('/')[-1], leave=False) as t:  # all optional kwargs
        urllib.request.urlretrieve(url, filename=save_loc, reporthook=t.update_to,
                           data=None)


def get_canary_data():
    # if files already exist
    if len(glob('../../data/processed/canary/delineations/*.img')) > 1:
        print('Data already download')
        return
    print('Downloading data')
    
    # which images to get and where to save them
    dl_output = '../../data/raw/canary/'
    img_output = '../../data/processed/canary/'
    images = ['http://extranet.orn.mpg.de/documents/vellema/downloads/T2_Rare_3D.rar',
             'http://extranet.orn.mpg.de/documents/vellema/downloads/T2_Flash_3D_HR.rar'
             ]
    delineations = [
        'http://extranet.orn.mpg.de/documents/vellema/downloads/skull.rar',
        'http://extranet.orn.mpg.de/documents/vellema/downloads/brain.rar',
        'http://extranet.orn.mpg.de/documents/vellema/downloads/subdivisions.rar',
        'http://extranet.orn.mpg.de/documents/vellema/downloads/nuclei.rar',
        'http://extranet.orn.mpg.de/documents/vellema/downloads/tracts.rar',
        'http://extranet.orn.mpg.de/documents/vellema/downloads/combined_set.rar'
    ]
    # make sure directories all exist
    ensure_dir(dl_output)
    ensure_dir(img_output)
    ensure_dir(img_output+'delineations/')
    
    # download data
    delination_zip = []
    # download each delineation file
    for delineation_loc in delineations:
        output_loc = dl_output + delineation_loc.split('/')[-1]
        tqdm_download(delineation_loc, output_loc)
        delination_zip.append(output_loc)
    img_zip = []
    # download each image file
    for image_loc in images:
        output_loc = dl_output + image_loc.split('/')[-1]
        tqdm_download(image_loc, output_loc)
        img_zip.append(output_loc)
    # unzip delineations
    for dz in tqdm(delination_zip):
        Archive(dz).extractall(img_output+'delineations/')
    # unzip images
    for iz in tqdm(img_zip):
        Archive(iz).extractall(img_output)


def get_starling_data():
    if len(glob('../../data/processed/starling/delineations/*.img')) > 1:
        print('Data already download')
        return
    
    print('Downloading data')
    dl_output = '../../data/raw/starling/' # data download
    img_output = '../../data/processed/starling/' # processed save spot
    data_url = 'http://uahost.uantwerpen.be/bioimaginglab/starling.zip'
    
    # ensure directories
    ensure_dir(dl_output)
    ensure_dir(img_output)
    ensure_dir(img_output+'delineations/')
    zip_loc = dl_output+'starling.zip'

    # download data
    tqdm_download(data_url, zip_loc)
    
    # extract the data
    Archive(zip_loc).extractall(dl_output)
    
    # move the data to the correct location
    for img_file in np.concatenate([glob(dl_output+'ATLAS_starling/*.'+ed) for ed in ['img', 'hdr']]):
        shutil.move(img_file, img_output + os.path.basename(img_file))
    for img_file in np.concatenate([glob(dl_output+'ATLAS_starling/delineations/*.'+ed) for ed in ['img', 'hdr', 'txt']]):
        shutil.move(img_file, img_output + 'delineations/' + os.path.basename(img_file))
    