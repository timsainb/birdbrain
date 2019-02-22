import urllib
from tqdm import tqdm
from birdbrain.utils import ensure_dir

# from pyunpack import Archive
import patoolib
from glob import glob
import numpy as np
import shutil
import os
import subprocess


def execute(cmd):
    """excecute a command in subprocess
    """
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def verbose_command(cmd):
    """ prints the executed command
    """
    for path in execute(cmd):
        print(path, end="")


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
    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=url.split("/")[-1],
        leave=False,
    ) as t:  # all optional kwargs
        urllib.request.urlretrieve(
            url, filename=save_loc, reporthook=t.update_to, data=None
        )


def get_canary_data():
    # if files already exist
    if len(glob("../../data/processed/canary/delineations/*.img")) > 1:
        print("Data already download")
        return
    print("Downloading data")

    # which images to get and where to save them
    dl_output = "../../data/raw/canary/"
    img_output = "../../data/processed/canary/"
    images = [
        "http://extranet.orn.mpg.de/documents/vellema/downloads/T2_Rare_3D.rar",
        "http://extranet.orn.mpg.de/documents/vellema/downloads/T2_Flash_3D_HR.rar",
    ]
    delineations = [
        "http://extranet.orn.mpg.de/documents/vellema/downloads/skull.rar",
        "http://extranet.orn.mpg.de/documents/vellema/downloads/brain.rar",
        "http://extranet.orn.mpg.de/documents/vellema/downloads/subdivisions.rar",
        "http://extranet.orn.mpg.de/documents/vellema/downloads/nuclei.rar",
        "http://extranet.orn.mpg.de/documents/vellema/downloads/tracts.rar",
        "http://extranet.orn.mpg.de/documents/vellema/downloads/combined_set.rar",
    ]
    # make sure directories all exist
    ensure_dir(dl_output)
    ensure_dir(img_output)
    ensure_dir(img_output + "delineations/")

    # download data
    delination_zip = []
    # download each delineation file
    for delineation_loc in delineations:
        output_loc = dl_output + delineation_loc.split("/")[-1]
        tqdm_download(delineation_loc, output_loc)
        delination_zip.append(output_loc)
    img_zip = []
    # download each image file
    for image_loc in images:
        output_loc = dl_output + image_loc.split("/")[-1]
        tqdm_download(image_loc, output_loc)
        img_zip.append(output_loc)
    # unzip delineations
    for dz in tqdm(delination_zip):
        patoolib.extract_archive(dz, outdir=img_output + "delineations/")
    # unzip images
    for iz in tqdm(img_zip):
        patoolib.extract_archive(iz, outdir=img_output)


def get_starling_data():
    if len(glob("../../data/processed/starling/delineations/*.img")) > 1:
        print("Data already download")
        return

    print("Downloading data")
    dl_output = "../../data/raw/starling/"  # data download
    img_output = "../../data/processed/starling/"  # processed save spot
    data_url = "http://uahost.uantwerpen.be/bioimaginglab/starling.zip"

    # ensure directories
    ensure_dir(dl_output)
    ensure_dir(img_output)
    ensure_dir(img_output + "delineations/")
    zip_loc = dl_output + "starling.zip"

    # download data
    tqdm_download(data_url, zip_loc)

    # extract the data
    patoolib.extract_archive(zip_loc, outdir=dl_output)

    # move the data to the correct location
    for img_file in np.concatenate(
        [glob(dl_output + "ATLAS_starling/*." + ed) for ed in ["img", "hdr"]]
    ):
        shutil.move(img_file, img_output + os.path.basename(img_file))
    for img_file in np.concatenate(
        [
            glob(dl_output + "ATLAS_starling/delineations/*." + ed)
            for ed in ["img", "hdr", "txt"]
        ]
    ):
        shutil.move(img_file, img_output + "delineations/" + os.path.basename(img_file))


def get_zebra_finch_data(password):
    if len(glob("../../data/processed/zebra_finch/delineations/*.img")) > 1:
        print("Data already download")
        return

    print("Downloading data")
    dl_output = "../../data/raw/zebra_finch/"  # data download
    img_output = "../../data/processed/zebra_finch/"  # processed save spot
    data_url = "http://uahost.uantwerpen.be/bioimaginglab/zebrafinch.zip"

    # ensure directories
    ensure_dir(dl_output)
    ensure_dir(img_output)
    ensure_dir(img_output + "delineations/")
    zip_loc = dl_output + "zebra_finch.zip"

    # download data
    tqdm_download(data_url, zip_loc)

    # unzip files
    cmd = ["unzip", "-P", password, zip_loc, "-d", dl_output]
    verbose_command(cmd)

    # images of interest need to be moved
    atlas_img = "../../data/raw/zebra_finch/atlas/atlas.img"
    atlas_hdr = "../../data/raw/zebra_finch/atlas/atlas.hdr"
    brain_del_img = "../../data/raw/zebra_finch/atlas/brain_delineations.img"
    brain_del_hdr = "../../data/raw/zebra_finch/atlas/brain_delineations.hdr"
    nuc_del_img = "../../data/raw/zebra_finch/atlas/nuclei_delineations.img"
    nuc_del_hdr = "../../data/raw/zebra_finch/atlas/nuclei_delineations.hdr"

    # move atlas files
    shutil.move(atlas_img, img_output + os.path.basename(atlas_img))
    shutil.move(atlas_hdr, img_output + os.path.basename(atlas_hdr))
    # move brain delineation files
    shutil.move(brain_del_img, img_output + "delineations/Brain.img")
    shutil.move(brain_del_hdr, img_output + "delineations/Brain.hdr")
    # move brain delineation files
    shutil.move(nuc_del_img, img_output + "delineations/Nuclei.img")
    shutil.move(nuc_del_hdr, img_output + "delineations/Nuclei.hdr")
