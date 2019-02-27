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
import pandas as pd
import xml.etree.ElementTree


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
        shutil.copy(img_file, img_output + os.path.basename(img_file))
    for img_file in np.concatenate(
        [
            glob(dl_output + "ATLAS_starling/delineations/*." + ed)
            for ed in ["img", "hdr", "txt"]
        ]
    ):
        shutil.copy(img_file, img_output + "delineations/" + os.path.basename(img_file))


def get_zebra_finch_data(password):
    if len(glob("../../data/processed/zebra_finch/delineations/*.img")) > 1:
        print("Data already download")
        return
    if password is None:
        raise ValueError('To request the zebra finch password email johan.vanaudekerke@uantwerpen.be')
                
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
    shutil.copy(atlas_img, img_output + os.path.basename(atlas_img))
    shutil.copy(atlas_hdr, img_output + os.path.basename(atlas_hdr))
    # move brain delineation files
    shutil.copy(brain_del_img, img_output + "delineations/Brain.img")
    shutil.copy(brain_del_hdr, img_output + "delineations/Brain.hdr")
    # move brain delineation files
    shutil.copy(nuc_del_img, img_output + "delineations/Nuclei.img")
    shutil.copy(nuc_del_hdr, img_output + "delineations/Nuclei.hdr")



def get_pigeon_data():
    """The pigeon data delineations exist across multiple files - so this list contains info on which delineations should be joined together 
    """
    SYSTEMS_DELINEATIONS = [
        ['Auditory1', [
            '../../data/raw/pigeon/Full_package/Auditory/auditory1.img'
        ]],
        ['Auditory2', [
            '../../data/raw/pigeon/Full_package/Auditory/auditory2.img'
                      ]],
        ['Olfactory', [
            '../../data/raw/pigeon/Full_package/Olfactory/Olfactory.img'
        ]],
        ['Visual_thalamofugal', [
            '../../data/raw/pigeon/Full_package/Visual/Thalamofugal/GLd-and-rotundus.img',
        ]],
        ['Somatosensory_wulst', [
            '../../data/raw/pigeon/Full_package/Somatosensory/Wulst_HA_HI_HD-frontal-from-A13.img'
        ]],
        ['Somatosensory_spinal_system_and_body_representation', [
            '../../data/raw/pigeon/Full_package/Somatosensory/Spinal system and body representation/GC_DLP_DIVA.img'
        ]],
        ['Somatosensory_trigeminal', [
            '../../data/raw/pigeon/Full_package/Somatosensory/Trigeminal/PrV-and-Basalis.img'
        ]],
        ['Hippocampus', [
            '../../data/raw/pigeon/Full_package/Hippocampus/hippocampus.img',
        ]],
        ['Visual_isthmic', [
            '../../data/raw/pigeon/Full_package/Visual/Isthmic nuclei/Isthmo-opticus.img',
            '../../data/raw/pigeon/Full_package/Visual/Isthmic nuclei/SLu-Ipc-Imc-right.img',
            '../../data/raw/pigeon/Full_package/Visual/Isthmic nuclei/SLu-Ipc-Imc-left.img'
        ]],
        ['Arcopallium', [
            '../../data/raw/pigeon/Full_package/Descending systems/Arcopallium/arcopallium.img'
        ]],
        ['Visual_wulst', [
            '../../data/raw/pigeon/Full_package/Visual/Thalamofugal/visual-Wulst_HA_HI_HD-until-A13.img',
        ]],

        ['Visual_aos', [
            '../../data/raw/pigeon/Full_package/Visual/Accessory Optic System/n-pontis-medialis.img',
            '../../data/raw/pigeon/Full_package/Visual/Accessory Optic System/nBOR-Lentiformis-mesencephali.img'
        ]],
        ['Brain', [
            '../../data/raw/pigeon/Full_package/Brainsurface/brainsurface_left.img',
            '../../data/raw/pigeon/Full_package/Brainsurface/brainsurface_right.img'
        ]],
        ['Nucleus_taeniae', [
            '../../data/raw/pigeon/Full_package/Olfactory/Nucleus-Taeniae.img',
        ]],
        ['Visual_tectofugal', [
            '../../data/raw/pigeon/Full_package/Visual/Tectofugal/entopallium.img',
            '../../data/raw/pigeon/Full_package/Visual/Tectofugal/rotundus.img',
            '../../data/raw/pigeon/Full_package/Visual/Tectofugal/Tectum-left.img',
            '../../data/raw/pigeon/Full_package/Visual/Tectofugal/Tectum-right.img',
        ]]
    ]


    if len(glob("../../data/processed/pigeon/delineations/*.img")) > 1:
        print("Data already download")
        return SYSTEMS_DELINEATIONS
    print("Downloading data")
    dl_output = "../../data/raw/pigeon/"  # data download
    img_output = "../../data/processed/pigeon/"  # processed save spot
    data_url = 'http://uahost.uantwerpen.be/bioimaginglab/pigeon.zip'

    # ensure directories
    ensure_dir(dl_output)
    ensure_dir(img_output)
    ensure_dir(img_output + "delineations/")
    zip_loc = dl_output + "pigeon.zip"

    # download data
    tqdm_download(data_url, zip_loc)

    # extract the data
    patoolib.extract_archive(zip_loc, outdir=dl_output)

    # There are several img files that dont correspond directly to the data description table at 
    #   https://www.uantwerpen.be/en/research-groups/bio-imaging-lab/research/mri-atlases/pigeon-brain-atlas/manual/
    #   I made a quick try at matching them up but this needs fixed

    # brainsurface will need to be merged...
    #delineation_files = ['Auditory', 'Descending systems', 'Hippocampus', 'Olfactory', 'Somatosensory', 'Brainsurface']
    img_files = ['CT', 'T2', 'T2star']

    # for some reason pigeon brain is 

    # get all hdr and img delination files
    all_delineation_files = np.concatenate([i[1] + [j[:-4]+'.hdr' for j in i[1]] for i in SYSTEMS_DELINEATIONS])

    # get the image files
    img_files = [
        '../../data/raw/pigeon/Full_package/T2/T2.hdr',
        '../../data/raw/pigeon/Full_package/T2/T2.img',
        '../../data/raw/pigeon/Full_package/T2star/T2star.hdr',
        '../../data/raw/pigeon/Full_package/T2star/T2star.img',
        '../../data/raw/pigeon/Full_package/CT/CT.hdr',
        '../../data/raw/pigeon/Full_package/CT/CT.img',
    ]

    for img_file in img_files:
            shutil.copy(img_file, img_output + os.path.basename(img_file))


    for del_file in all_delineation_files:
            shutil.copy(del_file, img_output+'delineations/' + os.path.basename(del_file))
    
    
    return SYSTEMS_DELINEATIONS


def join_data_pigeon(pigeon_atlas):
    """ the pigeon dataset delinations are seperated in a different way than the other datasets
    this is just a small function that tries to fix that. There are still parts that are missing
    TODO: fix delineations better in pigeon
    """
    new_delineations = []
    for system, imgs in pigeon_atlas.systems_delineations:
        for imgi, img in enumerate(imgs):
            if imgi == 0:
                base = pigeon_atlas.voxel_data.loc[os.path.basename(img)[:-4].capitalize(), 'voxels']
            else:
                vox = pigeon_atlas.voxel_data.loc[os.path.basename(img)[:-4].capitalize(), 'voxels']
                base[vox!=0] = vox[vox!=0]
        aff = pigeon_atlas.voxel_data.loc[os.path.basename(img)[:-4].capitalize(), 'affine']
        new_delineations.append(pd.DataFrame([[system, imgs, base, aff]], columns = ['type_', 'src', 'voxels', 'affine']))
    new_delineations = pd.concat(new_delineations)
    new_delineations.index = new_delineations.type_

    # remove rows with the same name
    new_labs = [i[0] for i in pigeon_atlas.systems_delineations]
    pigeon_atlas.voxel_data = pigeon_atlas.voxel_data[np.array([i not in new_labs for i in pigeon_atlas.voxel_data.index])]
    # join with atlas
    pigeon_atlas.voxel_data = pd.concat([pigeon_atlas.voxel_data, new_delineations])

def get_mustached_bat_data():
    def xml_to_pandas_brainregions(xml_loc):
        """ gets a pandas dataframe of labels from the brainregions xml
        """
        # load xml
        e = xml.etree.ElementTree.parse(xml_loc).getroot()

        # get brain_label information
        brain_labels = pd.DataFrame(columns = ['label', 'region', 'type_'])
        for area in e.getchildren():
            (_, abbrev), (_, value), (_, _), (_, name) = area.items()
            brain_labels.loc[len(brain_labels)] = [int(value), abbrev, 'Mustached_bat_delineations']
        brain_labels.index = brain_labels.region
        brain_labels['label'] = brain_labels['label'].astype('int')
        return brain_labels

    if len(glob("../../data/processed/mustached_bat/delineations/*.nii")) > 1:
        print("Data already download")
        return xml_to_pandas_brainregions('../../data/processed/mustached_bat/delineations/Mustached_Bat_Delineations.atlas.xml')
    
    # image locations to save to 
    dl_output = "../../data/raw/mustached_bat/"
    img_output = "../../data/processed/mustached_bat/"
    # make sure directories all exist
    ensure_dir(dl_output)
    ensure_dir(img_output)
    ensure_dir(img_output + "delineations/")
    data_url = 'http://uahost.uantwerpen.be/bioimaginglab/Bat.zip'
    zip_loc = zip_loc = dl_output + "Bat.zip"
    # download data
    tqdm_download(data_url, zip_loc)
    # extract the data
    patoolib.extract_archive(zip_loc, outdir=dl_output)
    xml_loc = '../../data/raw/mustached_bat/Bat/Mustached_Bat_Atlas/Mustached_Bat_Delineations.atlas.xml'
    shutil.copy(xml_loc, img_output + "delineations/" + os.path.basename(xml_loc))
    xml_copy = img_output + "delineations/" + os.path.basename(xml_loc)
    brain_labels = xml_to_pandas_brainregions(xml_copy)
    
    # image files to move
    imgs = [
    '../../data/raw/mustached_bat/Bat/Mustached_Bat_Atlas/ad.nii',
    '../../data/raw/mustached_bat/Bat/Mustached_Bat_Atlas/b0.nii',
    '../../data/raw/mustached_bat/Bat/Mustached_Bat_Atlas/col_fa.nii',
    '../../data/raw/mustached_bat/Bat/Mustached_Bat_Atlas/fa.nii',
    '../../data/raw/mustached_bat/Bat/Mustached_Bat_Atlas/md.nii',
    '../../data/raw/mustached_bat/Bat/Mustached_Bat_Atlas/rd.nii',
    '../../data/raw/mustached_bat/Bat/Mustached_Bat_Atlas/Skull_CT.nii',
    '../../data/raw/mustached_bat/Bat/Mustached_Bat_Atlas/T2w_3D_RARE.nii',
    '../../data/raw/mustached_bat/Bat/Mustached_Bat_Atlas/tdi_dti_color.nii',
    '../../data/raw/mustached_bat/Bat/Mustached_Bat_Atlas/tdi_dti_grey.nii',
    ]
    # copy all the images
    for img in imgs:
        shutil.copy(img, img_output + os.path.basename(img))

    # copy all the images into delineations locations
    img = '../../data/raw/mustached_bat/Bat/Mustached_Bat_Atlas/Mustached_Bat_Delineations.nii'
    shutil.copy(img, img_output + "delineations/" + os.path.basename(img))
    # copy all the images into delineations locations
    img = '../../data/raw/mustached_bat/Bat/Mustached_Bat_Atlas/mask.nii'

    shutil.copy(img, img_output + "delineations/" + 'Brain.nii')
    return brain_labels