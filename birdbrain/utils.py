import numpy as np
import pandas as pd

import nibabel
from tqdm.autonotebook import tqdm
import scipy.ndimage

# for transforming via the affine to um
from nipy.algorithms.registration.affine import inverse_affine
from nibabel.affines import apply_affine


def get_voxel_data(img_files):
    """ collects voxel data from .img files using nibabel
    """
    print("Getting voxel data from .img files...")
    image_data = pd.DataFrame(columns=["type_", "src", "voxels", "affine"])
    for data_file in img_files:
        fn = data_file.split("/")[-1][:-4]
        dta = nibabel.load(data_file)
        image_data.loc[len(image_data)] = [fn, data_file, dta.get_data(), dta.affine]
    image_data = image_data.set_index(image_data.type_.values)
    return image_data


def get_region_voxels(atlas, switch_lateralization=False, verbose=False):
    """makes a pandas dataframe of voxels and location for each nucleus
    TODO: update this function so that where is only applied to the boundaries
    Arguments:
        voxel_data {[type]} -- [description]
    Keyword Arguments:
        switch_lateralization {bool} -- [choose which hemisphere
            to locate nucleus] (default: {False})
    """
    print("Getting location for each nucleus/region from voxel data...")

    # get the location of the y sinus in voxels
    y_sinus_vox = um_to_vox(
        atlas.voxel_data.loc["Brain", "affine"],
        {"medial-lateral": 0, "posterior-anterior": 0, "ventral-dorsal": 0},
        atlas.um_mult,
        atlas.y_sinus_um_transform,
    )

    # create a dataset of voxels for each
    region_vox = pd.DataFrame(
        columns=[
            "region",
            "type_",
            "reg_ID",
            # "voxels",
            "region_bounds",
            "coords_vox",
        ]
    )

    # go though each structure and find corresponding voxels
    for idx, (nucleus_ID, nucleus, type_) in tqdm(
        atlas.brain_labels.iterrows(), leave=False, total=len(atlas.brain_labels)
    ):

        # locate the structure
        reg_mask = atlas.voxel_data.loc[type_, "voxels"] == nucleus_ID
        # boundaries of coordinates
        xmin, xmax = np.where(reg_mask.sum(axis=1).sum(axis=1) > 0)[0][
            [0, -1]
        ] + np.array([-1, 1])
        ymin, ymax = np.where(reg_mask.sum(axis=0).sum(axis=1) > 0)[0][
            [0, -1]
        ] + np.array([-1, 1])
        zmin, zmax = np.where(reg_mask.sum(axis=0).sum(axis=0) > 0)[0][
            [0, -1]
        ] + np.array([-1, 1])
        # get points
        xmin = np.max([0, xmin])
        ymin = np.max([0, ymin])
        zmin = np.max([0, zmin])

        voxel_pts = np.array(list(np.where(reg_mask[xmin:xmax, ymin:ymax, zmin:zmax])))

        # add mins
        voxel_pts[0, :] += xmin
        voxel_pts[1, :] += ymin
        voxel_pts[2, :] += zmin

        region_bounds = [xmin, xmax, ymin, ymax, zmin, zmax]

        if np.shape(voxel_pts)[1] == 0:
            raise ValueError("Region " + nucleus + " " + type_ + " not found in data")

        # subset only the voxels in one hemisphere for localization
        if switch_lateralization:
            voxel_pts_hem = voxel_pts[
                :, voxel_pts[0, :] <= y_sinus_vox["medial-lateral"]
            ]
            # try the other side if there are no regions over here
            if np.shape(voxel_pts_hem)[1] == 0:
                voxel_pts_hem = voxel_pts[
                    :, voxel_pts[0, :] > y_sinus_vox["medial-lateral"]
                ]

        else:
            voxel_pts_hem = voxel_pts[
                :, voxel_pts[0, :] > y_sinus_vox["medial-lateral"]
            ]
            if np.shape(voxel_pts_hem)[1] == 0:
                voxel_pts_hem = voxel_pts[
                    :, voxel_pts[0, :] <= y_sinus_vox["medial-lateral"]
                ]

        # get the mean of the voxel points in one hemisphere
        voxel_mean = np.mean(voxel_pts_hem, axis=1)

        # get the voxel mean
        coords_vox = {
            "medial-lateral": int(round(voxel_mean[0])),  # medio-lateral
            # posterior-anterior
            "posterior-anterior": int(round((voxel_mean[1]))),
            "ventral-dorsal": int(round((voxel_mean[2]))),  # ventral-dorsal
        }

        region_vox.loc[len(region_vox)] = [
            nucleus,
            type_,
            nucleus_ID,
            # reg_mask[xmin - 1 : xmax + 1, ymin - 1 : ymax + 1, zmin - 1 : zmax + 1],
            region_bounds,
            coords_vox,
        ]
        if verbose:
            print(
                "**These coordinates are relative to the y sinus from the files, \
                which are a bit off from what Gentner Lab usuaually defines as y sinus**"
            )
            print(nucleus_ID, nucleus, type_)

    # set index as nucleus
    region_vox = region_vox.set_index(region_vox.region)
    return region_vox


def get_brain_labels(text_files):
    """ Gets brain labels from text files
        """
    print("Loading brain labels...")
    all_labs = []
    for txtfile in text_files:
        labs = pd.read_csv(txtfile, delimiter=" ", header=None)
        labs["type_"] = txtfile.split("/")[-1][:-4]
        labs.columns = ["label", "region", "type_"]
        all_labs.append(labs)
    brain_labels = pd.concat(all_labs)
    brain_labels = brain_labels.reset_index()
    # rename duplicates
    for idx, row in brain_labels[brain_labels.region.duplicated()].iterrows():
        brain_labels.loc[idx, "region"] = row.region + "_" + row.type_
    # set index as regions
    brain_labels = brain_labels.set_index(brain_labels.region)
    brain_labels = brain_labels[["label", "region", "type_"]]
    return brain_labels.dropna()


def loc_dict_to_list(loc):
    return [loc["medial-lateral"], loc["posterior-anterior"], loc["ventral-dorsal"]]


def vox_to_um(affine, loc, um_mult, um_transform):
    # apply affine to loc, then multiply by um mult
    movement_relative_to_affine = {
        axis: loc * um_mult
        for axis, loc in zip(
            ["medial-lateral", "posterior-anterior", "ventral-dorsal"],
            apply_affine(affine, loc),
        )
    }
    # take into account the transform from the y sinus
    movement_relative_to_y_sinus = {
        axis: movement_relative_to_affine[axis] - um_transform[axis]
        for axis in ["medial-lateral", "posterior-anterior", "ventral-dorsal"]
    }
    return movement_relative_to_y_sinus


def um_to_vox(affine, location_in_um, um_mult, um_transform):
    # remove the transform
    loc_um_minus_transform = {
        axis: location_in_um[axis] + um_transform[axis]
        for axis in ["medial-lateral", "posterior-anterior", "ventral-dorsal"]
    }
    # get inverse affine transform
    inv_affine = inverse_affine(affine)

    # apply multiplier (for um)
    loc_um_list = np.array(loc_dict_to_list(loc_um_minus_transform)) / um_mult

    # apply inverse affine
    um = apply_affine(inv_affine, loc_um_list).round().astype("int")

    um = {
        axis: umi
        for axis, umi in zip(
            ["medial-lateral", "posterior-anterior", "ventral-dorsal"], um
        )
    }

    return um

def smooth_voxels(voxels, padding_vox=10, sigma=2):
    """
    TODO: smoothing params and padding should be in um
    padding_vox: zero padding to apply to kernals to ensure that smoothing doesn't go over edges
    voxels: binary voxel mask
    sigma: std dev for gaussian
    """

    # pad voxels
    voxels_padded = np.pad(voxels, padding_vox, "constant", constant_values=0)
    # make a template on which to dump voxels
    vox_template = np.zeros(voxels_padded.shape, dtype=np.uint8)
    # this method allows overwriting of masks (only one mask per voxel)

    unique_labs = np.unique(voxels)[np.unique(voxels) != 0]

    for mask in tqdm(unique_labs, leave=False):

        vox = voxels_padded == mask

        # boundaries of coordinates
        xmin, xmax = np.where(vox.sum(axis=1).sum(axis=1) > 0)[0][[0, -1]] + np.array(
            [-padding_vox, padding_vox]
        )
        ymin, ymax = np.where(vox.sum(axis=0).sum(axis=1) > 0)[0][[0, -1]] + np.array(
            [-padding_vox, padding_vox]
        )
        zmin, zmax = np.where(vox.sum(axis=0).sum(axis=0) > 0)[0][[0, -1]] + np.array(
            [-padding_vox, padding_vox]
        )
        # ensure nothing is below 0
        xmin = np.max([0, xmin])
        ymin = np.max([0, ymin])
        zmin = np.max([0, zmin])

        # smooth with gaussian
        vox_smooth = scipy.ndimage.gaussian_filter(
            vox[xmin:xmax, ymin:ymax, zmin:zmax].astype("float32"), sigma
        )

        # thresholds to choose from
        threshs = np.linspace(0, 1, 20)
        # choose the best threshold base on same # of voxels as original
        thresh_sim = [
            np.abs(np.sum(vox_smooth > thresh) - np.sum(vox)) for thresh in threshs
        ]
        best_thresh = threshs[np.argmin(thresh_sim)]
        # threshold
        vox_threshed = vox_smooth > best_thresh
        # add to final template
        vox_template[xmin:xmax, ymin:ymax, zmin:zmax][vox_threshed] = mask

    vox_template = vox_template[
        padding_vox:-padding_vox, padding_vox:-padding_vox, padding_vox:-padding_vox
    ]
    return vox_template


def get_shell(x):
    """ assumes outermost shell is zeros
    """
    x = np.pad(x, 1, "constant", constant_values=0)
    base = x[1:-1, 1:-1, 1:-1]
    shell = base & (
        (
            (x[:-2, 1:-1, 1:-1])
            & (x[1:-1, :-2, 1:-1])
            & (x[1:-1, 1:-1, :-2])
            & (x[2:, 1:-1, 1:-1])
            & (x[1:-1, 2:, 1:-1])
            & (x[1:-1, 1:-1, 2:])
        )
        == False
    )

    return shell


def norm(x, x_low, x_high, rescale_low, rescale_high):
    return ((x - x_low) / (x_high - x_low)) * (rescale_high - rescale_low) + rescale_low

def rgb2hex(r, g, b):
    return int("0x{:02x}{:02x}{:02x}".format(r, g, b), 16)
