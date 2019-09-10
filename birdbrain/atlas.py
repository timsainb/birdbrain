import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
from birdbrain import utils
from birdbrain.utils import vox_to_um, inverse_dict
import pandas as pd
import birdbrain.downloading as dl
from birdbrain.paths import DATA_DIR, PROJECT_DIR, ASSETS_DIR


def load_species_data(dset_dir, delin_path, species, password=None):
    """ load brain region data for each species
    """
    systems_delineations = None

    if species == "canary":
        brain_labels = pd.read_csv(
            ASSETS_DIR / "csv" / "canary_regions.csv", index_col="region"
        )
        brain_labels.columns = ["label", "region", "type_"]
        dl.get_canary_data()

    elif species == "starling":
        dl.get_starling_data(dset_dir)
        # path of labels
        text_files = list(delin_path.glob("*.txt"))
        # transcription labels ['label', 'region', 'type_']
        if len(text_files) > 0:
            brain_labels = utils.get_brain_labels(text_files)

    elif species == "zebra_finch":
        brain_labels = pd.read_csv(
            ASSETS_DIR / "csv" / "zebra_finch_regions.csv", index_col="region"
        )
        brain_labels.columns = ["label", "region", "type_"]
        dl.get_zebra_finch_data(password)

    elif species == "pigeon":

        brain_labels = pd.read_csv(
            ASSETS_DIR / "csv" / "pigeon_regions.csv", index_col="region"
        )
        brain_labels.columns = ["label", "region", "type_"]
        systems_delineations = dl.get_pigeon_data()

    elif species == "mustached_bat":
        brain_labels = dl.get_mustached_bat_data()

    return brain_labels, systems_delineations


def load_images(delin_path, dset_dir):
    img_files = (
        list(delin_path.glob("*.nii"))
        + list(delin_path.glob("*.img"))
        + list(dset_dir.glob("*.nii"))
        + list(dset_dir.glob("*.img"))
    )
    return img_files


class atlas(object):
    def __init__(
        self,
        dset_dir=None,
        label_cmap=None,
        um_mult=100,
        img_cmap=None,
        smoothing=[],
        smoothing_sigma=2,
        updated_y_sinus=None,
        species=None,
        password=None,
    ):

        # get the dataset location
        if dset_dir is None:
            dset_dir = DATA_DIR / "processed" / species

        # path of delineations
        delin_path = dset_dir / "delineations"

        self.brain_labels, self.systems_delineations = load_species_data(
            dset_dir, delin_path, species, password
        )

        # how axes labels relate to affine transformed data in voxels
        self.axes_dict = {
            "medial-lateral": 0,
            "anterior-posterior": 1,
            "dorsal-ventral": 2,
        }
        self.inverse_axes_dict = inverse_dict(self.axes_dict)

        # path of images
        img_files = load_images(delin_path, dset_dir)

        # images from each type of scan, as well as transcribed locations ['type_', 'src', 'voxels']
        self.voxel_data = utils.get_voxel_data(img_files)

        if species == "pigeon":
            dl.join_data_pigeon(self)

        # smooth the whole brain because the atlas is a bit noisy
        for img in smoothing:
            self.voxel_data.loc[img, "voxels"] = utils.smooth_voxels(
                self.voxel_data.loc[img, "voxels"], sigma=smoothing_sigma
            )

        # for some reason, units are um/100 in some datasets and um in others
        self.um_mult = um_mult

        # make a shadow background for plots
        self.create_shadows()

        # set the colormap for labels
        if label_cmap is None:
            self.label_cmap = label_cmap = plt.cm.tab20
            self.label_cmap.set_under(color=(0, 0, 0, 0))
        else:
            self.label_cmap = label_cmap

        # set the colormap for images
        if img_cmap is None:
            self.img_cmap = img_cmap = plt.cm.Greys
            self.img_cmap.set_under(color=(0, 0, 0, 0))
        else:
            self.img_cmap = img_cmap

        # unless the y sinus is updated from the original location (from the files), there is no transform
        self.y_sinus_um_transform = [0, 0, 0]

        # get the boundaries of voxel-space in um
        affine = self.voxel_data.loc["Brain", "affine"]
        voxels = self.voxel_data.loc["Brain", "voxels"]
        self.xmin, self.ymin, self.zmin = vox_to_um(
            np.array([0, 0, 0]), affine, self.um_mult, self.y_sinus_um_transform
        )
        self.xmax, self.ymax, self.zmax = vox_to_um(
            np.array(np.shape(voxels)) - 1,
            affine,
            self.um_mult,
            self.y_sinus_um_transform,
        )

        if updated_y_sinus is not None:
            self.update_y_sinus(updated_y_sinus)

        # voxels for individual regions ['region', 'type_', 'nucleus_ID', 'region_bounds', 'coords_vox', 'voxels']
        self.region_vox = utils.get_region_voxels(self)

        # determine where the brain exists (for plotting)
        self.determine_brain_limits()

        # get list of regions in atlas
        self.regions = {}
        for region in np.unique(self.brain_labels.type_):
            self.regions[region] = [
                [reg, region]
                for reg in self.brain_labels[
                    self.brain_labels.type_ == region
                ].region.values
            ]

        print("Atlas created")

    def determine_brain_limits(self):
        """only plot in regions where the brain exists
        """
        # get axis minima and maxima
        brain_label = "Brain"
        brain_vox = self.voxel_data.loc[brain_label, "voxels"]
        self.brain_limits = [
            np.where(brain_vox.sum(axis=1).sum(axis=1))[0][[0, -1]],
            np.where(brain_vox.sum(axis=0).sum(axis=1))[0][[0, -1]],
            np.where(brain_vox.sum(axis=0).sum(axis=0))[0][[0, -1]],
        ]

    def create_shadows(self):
        """ create backgrounds for visualizing transections
        """
        self.coronal_shadow = (
            np.rot90(np.sum(self.voxel_data.loc["Brain", "voxels"], axis=1) > 0)
            * np.shape(self.voxel_data.loc["Brain", "voxels"])[0]
        )
        self.transversal_shadow = (
            np.rot90(np.sum(self.voxel_data.loc["Brain", "voxels"], axis=2) > 0)
            * np.shape(self.voxel_data.loc["Brain", "voxels"])[1]
        )
        self.sagittal_shadow = (
            np.rot90(np.sum(self.voxel_data.loc["Brain", "voxels"], axis=0) > 0)
            * np.shape(self.voxel_data.loc["Brain", "voxels"])[2]
        )

    def update_y_sinus(self, updated_y_sinus):
        """update y sinus voxel location


        Arguments:
            updated_y_sinus {[list]} -- [list in voxels]
        """
        # update the y_sinus
        self.y_sinus_um_transform = np.array(updated_y_sinus)

        # get the boundaries of voxel-space in um
        affine = self.voxel_data.loc["Brain", "affine"]
        voxels = self.voxel_data.loc["Brain", "voxels"]
        self.xmin, self.ymin, self.zmin = vox_to_um(
            np.array([0, 0, 0]), affine, self.um_mult, self.y_sinus_um_transform
        )
        self.xmax, self.ymax, self.zmax = vox_to_um(
            np.array(np.shape(voxels)) - 1,
            affine,
            self.um_mult,
            self.y_sinus_um_transform,
        )
