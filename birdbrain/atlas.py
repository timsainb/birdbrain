import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
from birdbrain import utils
from birdbrain.utils import vox_to_um, inverse_dict
import pandas as pd
import birdbrain.downloading as dl


class atlas(object):
    def __init__(
        self,
        dset_dir,
        label_cmap=None,
        um_mult=100,
        img_cmap=None,
        smoothing=[],
        smoothing_sigma=2,
        updated_y_sinus=None,
        species=None,
        password=None,
    ):

        # path of delineations
        delin_path = os.path.join(os.path.abspath(dset_dir), "delineations/")

        if species == "canary":
            self.brain_labels = pd.read_csv(
                "../../assets/csv/canary_regions.csv", index_col="region"
            )
            self.brain_labels.columns = ["label", "region", "type_"]
            dl.get_canary_data()

        elif species == "starling":
            dl.get_starling_data()
            # path of labels
            text_files = glob(os.path.join(delin_path, "*.txt"))
            # transcription labels ['label', 'region', 'type_']
            if len(text_files) > 0:
                self.brain_labels = utils.get_brain_labels(text_files)

        elif species == "zebra_finch":
            self.brain_labels = pd.read_csv(
                "../../assets/csv/zebra_finch_regions.csv", index_col="region"
            )
            self.brain_labels.columns = ["label", "region", "type_"]
            dl.get_zebra_finch_data(password)

        elif species == "pigeon":

            self.brain_labels = pd.read_csv(
                "../../assets/csv/pigeon_regions.csv", index_col="region"
            )
            self.brain_labels.columns = ["label", "region", "type_"]
            self.systems_delineations = dl.get_pigeon_data()


        elif species == "mustached_bat":
            raise NotImplementedError("TODO")

        # how axes labels relate to affine transformed data in voxels
        self.axes_dict = {
            "medial-lateral": 0,
            "anterior-posterior": 1,
            "dorsal-ventral": 2,
        }
        self.inverse_axes_dict = inverse_dict(self.axes_dict)

        # path of images
        img_files = glob(os.path.join(delin_path, "*.img")) + glob(
            os.path.join(dset_dir, "*.img")
        )

        # images from each type of scan, as well as transcribed locations ['type_', 'src', 'voxels']
        self.voxel_data = utils.get_voxel_data(img_files)


        if species == 'pigeon':
            dl.join_data_pigeon(self)

        # smooth the whole brain because the atlas is a bit noisy
        for img in smoothing:
            self.voxel_data.loc[img, "voxels"] = utils.smooth_voxels(
                self.voxel_data.loc[img, "voxels"], sigma=smoothing_sigma
            )

        # for some reason, units are um/100 in this dataset
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
