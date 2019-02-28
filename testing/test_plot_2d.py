from birdbrain.atlas import atlas
import pytest
from birdbrain.visualization.plotting_2d import plot_transection, plot_2d_coordinates, make_label_data


@pytest.fixture(scope="module")
def starling_atlas():
    dset_dir = "../../data/processed/starling/"
    return atlas(
        species="starling",
        dset_dir=dset_dir,
        um_mult=100,
        smoothing=[],  # ['Brain', 'Brainregions']
    )


def test_plot_2d_coordinates(starling_atlas):
    plot_2d_coordinates(starling_atlas, point_in_um=[0, 0, 0])


def old_make_label_data(atlas, regions_to_plot, point_in_voxels):
    label_data = {
        r2p: {"voxels": atlas.voxel_data.loc[r2p, "voxels"]} for r2p in regions_to_plot
    }
    # get label data
    for r2p in regions_to_plot:
        x_lab = label_data[r2p]["voxels"][point_in_voxels[0], :, :].squeeze()
        y_lab = label_data[r2p]["voxels"][:, point_in_voxels[1], :].squeeze()
        z_lab = label_data[r2p]["voxels"][:, :, point_in_voxels[2]].squeeze()

        # subset labels dataframe for only the ones appearing here
        unique_labels = np.unique(
            list(x_lab.flatten()) + list(y_lab.flatten()) + list(z_lab.flatten())
        )

        label_data[r2p]["x_lab"] = x_lab
        label_data[r2p]["y_lab"] = y_lab
        label_data[r2p]["z_lab"] = z_lab
        label_data[r2p]["unique_labels"] = unique_labels

    # subset brain_labels dataframe for only the labels shown here
    regions_plotted = pd.concat(
        [
            atlas.brain_labels[
                (atlas.brain_labels.type_ == r2p)
                & (
                    [
                        label in label_data[r2p]["unique_labels"]
                        for label in atlas.brain_labels.label.values
                    ]
                )
            ]
            for r2p in regions_to_plot
        ]
    )
    # this doesn't seem like its ever used...
    # regions_plotted.index = np.arange(len(regions_plotted))
    return label_data, regions_plotted


@pytest.mark.parametrize("regions_to_plot", [["Brainregions"], ["Brainregions", "Nuclei"]])
def test_make_label_data(starling_atlas, regions_to_plot):
    point_in_voxels = [0, 1500, -200]
    label_data, regions_plotted = make_label_data(starling_atlas, regions_to_plot, point_in_voxels)
    old_label_data, old_regions_plotted = old_make_label_data(
        starling_atlas, regions_to_plot, point_in_voxels)
    assert len(regions_plotted) == len(old_regions_plotted)
    assert len(label_data) == len(old_label_data)
    for r2p in label_data:
        assert np.all(label_data[r2p]['unique_labels'] == old_label_data[r2p]['unique_labels'])
