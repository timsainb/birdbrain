[![Build Status](https://travis-ci.org/timsainb/birdbrain.svg?branch=master)](https://travis-ci.org/timsainb/birdbrain)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/timsainb/birdbrain/master?filepath=Index.ipynb)


birdbrain
==============================

Tim Sainburg & Marvin Thielk

This is a small library for viewing the [songbird/bat brain atlas'](https://www.uantwerpen.be/en/research-groups/bio-imaging-lab/research/mri-atlases/starling-brain-atlas/) (Poirier et al., 2008; De Groof et al., 2016; Vellema et al., 2011; Gunturken et al., 2013; Washington et al., 2018). It currently has examples of European starling, Canary, Zebra finch, Mustached bat, and Pigeon brain atlas'. 

The package can do things like:

- nuclei localization relative to a set stereotaxic reference point (e.g. y-sinus in starlings)
- 3d printing an STL of the brains
- Plotting recording locations in 2d and 3d on imaging data. 
- Creating visualizations / movies videos of nuclei of interest

There is an [online interactive demo](https://mybinder.org/v2/gh/timsainb/birdbrain/master?filepath=Index.ipynb) which should take no Python experience to use (just running cells in a Jupyter notebook). The demo uses Binder, which is a allows you to run a Jupyter notebook in a Docker environment online. It can take a bit to load, but has the benefit of not requiring you to install anything. If you want to install this software locally, the package is pip installable however. 

![screenshot](assets/img/3d_screenshot.png)

![field_l](assets/img/field_l.png)

### Online demo!
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/timsainb/birdbrain/master?filepath=Index.ipynb)

### Usage Instructions
You can either view the data directly from the [binder notebooks](https://mybinder.org/v2/gh/timsainb/birdbrain/master?filepath=Index.ipynb) via your internet browser (reccomeded at first), or you can install and run this package locally on your own computer. 

### Installation
To install the python package:

`pip install birdbrain`

##### Additional requirements
To be added...

### Citations

If you use this data, please cite the respecitve atlas papers:

**Zebra finch**

```
@article{poirier2008three,
  title={A three-dimensional MRI atlas of the zebra finch brain in stereotaxic coordinates},
  author={Poirier, Colline and Vellema, Michiel and Verhoye, Marleen and Van Meir, Vincent and Wild, J Martin and Balthazart, Jacques and Van Der Linden, Annemie},
  journal={Neuroimage},
  volume={41},
  number={1},
  pages={1--6},
  year={2008},
  publisher={Elsevier}
}
```

**European starling**

```
@article{de2016three,
  title={A three-dimensional digital atlas of the starling brain},
  author={De Groof, Geert and George, Isabelle and Touj, Sara and Stacho, Martin and Jonckers, Elisabeth and Cousillas, Hugo and Hausberger, Martine and G{\"u}nt{\"u}rk{\"u}n, Onur and Van der Linden, Annemie},
  journal={Brain Structure and Function},
  volume={221},
  number={4},
  pages={1899--1909},
  year={2016},
  publisher={Springer}
}
```


**Canary**

```
@article{vellema2011customizable,
  title={A customizable 3-dimensional digital atlas of the canary brain in multiple modalities},
  author={Vellema, Michiel and Verschueren, Jacob and Van Meir, Vincent and Van der Linden, Annemie},
  journal={Neuroimage},
  volume={57},
  number={2},
  pages={352--361},
  year={2011},
  publisher={Elsevier}
}

```

**Pigeon**

```
@article{gunturkun20133,
  title={A 3-dimensional digital atlas of the ascending sensory and the descending motor systems in the pigeon brain},
  author={G{\"u}nt{\"u}rk{\"u}n, Onur and Verhoye, Marleen and De Groof, Geert and Van der Linden, Annemie},
  journal={Brain Structure and Function},
  volume={218},
  number={1},
  pages={269--281},
  year={2013},
  publisher={Springer}
}

```

**Mustached bat**

```
@article{washington2018three,
  title={A three-dimensional digital neurological atlas of the mustached bat (Pteronotus parnellii)},
  author={Washington, Stuart D and Hamaide, Julie and Jeurissen, Ben and Van Steenkiste, Gwendolyn and Huysmans, Toon and Sijbers, Jan and Deleye, Steven and Kanwal, Jagmeet S and De Groof, Geert and Liang, Sayuan and others},
  journal={NeuroImage},
  volume={183},
  pages={300--313},
  year={2018},
  publisher={Elsevier}
}

```

#### References
- [Brain atlas](https://www.uantwerpen.be/en/research-groups/bio-imaging-lab/research/mri-atlases/starling-brain-atlas/) for starling, canary, zebra finch, pigeon, tilapia, and mustached bat brain atlas'
- [VTK python](https://pypi.org/project/vtk/) for 3d graphics 
- [K3D tools](https://github.com/K3D-tools/K3D-jupyter) for 3d visualization
- [nibabel](http://nipy.org/nibabel/) For reading/manipulating neuroimaging data (.img files)
- [cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science/) project template was used


#### TODO:
  - update to use high resolution T2 images rather than same resolution as delineations
  - Embed the javascript directly (e.g. save 3d to be embedded into website) (this won't allow for selecting regions though)

