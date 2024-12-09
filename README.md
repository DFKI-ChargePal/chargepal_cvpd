# ChargePal Compute Vision Pattern Detectors (CVPD)

ChargePal-CVPD is a python package to detect 3D poses from RGB images using marker and geometric patterns.

## Installation for users

1) Clone the repository 
```shell
git clone git@github.com:DFKI-ChargePal/chargepal_cvpd.git
```
2) Navigate to the repository directory
```shell
cd chargepal_cvpd
 ```
3) Initialize [poetry](https://python-poetry.org/)
```shell
poetry install
```

## Getting started

This module depends on the [Camera Kit](https://github.com/DFKI-ChargePal/chargepal_camera_kit) package.
Before the Detector can be started, make sure that a calibrated camera has been set up. A detector is created 
using a configuration file describing the style of the marker and/or pattern. Examples can be found in the folder
[demos/dtt_config](demos/dtt_config). Afterward, similar to the [demo script](demos/find_pose.py), a detector 
can be initialized and executed. 
