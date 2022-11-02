# scan20-docker
Docker for the Brats2020 scan20 submission

This is an adjusted version of the Docker currently available for the Brats2020 scan20 submission.
The original docker can be found on the Brats docker page: <https://hub.docker.com/r/brats/scan-20>

The original docker is used in the BraTS-Toolkit: <https://github.com/neuronflow/BraTS-Toolkit>

## Changes
This version of the docker has a few changes compared to the original docker:

- Packages updated to Pytorch 1.11 and cuda 11.3 to work with newer GPUs.
- Option to provide an input path where the files are located instead of a fixed path
- The input files are renamed:

| Scan type  | Original filename  |  New filename |
|---|---|---|
| pre-contrast T1  | t1.nii.gz  | T1.nii.gz  |
| post-contrast T1  | t1ce.nii.gz  |  T1GD.nii.gz  |
| T2 | t2.nii.gz | T2.nii.gz |
| T2w-FLAIR | flair.nii.gz | FLAIR.nii.gz |
