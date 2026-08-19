# Public prior art and license routing

This project borrows architecture and evaluation ideas, not copied model weights,
datasets, or clinical report text. Verify current upstream terms before every
integration.

## Suitable references and optional adapters

| Project | Code license | What to borrow |
|---|---:|---|
| [MONAI](https://github.com/Project-MONAI/MONAI) | Apache-2.0 | transforms, metrics, bundle/provenance patterns |
| [MONAI Label](https://github.com/Project-MONAI/MONAILabel) | Apache-2.0 | reviewer correction and active-learning patterns |
| [MONAI Deploy](https://github.com/Project-MONAI/monai-deploy) | Apache-2.0 | provider-neutral imaging application interfaces |
| [OHIF](https://github.com/OHIF/Viewers) | MIT | DICOMweb and measurement/SR/SEG concepts; UI stays private |
| [Cornerstone3D](https://github.com/cornerstonejs/cornerstone3D) | MIT | volume and coordinate/annotation concepts |
| [pydicom](https://github.com/pydicom/pydicom) | MIT | optional DICOM parsing adapter |
| [highdicom](https://github.com/ImagingDataCommons/highdicom) | MIT | optional SR/SEG/SCOORD serialization after human approval |
| [TorchXRayVision](https://github.com/mlmed/torchxrayvision) | Apache-2.0 | research baseline, never ground truth |
| [MedSAM](https://github.com/bowang-lab/MedSAM) | Apache-2.0 | prompted reviewer/detector ROI segmentation |
| [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) | Apache-2.0 | task-specific supervised baseline |
| [MedRAX](https://github.com/bowang-lab/MedRAX) | Apache-2.0 code | selective expert routing and task taxonomy |
| [MONAI VLM Radiology Agent Framework](https://github.com/Project-MONAI/VLM-Radiology-Agent-Framework) | Apache-2.0 code | 2D/3D/4D expert feedback architecture |
| [RadGraph](https://github.com/Stanford-AIMI/radgraph) | MIT | entity/relation/negation evaluation |
| [RadFact](https://github.com/microsoft/RadFact) | MIT | atomic logical and spatial claim evaluation |
| [MedPerf](https://github.com/mlcommons/medperf) | Apache-2.0 | future federated evaluation |
| [ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit) | BSD-3-Clause | synthetic ECG image degradation fixtures |
| [ECG-Digitiser](https://github.com/felixkrones/ECG-Digitiser) | BSD-2-Clause | digitization-QC architecture |
| [promptfoo](https://github.com/promptfoo/promptfoo) | MIT | isolated cross-provider regression/red-team runner |

## Keep outside the distributable core

- CheXbert has an academic/non-commercial license; do not vendor its code or
  checkpoint into a commercializable core.
- RadCliQ depends on components with different licenses; do not make it a mandatory
  release gate.
- VILA-M3 and MAIRA-2 weights carry non-commercial/research restrictions distinct
  from their surrounding code.
- CT-CHAT/CT-RATE are non-commercial; keep them in explicitly isolated research
  adapters, not a product dependency.
- TotalSegmentator task/weight licenses vary; review each task rather than relying on
  the repository license badge.
- MIMIC-CXR and other credentialed datasets remain local under their DUA.
- An absent or unclear dataset/model license means no redistribution.

Code, model weights, datasets, prompt/report text, and documentation are separate
rights layers. A permissive code license does not relicense downloaded assets.

## Governance references

- [DICOM PS3.15 confidentiality profiles](https://dicom.nema.org/medical/dicom/current/output/chtml/part15/PS3.15.html)
- [FDA Good Machine Learning Practice principles](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles)
- [FDA transparency principles](https://www.fda.gov/medical-devices/software-medical-device-samd/transparency-machine-learning-enabled-medical-devices-guiding-principles)

These references inform engineering checks; they are not a declaration of legal or
regulatory conformity.
