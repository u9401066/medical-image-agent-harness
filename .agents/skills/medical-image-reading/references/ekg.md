# EKG/ECG image protocol

## Signal-image quality gate

Before interpreting morphology, establish the visible lead labels and inventory,
layout, trace direction, grid legibility, calibration pulse, paper speed, gain,
artifact, clipping, and rhythm-strip duration. Never infer a lead name from a fixed
template when its printed label is unreadable.

If speed/gain/grid or waveform digitization quality is inadequate, do not report
numeric rate, interval, voltage, axis, or millimetre measurements. A screenshot crop
is not a calibrated raw waveform.

## Lead-conditioned blind pass

Required axes are `heart_rate`, `rhythm`, `regularity`, `axis`, `p_wave`,
`pr_interval`, `qrs_duration`, `qrs_morphology`, `st_segment`, `t_wave`,
`qtc_interval`, `chamber_enlargement`, `conduction`, `av_block`, `stemi_pattern`,
and `ischemia`.

Assess only axes supported by the visible leads:

- a lone rhythm strip supports rhythm/rate/regularity/ectopy, not territory, axis,
  R-wave progression, or chamber-voltage conclusions;
- territory claims require the relevant contiguous leads and should identify which
  labels are actually visible;
- an unlabeled panel stays `unknown`; do not guess;
- normal/negative axes belong in the checklist, with no boxes.

Localize an actionable morphology to a short representative time segment within
the declared lead region. Preserve source-image coordinates and avoid whole-row
boxes. If a candidate is plausible but unresolved, use low certainty plus a concrete
review question. Acute ischemic/injury, dangerous tachyarrhythmia, high-grade block,
or severe electrolyte-pattern candidates require urgent review while retaining the
distinction between pattern and confirmed etiology.

## Optional waveform evidence

Use a waveform model only when a trusted host supplies a matched artifact ID, lead
mode, calibration/digitization-quality result, and evidence nonce. Inspect the image
first. Treat uncalibrated probabilities as supporting evidence only, reconcile
disagreement explicitly, and never use waveform-model output as spatial evidence.
