# ASTRA-DF: Is Your Multimodal Deepfake Detector Actually Multimodal?

Code, trained models and per-clip results for:

> A. Gawali, A. Gupta, A. Indulkar, A. Gondavale, P. Mishra, S. Okuhara,
> **"Is Your Multimodal Deepfake Detector Actually Multimodal? A Modality Ablation Study,"**
> IEICE Transactions on Information and Systems, Special Section on KICSS 2026 (accepted, to appear).

All authors contributed equally to this work.

ASTRA-DF is a compact (7.11 M-parameter) audio–visual deepfake detector. It uses:
- a mouth-region MobileNetV3 encoder and an MFCC 1-D CNN encoder;
- Transformer or BiLSTM temporal backbones;
- a bidirectional cross-modal attention bridge;
- clip, frame and interval heads.

The paper uses it to test whether such a detector actually uses both modalities. It finds that, at this scale and at the mouth resolution of current benchmarks, decisions are dominated by audio. The evidence comes from four kinds of test:
- five-seed ablations;
- test-time removal of the video;
- modality-separated zero-shot evaluation;
- an attention temporal-alignment diagnostic.

This repository contains everything needed to reproduce every number in the revised paper.

## Repository layout

```
code/
  astra_df/                  model building blocks (encoders, temporal backbones, cross-modal bridge, heads)
  verify_setup.py            environment and model smoke check
  run_experiments.py         dataset (PackedLAVDF), model variants (VariantASTRA), training stages
  eval_suite.py              evaluation only: counterfactuals, zero-shot corpora, attention statistics
  analyse_suite.py           turns raw per-clip outputs into all reported statistics (LaTeX macros)
  prepare_subset.py          LAV-DF subset -> packed cache
  prepare_fakeavceleb.py     FakeAVCeleb -> packed cache
  prepare_timit.py           DeepfakeTIMIT + VidTIMIT -> packed cache (use --shared-audio)
  check_splits.py            source-video independence of the train/val/test partitions
  analyse_lavdf_audio.py     audio re-timing in LAV-DF "video-only" fakes
  bench_latency.py           latency with recorded hardware
  make_attention_figure.py   real attention maps (Fig. 3)
  make_counterfactual_figure.py   test-time input changes (Fig. 2)
  make_pipeline_figure.py    pipeline diagram (Fig. 1)
  measure_roi.py             native mouth-region size
  measure_mouth_band.py      height of the mouth band inside the letterboxed crops
  run_revision.sh            full training + evaluation queue (all variants, all seeds)
  run_randctrl.sh            random-initialisation capacity control for the lipreading encoder
results/
  revision/                  results of the revised paper
    results_rev.json         training records: config, metrics, per-clip test scores, every run
    eval_suite.json          per-clip scores and attention statistics for every evaluation
    *.json                   split check, latency, LAV-DF audio check, ROI sizes, mouth-band height, summary
  submitted_version/         results of the originally submitted manuscript, for reference
```

Training writes to `results/rev/`; `results/revision/` holds the outputs used in the paper.

## Setup

Two Python environments are used:

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt        # training / evaluation / analysis
python3.12 -m venv .venv-prep && .venv-prep/bin/pip install -r requirements-prep.txt   # cache building only
```

The prep environment needs `ffmpeg` on `PATH`. `imageio-ffmpeg` ships a static binary that you can link into `.venv-prep/bin/ffmpeg`.

Then run a quick check:

```bash
cd code && ../.venv/bin/python verify_setup.py
```

All results in the paper were produced with these exact versions, on an Apple M3 (8 GB, PyTorch MPS back-end). On 8 GB of unified memory a 32-clip batch does not fit, so training uses gradient accumulation:
- `--accum 2` (2 × 16) for MobileNet and audio variants;
- `--accum 4` (4 × 8) for the lipreading encoder.

The effective batch is 32 in every case.

## Data and models

The datasets are **not** redistributed here. Obtain them from their providers:

| Dataset | Use | Source |
|---|---|---|
| LAV-DF | training and in-domain test (stratified 902-video subset) | https://github.com/ControlNet/LAV-DF |
| FakeAVCeleb v1.2 | zero-shot, modality-separated categories | https://github.com/DASH-Lab/FakeAVCeleb |
| DeepfakeTIMIT | zero-shot face swaps with unmodified audio | https://www.idiap.ch/en/scientific-research/data/deepfaketimit |
| VidTIMIT | genuine counterparts for DeepfakeTIMIT | https://zenodo.org/records/158963 |

Place the inputs in `assets/` next to `code/`:
- the built caches: `assets/astra_cache`, `assets/favc_cache`, `assets/timit_cache`;
- the LAV-DF `metadata.json`, as `assets/lavdf_metadata.json`.

**Trained checkpoints.** All 44 checkpoints used in the paper are attached to the [`v1.0` release](../../releases/tag/v1.0):
- 7 variants × seeds 42–46;
- lipreading-encoder models (pretrained and random-init) × seeds 42–44.

Download them into `results/rev/checkpoints/` (the path `run_revision.sh` writes to) to run any evaluation without retraining.

**LRW lipreading weights** (for the lipreading-encoder experiment). Use the `resnet18_mstcn_video` model from [mpc001/Lipreading_using_Temporal_Convolutional_Networks](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks), saved as `assets/lrw_resnet18.pth`. `LipreadEncoder` reproduces that checkpoint's architecture exactly: Swish activations, grey-scale input, 88 × 88 centre crop. It refuses checkpoints that do not fully match.

## Reproducing the paper

```bash
cd code
# 1. build caches (prep environment)
PATH="../.venv-prep/bin:$PATH" ../.venv-prep/bin/python prepare_subset.py --root /path/to/LAV-DF --out ../assets/astra_cache --n-videos 900   # rounding per category gives the 902 videos used
PATH="../.venv-prep/bin:$PATH" ../.venv-prep/bin/python prepare_fakeavceleb.py --root /path/to/FakeAVCeleb_v1.2 --out ../assets/favc_cache
PATH="../.venv-prep/bin:$PATH" ../.venv-prep/bin/python prepare_timit.py --deepfake-timit /path/to/DeepfakeTIMIT \
    --vidtimit /path/to/VidTIMIT --out ../assets/timit_cache --shared-audio

# 2. train and evaluate every variant over all seeds (resumable)
nohup ./run_revision.sh > ../results/rev/queue.log 2>&1 &
./run_randctrl.sh                     # capacity control

# 3. statistics and checks
../.venv/bin/python check_splits.py --metadata ../assets/lavdf_metadata.json --cache ../assets/astra_cache
PATH="../.venv-prep/bin:$PATH" ../.venv/bin/python analyse_lavdf_audio.py --lavdf-root /path/to/LAV-DF --cache ../assets/astra_cache
../.venv/bin/python analyse_suite.py --lavdf ../assets/astra_cache --favc ../assets/favc_cache --dftimit ../assets/timit_cache
../.venv/bin/python measure_mouth_band.py ../assets/astra_cache ../assets/favc_cache --out ../results/rev/mouth_band.json
```

`analyse_suite.py` writes every reported number as a LaTeX macro and asserts that the per-clip labels it reconstructs match the stored ones before computing anything.

### Recomputing the statistics without retraining

`results/revision/` holds the per-clip outputs of every run. Once the three caches exist (step 1; only their `videos.csv` files are read), every statistic in the paper can be regenerated on a CPU in a few minutes:

```bash
cd code
../.venv/bin/python analyse_suite.py \
    --suite ../results/revision/eval_suite.json --results ../results/revision/results_rev.json \
    --split-check ../results/revision/split_check.json --latency ../results/revision/latency.json \
    --lavdf-audio ../results/revision/lavdf_audio_check.json \
    --out-tex ../results/results_suite.tex --out-json ../results/suite_summary.json \
    --lavdf ../assets/astra_cache --favc ../assets/favc_cache --dftimit ../assets/timit_cache
../.venv/bin/python make_counterfactual_figure.py --suite ../results/revision/eval_suite.json \
    --lavdf ../assets/astra_cache --out ../results/figures/counterfactual.pdf      # Fig. 2
```

Paired differences between the full and audio-only models are reported as audio-only minus full.

## Protocol notes

- **Threshold.** The decision threshold is fixed at τ = 0.5 for every experiment and is never tuned. FakeAVCeleb, DeepfakeTIMIT and VidTIMIT are evaluated strictly zero-shot.
- **DeepfakeTIMIT.** Each fake keeps the audio of its genuine VidTIMIT original. `--shared-audio` gives both the byte-identical 16 kHz recording, so any real/fake separation must come from the video. Without it, the different decoding paths leak (audio-only AUC 0.59 instead of 0.50).
- **Mouth crops.** The padded lip box is resized with its aspect ratio kept onto a black square canvas (letterboxing). At the model's 96-pixel input the mouth fills a band of about 96 × 35 pixels (`measure_mouth_band.py`).
- **LAV-DF "video-only" fakes.** Their real audio is spliced and re-timed at the manipulated word: it matches the source video before the word, and after it continues delayed by a median of 100 ms (`analyse_lavdf_audio.py`). This is why an audio-only model detects them.

## Licence

Code: MIT (see `LICENSE`). Datasets and the LRW weights remain under their own licences.
