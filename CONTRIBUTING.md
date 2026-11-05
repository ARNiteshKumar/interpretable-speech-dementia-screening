# Notes for running this project

## Data
The DementiaBank Pitt corpus is access-controlled. Request access at
<https://dementia.talkbank.org/>. Place the transcripts archive and the audio archive
somewhere readable and point `CFG.pitt_zip` / `CFG.audio_zip` at them.
**Never commit audio, transcripts, or any participant data** (`.gitignore` blocks them).

## Hardware
A CUDA GPU is required for the full run (Colab T4 is sufficient). On CPU, feature
extraction over 552 recordings is impractical — use `CFG.limit = 6` for a smoke-test.

## First run
1. `CFG.limit = 6` → confirms audio decodes, encoders load, CV executes (~2 min).
2. `CFG.limit = None` → full run. Extraction takes ~30–45 min and is cached/resumable.

## Audio format gotcha
Some Pitt "`.wav`" files are actually MP3 (ID3 header). The loader falls back
soundfile → ffmpeg → librosa, so both decode correctly. An integrity check stops the run
early with a clear message if any audio file is an empty placeholder.
