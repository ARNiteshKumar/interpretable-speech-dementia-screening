"""Central configuration for the interpretable speech-screening pipeline."""
from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class Config:
    # ---- data ----
    pitt_zip: Optional[str] = None      # transcripts (.cha) archive; None = auto-search
    audio_zip: Optional[str] = None     # audio archive; None = auto-search
    limit: Optional[int] = None         # None = all 552 recordings; 6 = quick smoke-test
    min_seg_s: float = 0.30             # drop patient segments shorter than this

    # ---- encoders (frozen) ----
    whisper_model: str = "large-v3-turbo"
    asr_batch: int = 16
    acoustic_model: str = "microsoft/wavlm-base-plus"
    text_model: str = "roberta-base"
    max_utts: int = 40

    # ---- fusion / bottleneck ----
    d_model: int = 128
    heads: int = 4
    dropout: float = 0.3
    lambda_concept: float = 5.0         # weight of the concept-supervision (MSE) term

    # ---- optimisation ----
    epochs: int = 80
    patience: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16

    # ---- evaluation ----
    kfolds: int = 5
    repeats: int = 3                    # speaker-independent repeated stratified GROUP k-fold
    threshold_mode: str = "youden"
    sens_floor: float = 0.85
    cutoffs_s: List[int] = field(default_factory=lambda: [10, 20, 30, 45, 60, 90, 999])
    knee_tol: float = 0.02

    # ---- paths ----
    work: str = "/content/work" if os.path.isdir("/content") else "/tmp/work"

    def __post_init__(self):
        self.feat_cache = os.path.join(self.work, "feat")
        self.text_cache = os.path.join(self.work, "txt")
        self.audio_cache = os.path.join(self.work, "aud")
        for d in (self.work, self.feat_cache, self.text_cache, self.audio_cache):
            os.makedirs(d, exist_ok=True)


CFG = Config()
