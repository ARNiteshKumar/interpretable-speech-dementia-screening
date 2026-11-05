"""The 24 interpretable clinical concepts (speech + language markers).

Each concept is a real, measurable quantity. During training the bottleneck is
supervised against these measured values, which is what makes the resulting
explanation faithful rather than merely plausible.
"""

LINGUISTIC = [
    "ttr",              # type-token ratio (lexical diversity)
    "mattr",            # moving-average TTR (length-robust lexical diversity)
    "mlu_words",        # mean length of utterance, in words
    "pronoun_ratio",    # pronouns / tokens  (word-finding difficulty -> up)
    "noun_ratio",       # nouns / tokens
    "verb_ratio",       # verbs / tokens
    "pronoun_noun",     # pronoun-to-noun ratio ("empty speech" indicator)
    "noun_verb",        # noun-to-verb ratio
    "lex_density",      # content words / tokens
    "speech_rate",      # words per minute
    "mean_pause",       # mean inter-word pause (ms)
    "long_pause",       # fraction of pauses over a long-pause threshold
    "filled_pause",     # filled-pause rate, measured ACOUSTICALLY (ASR-independent)
]

ACOUSTIC = [
    "f0_mean", "f0_std", "f0_range",   # pitch level and variation (monotonicity -> down)
    "jitter", "shimmer", "hnr",        # voice quality / stability
    "spec_centroid", "spec_bw",        # spectral shape
    "zcr", "rms",                      # zero-crossing rate, loudness
]

CONCEPTS = LINGUISTIC + ACOUSTIC          # 23 named markers + bias term in the head
N_CONCEPTS = len(CONCEPTS)

# Clinically expected direction (+1 = higher value indicates impairment)
EXPECTED_DIRECTION = {
    "pronoun_ratio": +1, "pronoun_noun": +1, "mean_pause": +1, "long_pause": +1,
    "filled_pause": +1, "jitter": +1, "shimmer": +1,
    "ttr": -1, "mattr": -1, "mlu_words": -1, "lex_density": -1,
    "speech_rate": -1, "noun_ratio": -1, "noun_verb": -1, "f0_std": -1,
    "f0_range": -1, "hnr": -1,
}
