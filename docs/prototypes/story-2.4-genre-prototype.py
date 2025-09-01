"""
Prototype code for Story 2.4: Genre Detection using Discogs EffNet
This is reference implementation for research spike Story 2.2.

Dependencies:
- essentia
- tensorflow (implicitly required by essentia models)
- numpy

Note: This code is part of research spike Story 2.2 to validate approach.
"""

from itertools import chain
from json import load
from os import environ
from pathlib import Path
from sys import argv
from typing import Any

import essentia
import essentia.standard as es
import numpy as np

environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
essentia.log.infoActive = False
essentia.log.warningActive = False


def process_labels(label: str) -> str:
    return label.replace("---", "/")


def get_genres_per_minute(filename: str) -> dict[str, Any]:
    top_n = 5
    json_file = "discogs/discogs-effnet-bs64-1.json"
    model_file = "discogs/discogs-effnet-bs64-1.pb"

    with Path(json_file).open() as f:
        metadata = load(f)

    sample_rate = metadata["inference"]["sample_rate"]
    labels = list(map(process_labels, metadata["classes"]))

    classifier = es.TensorflowPredictEffnetDiscogs(graphFilename=model_file)

    audio = es.MonoLoader(filename=filename, sampleRate=sample_rate)()

    activations = classifier(audio)
    averaged_predictions = np.mean(activations, axis=0)
    top_n_predictions = np.argsort(averaged_predictions)[::-1][:top_n]

    result = {
        "label": list(chain(*[[labels[idx]] * activations.shape[0] for idx in top_n_predictions])),
        "activation": list(chain(*[activations[:, idx] for idx in top_n_predictions])),
    }

    print(f"File                         : {filename}")
    print(f"Sample Rate                  : {sample_rate / 1000} kHz")
    print(f"Genre/Style (unique)         : {list(set(result['label']))}")
    print(f"Activation Energy (first {top_n * 3}) : {result['activation'][: top_n * 3]}")

    return result


if __name__ == "__main__":
    get_genres_per_minute(argv[1])
