"""
Prototype code for Story 2.4 (formerly 2.3): Musical Key and Mood Detection
This is reference implementation for research spike - requires validation in Story 2.2.

Dependencies:
- essentia
- tensorflow (implicitly required by essentia models)
- numpy

Note: This code is part of research spike Story 2.2 to validate approach.
"""

from dataclasses import dataclass
from json import dump, load
from os import environ
from pathlib import Path
from sys import argv
from typing import Any

import essentia
import numpy as np
from essentia.standard import (
    MonoLoader,
    TensorflowPredictMusiCNN,
    TensorflowPredictVGGish,
)

environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
essentia.log.infoActive = False
essentia.log.warningActive = False

MODELS_DIRECTORY = "./models"


class Model:
    def __init__(self, type: str, model: str) -> None:
        self.__type = type
        self.__metadata_file = model + ".json"
        self.__model_file = model + ".pb"
        self.__classifier = None

        with Path(MODELS_DIRECTORY, self.__metadata_file).open() as metadata_file:
            self.__metadata = load(metadata_file)

        self.__labels = list(map(self.__process_labels, self.__metadata["classes"]))

    def __process_labels(self, label: str) -> str:
        return label.replace("---", "/")

    def get_type(self) -> str:
        return self.__type

    type = property(get_type)

    def get_labels(self) -> list[str]:
        return self.__labels

    labels = property(get_labels)

    def get_metadata(self) -> dict[str, Any]:
        return dict(self.__metadata)

    metadata = property(get_metadata)

    def get_model(self) -> Any:
        return self.__model_file

    model = property(get_model)

    def get_classifier(self) -> Any:
        if self.__classifier is not None:
            return self.__classifier

        if "musicnn" in self.__model_file.lower():
            self.__classifier = TensorflowPredictMusiCNN(graphFilename=str(Path(MODELS_DIRECTORY) / self.__model_file))
        elif "vggish" in self.__model_file.lower():
            self.__classifier = TensorflowPredictVGGish(graphFilename=str(Path(MODELS_DIRECTORY) / self.__model_file))
        else:
            raise Exception("Unknown classifier.")

        return self.__classifier

    classifier = property(get_classifier)


@dataclass
class ModelSet:
    name: str
    models: list[Model]


class MusicAttributePredictor:
    def __init__(self, filename: str, sample_rate: int = 16000) -> None:
        self.__sample_rate = sample_rate
        self.__audio = MonoLoader(filename=filename, sampleRate=self.__sample_rate)()

    def __enter__(self) -> "MusicAttributePredictor":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.__audio = None
        return True

    def __predict_single_model(self, model: Model) -> Any:
        activations = model.classifier(self.__audio)
        return np.mean(activations, axis=0)

    def predict(self, model_set: ModelSet) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for model in model_set.models:
            predictions = self.__predict_single_model(model).tolist()
            results[model.type] = []
            for i, label in enumerate(model.labels):
                results[model.type].append({"label": label, "prediction": predictions[i]})

        return results


# Model sets for different mood attributes
model_sets = [
    ModelSet(
        "mood_acoustic",
        [
            Model("MusiCNN MSD", "mood_acoustic-musicnn-msd-2"),
            Model("MusiCNN MTT", "mood_acoustic-musicnn-mtt-2"),
            Model("VGGish", "mood_acoustic-vggish-audioset-1"),
        ],
    ),
    ModelSet(
        "mood_electronic",
        [
            Model("MusiCNN MSD", "mood_electronic-musicnn-msd-2"),
            Model("MusiCNN MTT", "mood_electronic-musicnn-mtt-2"),
            Model("VGGish", "mood_electronic-vggish-audioset-1"),
        ],
    ),
    ModelSet(
        "mood_aggressive",
        [
            Model("MusiCNN MSD", "mood_aggressive-musicnn-msd-2"),
            Model("MusiCNN MTT", "mood_aggressive-musicnn-mtt-2"),
            Model("VGGish", "mood_aggressive-vggish-audioset-1"),
        ],
    ),
    ModelSet(
        "mood_relaxed",
        [
            Model("MusiCNN MSD", "mood_relaxed-musicnn-msd-2"),
            Model("MusiCNN MTT", "mood_relaxed-musicnn-mtt-2"),
            Model("VGGish", "mood_relaxed-vggish-audioset-1"),
        ],
    ),
    ModelSet(
        "mood_happy",
        [
            Model("MusiCNN MSD", "mood_happy-musicnn-msd-2"),
            Model("MusiCNN MTT", "mood_happy-musicnn-mtt-2"),
            Model("VGGish", "mood_happy-vggish-audioset-1"),
        ],
    ),
    ModelSet(
        "mood_sad",
        [
            Model("MusiCNN MSD", "mood_sad-musicnn-msd-2"),
            Model("MusiCNN MTT", "mood_sad-musicnn-mtt-2"),
            Model("VGGish", "mood_sad-vggish-audioset-1"),
        ],
    ),
    ModelSet(
        "Mood Party",
        [
            Model("MusiCNN MSD", "mood_party-musicnn-msd-2"),
            Model("MusiCNN MTT", "mood_party-musicnn-mtt-2"),
            Model("VGGish", "mood_party-vggish-audioset-1"),
        ],
    ),
    ModelSet(
        "danceability",
        [
            Model("MusiCNN MSD", "danceability-musicnn-msd-2"),
            Model("MusiCNN MTT", "danceability-musicnn-mtt-2"),
            Model("VGGish", "danceability-vggish-audioset-1"),
        ],
    ),
    ModelSet(
        "gender",
        [
            Model("MusiCNN MSD", "gender-musicnn-msd-2"),
            Model("MusiCNN MTT", "gender-musicnn-mtt-2"),
            Model("VGGish", "gender-vggish-audioset-1"),
        ],
    ),
    ModelSet(
        "tonality",
        [
            Model("MusiCNN MSD", "tonal_atonal-musicnn-msd-2"),
            Model("MusiCNN MTT", "tonal_atonal-musicnn-mtt-2"),
            Model("VGGish", "tonal_atonal-vggish-audioset-1"),
        ],
    ),
    ModelSet(
        "voice_instrumental",
        [
            Model("MusiCNN MSD", "voice_instrumental-musicnn-msd-1"),
            Model("MusiCNN MTT", "voice_instrumental-musicnn-mtt-2"),
            Model("VGGish", "voice_instrumental-vggish-audioset-1"),
        ],
    ),
]

# Example usage
if __name__ == "__main__":
    results = {}
    with MusicAttributePredictor(argv[1]) as p:
        for model_set in model_sets:
            results[model_set.name] = p.predict(model_set)

    with Path("predictions.json").open("w") as f:
        dump(results, f, indent=2, sort_keys=True)
