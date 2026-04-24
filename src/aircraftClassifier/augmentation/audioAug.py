"""
Audiomentations-based augmentation pipeline.

CPU-side augmentation applied to raw waveforms (numpy arrays) before
spectrogram conversion. Pass a folder of background noise recordings
(no aircraft present) to AddBackgroundNoise — this is the highest-impact
augmentation for this task.
"""

from audiomentations import (
    Compose,
    AddGaussianNoise,
    AddBackgroundNoise,
    TimeStretch,
    PitchShift,
    Shift,
    Gain,
    ApplyImpulseResponse,
)


def buildAugPipeline(bgNoiseDir: str | None = None) -> Compose:
    """
    Build the standard augmentation pipeline.

    Args:
        bgNoiseDir: Path to a folder of background noise WAV files
                    (ambient recordings with no aircraft). When provided,
                    AddBackgroundNoise is included — strongly recommended.
    """
    transforms = [
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
        Gain(min_gain_db=-12, max_gain_db=12, p=0.5),
    ]

    if bgNoiseDir is not None:
        transforms.insert(0, AddBackgroundNoise(
            sounds_path=bgNoiseDir,
            min_snr_db=3.0,
            max_snr_db=30.0,
            p=0.5,
        ))

    return Compose(transforms)


# Usage in a Dataset.__getitem__:
#
#   augPipeline = buildAugPipeline(bgNoiseDir="data/background_noise/")
#
#   waveform = waveform.numpy()[0]       # audiomentations expects numpy float32
#   if augment:
#       waveform = augPipeline(waveform, sample_rate=sampleRate)
#   waveform = torch.from_numpy(waveform).unsqueeze(0)
