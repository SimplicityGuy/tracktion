"""
Generate synthetic test audio files for BPM detection testing.

This script creates test audio files with known characteristics
for unit and integration testing without requiring actual music files.
"""

import wave
from pathlib import Path

import numpy as np


def generate_click_track(
    bpm: float,
    duration_seconds: float = 10.0,
    sample_rate: int = 44100,
    click_frequency: float = 1000.0,
    click_duration: float = 0.01,
) -> np.ndarray:
    """Generate a click track at a specific BPM.

    Args:
        bpm: Beats per minute
        duration_seconds: Duration of the track
        sample_rate: Sample rate in Hz
        click_frequency: Frequency of the click sound in Hz
        click_duration: Duration of each click in seconds

    Returns:
        Audio signal as numpy array
    """
    # Calculate timing
    beat_interval = 60.0 / bpm  # seconds between beats
    num_samples = int(duration_seconds * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float32)

    # Generate clicks
    click_samples = int(click_duration * sample_rate)
    t = np.linspace(0, click_duration, click_samples)
    click = np.sin(2 * np.pi * click_frequency * t) * np.exp(-t * 10)  # Decaying sine

    # Place clicks at beat intervals
    current_time = 0.0
    while current_time < duration_seconds:
        sample_pos = int(current_time * sample_rate)
        end_pos = min(sample_pos + click_samples, num_samples)
        actual_samples = end_pos - sample_pos
        audio[sample_pos:end_pos] += click[:actual_samples] * 0.5
        current_time += beat_interval

    return audio


def generate_drum_pattern(
    bpm: float,
    duration_seconds: float = 10.0,
    sample_rate: int = 44100,
    pattern: str = "basic",
) -> np.ndarray:
    """Generate a drum pattern at a specific BPM.

    Args:
        bpm: Beats per minute
        duration_seconds: Duration of the track
        sample_rate: Sample rate in Hz
        pattern: Type of pattern ('basic', 'complex')

    Returns:
        Audio signal as numpy array
    """
    beat_interval = 60.0 / bpm
    num_samples = int(duration_seconds * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float32)

    # Kick drum (low frequency)
    kick_freq = 60.0
    kick_duration = 0.1

    # Snare drum (mid frequency with noise)
    snare_freq = 200.0
    snare_duration = 0.05

    # Hi-hat (high frequency noise)
    hihat_duration = 0.02

    # Generate drum sounds
    kick_samples = int(kick_duration * sample_rate)
    t_kick = np.linspace(0, kick_duration, kick_samples)
    kick = np.sin(2 * np.pi * kick_freq * t_kick) * np.exp(-t_kick * 20)

    snare_samples = int(snare_duration * sample_rate)
    t_snare = np.linspace(0, snare_duration, snare_samples)
    snare = (np.sin(2 * np.pi * snare_freq * t_snare) + np.random.normal(0, 0.1, snare_samples)) * np.exp(-t_snare * 30)

    hihat_samples = int(hihat_duration * sample_rate)
    hihat = np.random.normal(0, 0.05, hihat_samples) * np.exp(-np.linspace(0, hihat_duration, hihat_samples) * 50)

    if pattern == "basic":
        # Basic 4/4 pattern: kick on 1&3, snare on 2&4
        current_time = 0.0
        beat_count = 0
        while current_time < duration_seconds:
            sample_pos = int(current_time * sample_rate)

            if beat_count % 4 == 0 or beat_count % 4 == 2:  # Kick
                end_pos = min(sample_pos + kick_samples, num_samples)
                actual_samples = end_pos - sample_pos
                audio[sample_pos:end_pos] += kick[:actual_samples] * 0.7

            if beat_count % 4 == 1 or beat_count % 4 == 3:  # Snare
                end_pos = min(sample_pos + snare_samples, num_samples)
                actual_samples = end_pos - sample_pos
                audio[sample_pos:end_pos] += snare[:actual_samples] * 0.5

            # Hi-hat on every beat
            end_pos = min(sample_pos + hihat_samples, num_samples)
            actual_samples = end_pos - sample_pos
            audio[sample_pos:end_pos] += hihat[:actual_samples]

            current_time += beat_interval
            beat_count += 1

    return audio


def generate_variable_tempo(
    start_bpm: float,
    end_bpm: float,
    duration_seconds: float = 10.0,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Generate audio with gradually changing tempo.

    Args:
        start_bpm: Starting BPM
        end_bpm: Ending BPM
        duration_seconds: Duration of the track
        sample_rate: Sample rate in Hz

    Returns:
        Audio signal as numpy array
    """
    num_samples = int(duration_seconds * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float32)

    # Linear tempo change
    current_time = 0.0
    beat_count = 0
    click_samples = int(0.01 * sample_rate)
    t = np.linspace(0, 0.01, click_samples)
    click = np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 10)

    while current_time < duration_seconds:
        # Calculate current BPM
        progress = current_time / duration_seconds
        current_bpm = start_bpm + (end_bpm - start_bpm) * progress
        beat_interval = 60.0 / current_bpm

        # Place click
        sample_pos = int(current_time * sample_rate)
        end_pos = min(sample_pos + click_samples, num_samples)
        actual_samples = end_pos - sample_pos
        audio[sample_pos:end_pos] += click[:actual_samples] * 0.5

        current_time += beat_interval
        beat_count += 1

    return audio


def generate_silence(duration_seconds: float = 5.0, sample_rate: int = 44100) -> np.ndarray:
    """Generate silent audio.

    Args:
        duration_seconds: Duration of silence
        sample_rate: Sample rate in Hz

    Returns:
        Silent audio signal
    """
    return np.zeros(int(duration_seconds * sample_rate), dtype=np.float32)


def generate_noise(duration_seconds: float = 5.0, sample_rate: int = 44100, noise_type: str = "white") -> np.ndarray:
    """Generate noise audio.

    Args:
        duration_seconds: Duration of noise
        sample_rate: Sample rate in Hz
        noise_type: Type of noise ('white', 'pink')

    Returns:
        Noise audio signal
    """
    num_samples = int(duration_seconds * sample_rate)

    if noise_type == "white":
        return np.random.normal(0, 0.1, num_samples).astype(np.float32)
    elif noise_type == "pink":
        # Simple pink noise approximation
        white = np.random.normal(0, 0.1, num_samples)
        # Apply simple lowpass filter
        pink = np.zeros_like(white)
        pink[0] = white[0]
        for i in range(1, len(white)):
            pink[i] = pink[i - 1] * 0.9 + white[i] * 0.1
        return pink.astype(np.float32)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def save_wav(audio: np.ndarray, filename: str, sample_rate: int = 44100) -> None:
    """Save audio to WAV file.

    Args:
        audio: Audio signal as numpy array
        filename: Output filename
        sample_rate: Sample rate in Hz
    """
    # Normalize to prevent clipping
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)

    # Save as WAV
    with wave.open(filename, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    print(f"Generated: {filename}")


def main():
    """Generate all test audio files."""
    # Create output directory
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)

    # Generate constant BPM tracks
    test_bpms = [
        (60, "slow"),
        (85, "hiphop"),
        (120, "rock"),
        (128, "electronic"),
        (140, "dnb"),
        (175, "fast"),
    ]

    for bpm, genre in test_bpms:
        # Click track
        audio = generate_click_track(bpm, duration_seconds=30.0)
        save_wav(audio, str(output_dir / f"test_{bpm}bpm_click.wav"))

        # Drum pattern
        audio = generate_drum_pattern(bpm, duration_seconds=30.0)
        save_wav(audio, str(output_dir / f"test_{bpm}bpm_{genre}.wav"))

    # Generate variable tempo track
    audio = generate_variable_tempo(100, 140, duration_seconds=30.0)
    save_wav(audio, str(output_dir / "test_variable_tempo.wav"))

    # Generate edge cases
    audio = generate_silence(duration_seconds=10.0)
    save_wav(audio, str(output_dir / "test_silence.wav"))

    audio = generate_noise(duration_seconds=10.0, noise_type="white")
    save_wav(audio, str(output_dir / "test_white_noise.wav"))

    # Very short file (0.5 seconds)
    audio = generate_click_track(120, duration_seconds=0.5)
    save_wav(audio, str(output_dir / "test_very_short.wav"))

    print("\nTest audio files generated successfully!")
    print("Files are in WAV format for maximum compatibility.")
    print("Convert to other formats (MP3, FLAC, etc.) as needed for format testing.")


if __name__ == "__main__":
    main()
