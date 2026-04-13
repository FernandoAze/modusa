"""Simple Modusa example: waveform + spectrogram.

This script shows the modular workflow Modusa is built around:
1. Prepare the data.
2. Create a figure layout.
3. Paint each layer onto its own axis.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import modusa as ms


def main() -> None:
    # Set this to the audio file you want to visualize.
    audio_path = "/Users/fernando/Desktop/AI SONG.wav"

    # 1) Load the waveform from disk.
    y, sr, _ = ms.load.audio(audio_path)
    if y.ndim == 2:
        y = y.mean(axis=0)

    t = np.arange(y.size) / sr
    duration_sec = (y.size / sr) if y.size else 0.0

    # 2) Compute a spectrogram with Modusa's numpy-only STFT helper.
    winlen = int(0.064 * sr)
    hoplen = max(1, winlen // 4)
    S, freqs, times = ms.compute.stft(y, sr, winlen=winlen, hoplen=hoplen)
    S_db = 20 * np.log10(np.maximum(np.abs(S), 1e-10))

    # 3) Create a time-aligned figure with one waveform tier and one spectrogram tier.
    fig, axs = ms.create.figure.tracks(
        "sm",
        fig_width=12,
        xlim=(0, duration_sec),
        ylabels=["Amplitude", "Hz"],
        xlabels=[None, "Time (s)"],
        titles=["Waveform", "Spectrogram"],
        grid=True,
        abc=True,
    )

    # 4) Paint the waveform and spectrogram onto their own axes.
    ms.paint.signal(axs[0, 0], y, x=t, c="black", lw=1.0)
    ms.paint.image(
        axs[1, 0],
        S_db,
        x=times,
        y=freqs,
        c="magma",
        o="lower",
        clabel="Magnitude (dB)",
        cax=axs[1, 1],
    )

    axs[0, 0].set_ylim(-1.1, 1.1)
    axs[1, 0].set_ylim(20, 4_000)

    plt.show()


if __name__ == "__main__":
    main()