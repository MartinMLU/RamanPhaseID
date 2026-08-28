from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_spectrum(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(file_path, comments="#")
    if data.ndim == 1 and data.size == 2:
        data = data.reshape(1, 2)
    if data.shape[1] < 2:
        raise ValueError(f"Unexpected spectrum format in {file_path.name}")
    return data[:, 0], data[:, 1]


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    spectra_files = sorted(base_dir.glob("*Edge*.txt"))

    if not spectra_files:
        raise FileNotFoundError("No files matching 'Calcite*.txt' were found.")

    plt.figure(figsize=(12, 7))
    for spectrum_file in spectra_files:
        x, y = load_spectrum(spectrum_file)
        plt.plot(x, y, linewidth=1.0, label=spectrum_file.stem)

    plt.title("Calcite Raman Spectra")
    plt.xlabel("Raman shift (cm^-1)")
    plt.ylabel("Intensity (a.u.)")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()

    output_file = base_dir / "calcite_raman_overlay.png"
    plt.savefig(output_file, dpi=200)
    print(f"Saved plot to: {output_file}")
    plt.show()


if __name__ == "__main__":
    main()
