# Spectrogram Generator 🎵📈

This script generates random MIDI files, converts them to WAV files, and creates spectrograms. You can customize the process using various options, including saving the generated MIDI and WAV files or only keeping the spectrograms.

## Features

- Generate random MIDI files with customizable instruments.
- Convert MIDI files to WAV audio using a SoundFont file.
- Create and save spectrograms of the generated audio.
- Optionally save MIDI and WAV files or just generate spectrograms.

## Requirements

Make sure you have the following Python packages installed:

- `mido`
- `librosa`
- `matplotlib`
- `numpy`
- `pyFluidSynth`
- `soundfile`

Make sure you also install the FluidSynth application (`sudo apt install fluidsynth` on Ubuntu). Otherwise pyFluidSynth will not work. 

You also need a SoundFont file (`.sf2`) for audio synthesis and you will need to install the fluidsynth application. Here you can download sample file: https://member.keymusician.com/Member/FluidR3_GM/index.html

## Usage

Run the script using the command below:

```bash
python script.py [options]
```

### Parameters

| Parameter            | Description                                                   | Default          |
| -------------------- | ------------------------------------------------------------- | ---------------- |
| `--fname`            | File name of the generated spectrogram                        | `spectrogram1`   |
| `--soundfont`        | Path to the SoundFont file (`.sf2`) used for WAV generation.  | `FluidR3_GM.sf2` |
| `--output_folder`    | Folder to save generated files (MIDI, WAV, and spectrograms). | `output`         |
| `--instruments`      | MIDI instruments number (0-127).                              | `0` (Piano)      |
| `--save_midi`        | Save generated MIDI files.                                    | Disabled         |
| `--save_wav`         | Save generated WAV files.                                     | Disabled         |
| `--resolution`       | Resolution settings: n_mels,fig_width,fig_height.             | `512,10,10`      |

### Example

**Generate a spectrogram with soundfont path `path/to/FluidR3_GM.sf2`, saving WAV files, played on the clarinet:**

```bash
python generate_spectrogram.py --fname spectrogram1 --soundfont path/to/FluidR3_GM.sf2 --save_wav --instruments 0 40 71
```

## Output Structure

When files are saved, they are organized in the following folder structure:

```
output/
├── midi/           # MIDI files (if saved)
├── wav/            # WAV files (if saved)
└── spectrogram/    # Spectrogram images
```

## Here you can find codes of the most common instruments

| **Instrument**       | **Program Number** | **MIDI Bank** |
| -------------------- | ------------------ | ------------- |
| Acoustic Grand Piano | 0                  | 0             |
| Violin               | 40                 | 0             |
| Viola                | 41                 | 0             |
| Cello                | 42                 | 0             |
| Contrabass           | 43                 | 0             |
| Flute                | 73                 | 0             |
| Oboe                 | 68                 | 0             |
| Clarinet             | 71                 | 0             |
| Trumpet              | 56                 | 0             |
| French Horn          | 60                 | 0             |

Enjoy generating spectrograms! 🎶✨
