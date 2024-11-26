import mido
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import fluidsynth
import soundfile as sf
import argparse
import time


# 1. Generate MIDI
def generate_random_midi(filename, instrument):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.Message('program_change', program=instrument))  # Set instrument
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120)))

    for i in range(5):
        note = random.randint(60, 72)  # Random pitch in one octave
        velocity = random.randint(50, 100)  # Random velocity
        track.append(mido.Message('note_on', note=note, velocity=velocity, time=0))
        track.append(mido.Message('note_off', note=note, velocity=velocity, time=mido.second2tick(1, mid.ticks_per_beat, mido.bpm2tempo(120))))

    mid.save(filename)
    print(f'MIDI file has been saved: {filename}')


# 2. Convert MIDI to WAV
def convert_midi_to_wav(midi_file, soundfont_file, output_wav):
    fs = fluidsynth.Synth()
    sfid = fs.sfload(soundfont_file)
    fs.program_select(0, sfid, 0, 0)

    midi_data = mido.MidiFile(midi_file)

    audio_data = []

    for msg in midi_data:
        if not msg.is_meta:
            time_to_wait = mido.tick2second(msg.time, midi_data.ticks_per_beat, mido.bpm2tempo(120))
            time.sleep(time_to_wait)

            if msg.type == 'note_on':
                fs.noteon(0, msg.note, msg.velocity)
            elif msg.type == 'note_off':
                fs.noteoff(0, msg.note)
            elif msg.type == 'control_change':
                fs.cc(0, msg.control, msg.value)
            elif msg.type == 'program_change':
                fs.program_change(0, msg.program)

            audio_data.extend(fs.get_samples(10000))

    audio_data = np.array(audio_data)

    sf.write(output_wav, audio_data, samplerate=44100, subtype='PCM_16')

    fs.delete()


# 3. Generate spectrogram and save to file
def generate_spectrogram(wav_file, output_file, n_mels, fig_size):
    print(f'Generating spectrogram for {wav_file}...')
    y, sr = librosa.load(wav_file)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=fig_size)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f'Spectrogram saved to {output_file}')


def main():
    parser = argparse.ArgumentParser(description="Generate and save spectrograms.")
    parser.add_argument("num_spectrograms", type=int, help="Number of spectrograms to generate")
    parser.add_argument("--soundfont", type=str, default="FluidR3_GM.sf2", help="Path to the SoundFont file")
    parser.add_argument("--output_folder", type=str, default="output", help="Folder to save the generated files")
    parser.add_argument("--instrument", type=int, default=0, help="MIDI instrument number (0-127)")
    parser.add_argument("--save_midi", action="store_true", help="Save generated MIDI files")
    parser.add_argument("--save_wav", action="store_true", help="Save generated WAV files")
    parser.add_argument("--resolution", type=str, default="512,10,10", help="Resolution settings: n_mels,fig_width,fig_height")

    args = parser.parse_args()
    num_spectrograms = args.num_spectrograms
    soundfont_file = args.soundfont
    output_folder = args.output_folder
    instrument = args.instrument
    save_midi = args.save_midi
    save_wav = args.save_wav
    resolution = list(map(float, args.resolution.split(',')))

    n_mels = int(resolution[0])
    fig_size = (resolution[1], resolution[2])

    if not os.path.exists(soundfont_file):
        print(f"SoundFont file {soundfont_file} not found.")
        return

    midi_folder = os.path.join(output_folder, "midi") if save_midi else None
    wav_folder = os.path.join(output_folder, "wav") if save_wav else None
    spectrogram_folder = os.path.join(output_folder, "spectrogram")

    if save_midi:
        os.makedirs(midi_folder, exist_ok=True)
    if save_wav:
        os.makedirs(wav_folder, exist_ok=True)
    os.makedirs(spectrogram_folder, exist_ok=True)

    for i in range(num_spectrograms):
        midi_file = os.path.join(midi_folder, f'random_midi_{i + 1}.mid') if save_midi else None
        wav_file = os.path.join(wav_folder, f'output_{i + 1}.wav') if save_wav else "temp.wav"
        spectrogram_file = os.path.join(spectrogram_folder, f'spectrogram_{i + 1}.png')

        if save_midi:
            generate_random_midi(midi_file, instrument)
        else:
            midi_file = "temp.mid"
            generate_random_midi(midi_file, instrument)

        if save_wav:
            convert_midi_to_wav(midi_file, soundfont_file, wav_file)
        else:
            wav_file = "temp.wav"
            convert_midi_to_wav(midi_file, soundfont_file, wav_file)

        if os.path.exists(wav_file):
            generate_spectrogram(wav_file, spectrogram_file, n_mels, fig_size)
        else:
            print(f'WAV file {wav_file} not found.')

        if not save_midi and os.path.exists(midi_file):
            os.remove(midi_file)
        if not save_wav and os.path.exists(wav_file):
            os.remove(wav_file)


if __name__ == "__main__":
    main()
