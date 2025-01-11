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

def generate_random_midi_multi(filename, instruments):
    mid = mido.MidiFile()

    tempo_bpm = 120
    tempo = mido.bpm2tempo(tempo_bpm)

    for i, inst in enumerate(instruments):
        channel = i % 16
        track = mido.MidiTrack()
        mid.tracks.append(track)

        track.append(mido.Message('program_change', program=inst, channel=channel))
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))

        for _ in range(5):
            note = random.randint(60, 72)
            velocity = random.randint(50, 100)

            note_on_time  = mido.second2tick(random.uniform(0.0, 1.0), mid.ticks_per_beat, tempo)
            note_off_time = mido.second2tick(random.uniform(0.3, 2.0), mid.ticks_per_beat, tempo)

            track.append(mido.Message('note_on',  note=note, velocity=velocity, time=int(note_on_time),  channel=channel))
            track.append(mido.Message('note_off', note=note, velocity=velocity, time=int(note_off_time), channel=channel))


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
                fs.noteon(msg.channel, msg.note, msg.velocity)
            elif msg.type == 'note_off':
                fs.noteoff(msg.channel, msg.note)
            elif msg.type == 'program_change':
                fs.program_change(msg.channel, msg.program)

            audio_data.extend(fs.get_samples(10000))

    audio_data = np.array(audio_data)
    sf.write(output_wav, audio_data, samplerate=44100, subtype='PCM_16')

    fs.delete()


# 3. Generate spectrogram and save to file
def generate_spectrogram(wav_file, output_file, n_mels, fig_size, showAxis=False):
    print(f'Generating spectrogram for {wav_file}...')
    y, sr = librosa.load(wav_file)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=fig_size)
    if showAxis:
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.tight_layout()
    else:
        librosa.display.specshow(S_dB, sr=sr, x_axis='off', y_axis='off')
        plt.axis('off')
    plt.savefig(output_file, bbox_inches="tight", dpi=200)
    plt.close()
    print(f'Spectrogram saved to {output_file}')


def main():
    parser = argparse.ArgumentParser(description="Generate and save spectrograms.")
    parser.add_argument("--fname", type=str, help="Spectrogram file name")
    parser.add_argument("--soundfont", type=str, default="FluidR3_GM.sf2", help="Path to the SoundFont file")
    parser.add_argument("--output_folder", type=str, default="output", help="Folder to save the generated files")
    parser.add_argument("--instruments", type=int, nargs='+', default=[0],
                        help="List of MIDI instruments (0-127). E.g. --instruments 0 24 40")
    parser.add_argument("--save_midi", action="store_true", help="Save generated MIDI files")
    parser.add_argument("--save_wav", action="store_true", help="Save generated WAV files")
    parser.add_argument("--resolution", type=str, default="512,10,10",
                        help="Resolution settings: n_mels,fig_width,fig_height")

    args = parser.parse_args()
    fname = args.fname
    soundfont_file = args.soundfont
    output_folder = args.output_folder
    instruments = args.instruments
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

    midi_file = os.path.join(midi_folder, fname + f'.mid') if save_midi else None
    wav_file = os.path.join(wav_folder, fname + f'.wav') if save_wav else "temp.wav"
    spectrogram_file = os.path.join(spectrogram_folder, fname + f'.png')

    if save_midi:
        generate_random_midi_multi(midi_file, instruments)
    else:
        midi_file = "temp.mid"
        generate_random_midi_multi(midi_file, instruments)

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
