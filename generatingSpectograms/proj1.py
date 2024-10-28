import mido
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import fluidsynth
import soundfile as sf
import time

# 1. Generate MIDI
def generate_random_midi(filename):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.Message('program_change', program=0))  # Choose instrument
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
    
    fs.start(driver="dsound")
    
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


# 3. Generate spectogram
def generate_spectrogram(wav_file):
    print(f'Generating spectogram for {wav_file}...')
    y, sr = librosa.load(wav_file)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.show()

def main():
    midi_file = 'random_mido_midi.mid'
    soundfont_file = 'soundfont.sf2'
    wav_file = 'output_pyfluidsynth.wav'

    generate_random_midi(midi_file)

    if os.path.exists(soundfont_file):
        convert_midi_to_wav(midi_file, soundfont_file, wav_file)
    else:
        print(f'SoundFont file {soundfont_file} has not been found.')

    if os.path.exists(wav_file):
        generate_spectrogram(wav_file)
    else:
        print(f'WAV file {wav_file} has not been found.')

if __name__ == "__main__":
    main()

