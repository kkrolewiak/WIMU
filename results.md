# Wyniki eksperymentów
## Generownaie danych
Generowanie danych składa się z 3 głównych etapów: wygenerowania pliku MIDI, konwersji MIDI na WAV oraz stwozenia mel-spektrogramów z plików .wav. 
Generowane są pliki MIDI składające się z losowo wybranych ścieżek. 
12 instrumentów, 1 do 4 instrumentów w próbce, każdy instrument występuje w około 20% próbek.
Pliki MIDI są syntezowane do WAV przy użyciu biblioteki fluidsynth.
Z plików WAV tworzone są spektrogramy przy użyciu biblioteki librosa
