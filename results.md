# Wyniki eksperymentów
## Generownaie danych
Generowanie danych składa się z 3 głównych etapów: wygenerowania pliku MIDI, konwersji MIDI na WAV oraz stwozenia mel-spektrogramów z plików .wav. 
Za pomocą bibliteki mido tworzone są pliki MIDI składające się z losowo wybranych ścieżek.
Pliki MIDI są syntezowane do WAV przy użyciu biblioteki fluidsynth z samplowaniem 44.1 kHz.
Z plików WAV tworzone są spektrogramy przy użyciu biblioteki librosa.

W przeprowadzonych przez nas eksperymentach wygenerowaliśmy dźwięki 12 różnych instrumentów. W każdej próbce znajdują się dźwięki od 1 do 4 instrumentów.
Każdy instrument wstępuje w około 20% próbek.

Przykładowy wygenerowany plik wav - akustyczna gitara basowa


https://github.com/user-attachments/assets/0a07fd1a-fd52-47f1-bb53-71b741f06896

Przykładowy wygenerowana plik wav - pianino, skrzypce, trabka i new age pad


https://github.com/user-attachments/assets/af86d618-3c87-4bd6-8d51-aa367ce6f821

Przykładowy spektrogram - fortepian

![Grand Piano](https://github.com/user-attachments/assets/3b222643-3888-43b9-870c-5e747814415d)

Przykładowy spektrogram - fortepian, skrzypce, trabka i new age pad

![4 together](https://github.com/user-attachments/assets/54a76c72-9c99-43d4-bacd-7b9bdc9d6a75)

## Architektura modelu

## Trening modelu

![training_loss](https://github.com/user-attachments/assets/875a24a0-733b-4039-a309-7fad8184e543)


## Wyniki modelu
Jako metrykę skuteczności modelu przyjęliśmy makrouśrednianą miarę F1 Score, ze względu na istotną dysproporcję między liczbą próbek, gdzie dany instrument nie występuje niż występuje. Wytrenowany przez nas model osiągnął wynik 0.92. Poniżej przedstawione zostały wyniki przed uśrednieniem, dla każdego instrumenu.

| Instrument              | F1 Score |
|-------------------------|----------|
| Acoustic Grand Piano    | 0.82     |
| Nylon Acoustic Guitar   | 0.93     |
| String Ensemble 1       | 0.93     |
| Piccolo                 | 0.97     |
| Celesta                 | 0.87     |
| Acoustic Bass           | 0.78     |
| Trumpet                 | 0.90     |
| Square Wave Lead        | 0.99     |
| Hammond Organ           | 0.87     |
| Violin                  | 0.97     |
| Soprano Sax             | 0.96     |
| New Age Pad             | 1.00     |
