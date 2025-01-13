# Wyniki eksperymentów
## Generownaie danych
Generowanie danych składa się z 3 głównych etapów: wygenerowania pliku MIDI, konwersji MIDI na WAV oraz stwozenia mel-spektrogramów z plików .wav. 
Za pomocą bibliteki mido tworzone są pliki MIDI składające się z losowych dźwięków o wysokości z przedziału 60–72 (od C4 do C5 włącznie), różnej długości.
Pliki MIDI są syntezowane do WAV przy użyciu biblioteki fluidsynth z samplowaniem 44.1 kHz.
Z plików WAV tworzone są spektrogramy przy użyciu biblioteki librosa z wartościami `window size`: 2048, `hop length`: 512, rodzaj funkcji kształtującej okno: `Hann window`.

W przeprowadzonych przez nas eksperymentach wygenerowaliśmy dźwięki 12 różnych instrumentów. W każdej próbce znajdują się dźwięki od 1 do 4 instrumentów.
Każdy instrument wstępuje w około 20% próbek. Do eksperymentów zostało wygenerowanych 8000 próbek.

Przykładowy wygenerowany plik wav - fortepian


https://github.com/user-attachments/assets/b03dd50b-78a2-4a85-91bf-e1c7651b83ae

Przykładowy wygenerowana plik wav - fortepian, skrzypce, trabka i new age pad


https://github.com/user-attachments/assets/af86d618-3c87-4bd6-8d51-aa367ce6f821

Przykładowy spektrogram - fortepian

![Grand Piano](https://github.com/user-attachments/assets/3b222643-3888-43b9-870c-5e747814415d)

Przykładowy spektrogram - fortepian, skrzypce, trabka i new age pad

![4 together](https://github.com/user-attachments/assets/54a76c72-9c99-43d4-bacd-7b9bdc9d6a75)

## Architektura modelu

Stworzony przez nas model konwolucyjna sieć neuronowa składająca się z czterech warstw konwolucyjnych, każda z normalizacją batchową i aktywacją ReLU, wspieranych poolingiem maksymalnym do redukcji wymiarów. Dla każdej kolejnej warstwy konwolucyjnej liczba map cech rośnie progresywnie od 8 do 128. Po przejściu przez warstwy konwolucyjne dane są spłaszczane, przechodzą przez warstwę dropout i trafiają do warstwy w pełni połączonej, która generuje wyjście o rozmiarze odpowiadającym liczbie klas. Podczas generowania predykcji modelu na wyjściu jest sigmoid, określający prawdopodobieństwo czy dany instrument jest w próbce.

| Layer (type)      | Output Shape          | Param #   |
|-------------------|-----------------------|-----------|
| Conv2d-1          | [-1, 8, 398, 398]    | 224       |
| BatchNorm2d-2     | [-1, 8, 398, 398]    | 16        |
| MaxPool2d-3       | [-1, 8, 199, 199]    | 0         |
| Conv2d-4          | [-1, 16, 197, 197]   | 1,168     |
| BatchNorm2d-5     | [-1, 16, 197, 197]   | 32        |
| MaxPool2d-6       | [-1, 16, 98, 98]     | 0         |
| Conv2d-7          | [-1, 32, 96, 96]     | 4,640     |
| BatchNorm2d-8     | [-1, 32, 96, 96]     | 64        |
| MaxPool2d-9       | [-1, 32, 48, 48]     | 0         |
| Conv2d-10         | [-1, 128, 46, 46]    | 36,992    |
| BatchNorm2d-11    | [-1, 128, 46, 46]    | 256       |
| MaxPool2d-12      | [-1, 128, 23, 23]    | 0         |
| Dropout-13        | [-1, 67712]          | 0         |
| Linear-14         | [-1, 12]             | 812,556   |
| **Total params** |                       | 855,948   |
| **Trainable params** |                    | 855,948   |
| **Non-trainable params** |               | 0         |

## Trening modelu

Jako funkcję straty wykorzystaliśmy binarną entropię krzyżową z logitami (BCEWithLogitsLoss z torch.nn) z wagami dla każdej klasy. Wagi zostały obliczone na danych treningowych jako stosunek liczby próbek w których danego instrumentu nie ma do liczby próbek w których instrument jest.
Wielkość batcha została eksperymentalnie ustawiona na 64, model osiągał przy niej najlepsze wyniki. Jako optimizer został wybrany Adam z parametrem learning rate 0.001. Model był trenowany przez 20 epok. 

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
