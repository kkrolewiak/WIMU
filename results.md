# Wyniki eksperymentów
## Generownaie danych
Generowanie danych składa się z 3 głównych etapów: wygenerowania pliku MIDI, konwersji MIDI na WAV oraz stwozenia mel-spektrogramów z plików .wav. 
Za pomocą bibliteki mido tworzone są pliki MIDI składające się z losowych dźwięków o wysokości z przedziału 60–72 (od C4 do C5 włącznie), różnej długości.
Pliki MIDI są syntezowane do WAV przy użyciu biblioteki fluidsynth z samplowaniem 44.1 kHz.
Z plików WAV tworzone są spektrogramy przy użyciu biblioteki librosa z wartościami `window size`: 2048, `hop length`: 512, rodzaj funkcji kształtującej okno: `Hann window`.

W przeprowadzonych przez nas eksperymentach wygenerowaliśmy dźwięki 12 różnych instrumentów: 
- Acoustic Grand Piano
- Nylon Acoustic Guitar
- String Ensemble 1
- Piccolo
- Celesta
- Acoustic Bass
- Trumpet
- Square Wave Lead
- Hammond Organ
- Violin
- Soprano Sax
- New Age Pad

W każdej próbce znajdują się dźwięki od 1 do 4 instrumentów. Każdy instrument wstępuje w około 20% próbek. Do eksperymentów zostało wygenerowanych 8000 próbek. Dokładne statystki każdego instrumentu podane są poniżej:

Instrument name          Occurences in dataset
Acoustic Grand Piano     1671
Nylon Acoustic Guitar    1646
String Ensemble 1        1643
Piccolo                  1679
Celesta                  1652
Acoustic Bass            1684
Trumpet                  1614
Square Wave Lead         1716
Hammond Organ            1638
Violin                   1698
Soprano Sax              1711
New Age Pad              1702

Liczba instrumentów w przykładzie    Ilość wystąpień
1                                    2031
2                                    1931
3                                    1991
4                                    2047

Zbiór został podzielony na część treningową i walidacyjną w proorcji 80-20. 

Przykładowy wygenerowany plik wav - fortepian


https://github.com/user-attachments/assets/b03dd50b-78a2-4a85-91bf-e1c7651b83ae

Przykładowy wygenerowana plik wav - fortepian, skrzypce, trabka i new age pad


https://github.com/user-attachments/assets/af86d618-3c87-4bd6-8d51-aa367ce6f821

Przykładowy spektrogram - fortepian

![Grand Piano](https://github.com/user-attachments/assets/3b222643-3888-43b9-870c-5e747814415d)

Przykładowy spektrogram - fortepian, skrzypce, trabka i new age pad

![4 together](https://github.com/user-attachments/assets/54a76c72-9c99-43d4-bacd-7b9bdc9d6a75)

## Architektura modelu

Stworzony przez nas model konwolucyjna sieć neuronowa składająca się z czterech warstw konwolucyjnych, każda z normalizacją batchową i aktywacją ReLU, wspieranych poolingiem maksymalnym do redukcji wymiarów. Na wejście sieci podawane są czarno-białe spectrogramy o rozdzielczości 400x400 pikseli. Dla każdej kolejnej warstwy konwolucyjnej liczba map cech rośnie progresywnie od 8 do 128. Po przejściu przez warstwy konwolucyjne dane są spłaszczane, przechodzą przez warstwę dropout i trafiają do warstwy w pełni połączonej, która generuje wyjście o rozmiarze odpowiadającym liczbie klas, czyli ilości różnych intrumentów w zbiorze trenującym.
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
Ze względu na istotną dysproporcję między liczbą próbek, gdzie dany instrument nie występuje a liczbą próbek, gdzie występuje, jako metrykę skuteczności modelu przyjęliśmy makrouśrednianą miarę F1 Score. Wytrenowany przez nas model osiągnął wynik 0.92. Poniżej przedstawione zostały wyniki przed uśrednieniem, dla każdego instrumenu.

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


## Analiza Concept Relevence Propragation

Model został przeanalizowany techniką Concept Relevence Propagation (CRP). Kod źródłowy oryginalnej implmentacji dostępny: https://github.com/rachtibat/zennit-crp/tree/master

Biblioteka ta pozwala na wizualiacje wpływu części spectrogramu na decyzję sieci. Kolor Niebieski pokazuje czynniki wpływające negatywnie (widzę to więc wiem że to nie ten instrument), czerwone zaś pozytywnie (widzę to więc wiem że to ten instrument). Przygotowany został bardzo mały zbiór danych zawierający ciekawe kombinacje intrumentów do analizy oraz skrypt pozwalający oglądać spectrogramy pod kątem klasyfikacji jako konkretny instrument, oraz całościowo jako jeden z wielu (concept_propagation_tests.ipynb)

Poniżej pokazana analiza przykładu który zawiera wyłącznie instrument "Acustic Bass":
Spectrogram:
![spectrogram2](https://github.com/user-attachments/assets/c6e7df4e-acae-42e2-8756-918c156900d5)
Atrybucja CRP:
![Bass_crp](https://github.com/user-attachments/assets/608d3fa6-e2a6-4852-b0cc-fd6141eb61aa)

Widać że model zwraca uwagę na atak pierwszej nuty zagranej przez instrument, i ogólnie skupia się na najiższych i najwyższych harmonicznych. Zdaje się to być bardzo sensowane, ludzie też zazwyczają rozpoznają brzmie intrumentu po ataku oraz rozkładzie harmoniczncyh. 

Rzućmy okiem na przykład zawierający instrumenty "Square Lead" i "New Age Pad":
Spectrogram:
![leadPad](https://github.com/user-attachments/assets/157a1e0f-403b-4c26-b582-3d63f5ae636c)
Atrybucja CRP:
![cpr_square_pad](https://github.com/user-attachments/assets/6934d440-07ae-466b-9bd5-17884e32a6ef)

Model dokonuje tutaj prawidłowej klasyfikacji zwracając uwagę na zupełnie inny rejestr harmonicznych. Oba te intrumenty mają dużą składową wysokich harmonicznych w porównaniu do instrumentów akustycznych. Square lead sechuje się bardzo krótkim atakiem i releasem co skutkije bardzo ostrym spectrogramem, pad caś ma długi atak i bardzo długi release, oraz duży pokłos i więcej różnych składowych harmonicznych, nie tylko nieparzystych wielokrotności co skutkuje rozmazanym spectrogramem. Jest to poniekąd widoczne na analizie w postaci dłuższych czerwonych smug na atrybucji dla pada. 

Ostatnia analiza której się przyjżymy to przykład dla 4 instrumentów na raz: "Acoustic Grand Piano", "Trumpet", "Violin" oraz "New Age Pad"
Spectrogram:
![pianoviolinnewagepadtrumpet](https://github.com/user-attachments/assets/6fb06089-b3c2-442d-9bb4-6a4bdbd33d95)
Atrybucja CRP:
![4instr_crp](https://github.com/user-attachments/assets/0b3d0681-35c5-4bcb-87e8-5ff7959ec2ee)

Na tym przykładzie truno już jest się połapać patrząc na spectrogram. Co więcej pianino jest praktycznie niesłyszalne w pliku audio. Niemniej nie przeszkadza to modelowi w prawidłowej klasyfikacji. Miałem nadzieję że model zwróci uwagę na to że dźwięk pochodzący od "violin" na spectrogramie ma bardzo pofalowane spektrum od efektu wibrato, "trumpet" zaś zaczyna każdą nutę od nieco niższej częstotliwości a później narasta. Z atrybucji ciężko jest jednoznacznie powiedzieć że model zwraca na to uwagę. Raczej po prostu bardzo dobrze nauczył się rozkładu harmonicznych dla każdego instrumentu obrazują mocno zarysowane linie w bardzo konkretnych odstępach i w konkretych pasmach. Jest to też na pewno jeden z poprawnych sposobów na rozpoznanie brzmienia intrumentu jako że to właśnie rozkład harmonicznych decyduje o brzmieniu. 
