# WIMU - Music Spectrogram Explained Using Concept Relevance Propagation
## Autorzy
Mateusz Szczęsny, Kacper Królewiak, Kacper Łobodecki
## Wstęp
Projekt polega na tłumaczeniu działania sieci neuronowych używanych do zagadnień MIR za pomocą techniki znanej jako Concept Relevance Propagation [[https://arxiv.org/pdf/2206.03208](https://arxiv.org/pdf/2206.03208)].

Do tego celu zostanie wytrenowana kilku warstwowa, płytka splotowa sieć neuronowa, której celem będzie rozpoznawanie, jakie instrumenty muzyczne są obecne w wejściowym fragmencie muzyki reprezentowanym jako spektrogram. Stworzymy do tego własną sieć, żeby mieć możliwie jak największą kontrolę nad jej rozmiarem i parametrami. To pozwoli nam na łatwiejsze „zaglądanie do środka”, w celu weryfikacji zachowania sieci oraz sprawdzenia, czy tłumaczenie jest poprawne.

Sieć będzie trenowana na również przygotowanym przez nas zbiorze danych, który będzie składał się ze spektrogramów 5-sekundowych fragmentów wygenerowanej muzyki. Jego konstrukcja będzie lepiej wytłumaczona w podsekcji „zbiory danych”.

Tak przygotowana i wytrenowana sieć zostanie przebadana narzędziem Concept Relevance Propagation [[https://github.com/fxnnxc/crp_pytorch](https://github.com/fxnnxc/crp_pytorch)]. Zostanie ono w razie potrzeby przystosowane do naszego zastosowania.

## Zbiory danych
Żeby mieć jak największą kontrolę nad uczeniem naszej sieci, przygotujemy własny zbiór danych uczących. Zbiór ten będzie składał się ze spektrogramów 5-sekundowych fragmentów wygenerowanej komputerowo muzyki. Proces generacji będzie przebiegał następująco:
1.  Wylosowane zostaną instrumenty z listy
2.  Dla każdego instrumentu zostanie:
    - Wygenerowana losowa melodia
	-  Wygenerowana melodia przy użyciu narzędzia Symbotunes [[https://github.com/pskiers/Symbotunes](https://github.com/pskiers/Symbotunes)]
3.  Fragment muzyki zostanie syntezowany i na jego podstawie zostanie wygenerowany spektrogram.
4.  Spektrogram będzie skojarzony z informacją, jakie instrumenty były używane przy jego generacji.
    
Dla każdego instrumentu zostanie zdefiniowany zakres wysokości dźwięków, które ten instrument może zagrać oraz typowy zakres ilości dźwięków, które mogą być grane na raz (np. na pianinie często gra się od 1 do 8 klawiszy na raz, na trąbce granie więcej niż jednego dźwięku jednocześnie jest problematyczne)

Przygotowanie własnego zbioru danych będzie nam pozwalać na uzyskanie dużej kontroli nad liczbą instrumentów, które będą grały w danym fragmencie muzyki oraz dobieranie instrumentów, które charakteryzują się podobnym lub różnym brzmieniem (prawdopodobnie trudniej jest rozpoznać poszczególne instrumenty smyczkowe na tle orkiestry niż rozpoznać gitarę i perkusję w nagraniu 2-instrumentowym)

## Proponowany zakres eksperymentów
Testy modelu:
-   Sprawdzenie modelu na wygenerowanych próbkach testowych jedno i wiele instrumentalnych, a także na prawdziwych nagraniach utworów muzyki różnego typu (pop, orkiestra symfoniczna, rock)
-   Wpływ liczby instrumentów na jakość klasyfikacji.
-   Rozpoznawanie podobnych do siebie instrumentów grających na raz np. Różne smyczki, różne dęte.

Testy XAI:
-   Czy model o większej liczbie warstw będzie miał taką samą zasadę działania?
-   Czy modele trenowane na bardziej skomplikowanym zbiorze danych będą miały inny zasadę działania?

## Funkcjonalności
-   Identyfikacja instrumentów użytych do wygenerowania melodii widocznej na spektrogramie.
-   Wizualizacja, które fragmenty spektrogramu są kluczowe dla klasyfikacji za pomocą Concept Relevance Propagation.

## Harmonogram
18.10.24 Początek realizacji projektu

20.10.24 Oddanie dokumentacji wstępnej

27.10.24 Wstępny skrypt do generowania próbek, konfiguracja środowiska eksperymentalnego

03.11.24 Pierwsza działająca sieć do klasyfikacji próbek, zakończenie analizy literaturowej

17.11.24 Pierwsze rezultaty Concept Relevance Propagation na bazowej sieci

23.11.24 Ocena czy prosta sieć splotowa nadaje się do zagadnienia czy należy zmienić podejście

01.12.24 Przygotowanie architektur sieci, które będą nadawać się do analizy.

15.12.24 Wstępne wyniki Concept Relevance Propagation

30.01.24 Szlifowanie, oddanie projektu

## Stack technologiczny
-   Python do generacji zbiorów danych
-   Poetry do konfiguracji środowiska wirtualnego
-   Black jako autoformatter, Flake8 jako linter
-   Pytorch do tworzenia sieci neuronowych
-   Biblioteka do CRP [[https://github.com/fxnnxc/crp_pytorch](https://github.com/fxnnxc/crp_pytorch)]
-   Biblioteka MIDO - generowanie plików midi, które będą syntezowane, żeby tworzyć zbiór uczący, alternatywnie Symbotunes
-   Biblioteka SciPy - tworzenie spektrogramów
-   Biblioteka pyFluidSynth - pozwoli na odtwarzanie plików midi używając typowych brzmień general MIDI synth. (Można potem spróbować czymś lepszej jakości)

## Bibliografia
1. From Attribution Maps to Human-Understandable Explanations through Concept Relevance Propagation - [https://arxiv.org/pdf/2206.03208](https://arxiv.org/pdf/2206.03208)
2. Deep Convolutional Neural Networks for Predominant Instrument Recognition in Polyphonic Music - [https://sci-hub.se/10.1109/taslp.2016.2632307](https://sci-hub.se/10.1109/taslp.2016.2632307)
