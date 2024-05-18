[![en](https://img.shields.io/badge/language-EN-blue.svg)](https://github.com/pzemla/Feed-forward-neural-network-library/blob/main/README.md)
# Biblioteka sieci neuronowych typu feed forward

## Przegląd
Podstawowa biblioteka napisana w języku Python do tworzenia sieci neuronowych typu feed forward, zawierająca opcje funkcji aktywacji, funkcji straty, optymalizatorów oraz dropout. Plik `main.py` zawiera przykład trenowania sieci neuronowej stworzonej do klasyfikacji, czy osoba ma chorobę serca na podstawie danych z pliku `heart.dat` po ich wstępnym przetwarzaniu.
Projekt ten służy jako ćwiczenie edukacyjne mające na celu zrozumienie zasad matematycznych, na których opierają się sieci neuronowe.

# Jak używać
Korzystanie z biblioteki jest podobne do biblioteki Pytorch, na której została oparta (z bardziej ograniczonymi możliwościami). Bardziej szczegółowe informacje o klasach, ich funkcjach i argumentach znajdują się w pliku Dokumentacja.pdf. Aby zobaczyć przykład sieci neuronowej zaimplementowanej z biblioteką, upewnij się, że wszystkie pliki .py, a także plik heart.dat znajdują się w jednym folderze i uruchom plik main.py, który zawiera aplikację sieci neuronowej typu feed forward.

## Pliki:
**Loader danych** – służy do efektywnego ładowania i iterowania po zbiorach danych w trakcie trenowania i oceny sieci neuronowej.

**Sieć (Network)** - Zawiera główną implementację sieci neuronowej. Odpowiada za tworzenie i zarządzanie warstwami, wykonywanie przekazów w przód i wstecz, oraz aktualizowanie parametrów podczas trenowania.

**Warstwa liniowa (Linear)** - Definiuje warstwę liniową używaną w sieci.

**Funkcje aktywacji** - Zawiera implementacje różnych funkcji aktywacji:
- Sigmoidalna
- Tangens hiperboliczny (Tanh)
- ReLU
- Leaky ReLU

**Dropout** - Implementuje funkcjonalność dropoutu w celu zapobiegania nadmiernemu dopasowaniu (overfittingowi).

**Funkcje straty** - Oferuje różne funkcje straty do trenowania sieci:
- Średni błąd bezwzględny (MAE)
- Średni błąd kwadratowy (MSE)
- Binarna entropia krzyżowa (BCE)

**Optymalizatory** - Oferuje różne optymalizatory do aktualizacji parametrów sieci podczas trenowania:
- Stochastyczny spadek gradientu (SGD)
- Adadelta
- Adam

## Licencja
Ten projekt jest dostępny na licencji MIT - zobacz plik LICENSE dla szczegółów.