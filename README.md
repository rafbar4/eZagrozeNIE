# e-ZagrożeNIE - Road Risk Platform / Platforma Zagrożeń Drogowych

## Link to the Website:  https://e-zagrozenie.streamlit.app

## English Version

### Project Description
eZagrozeNIE is a web application built with Python and the Streamlit framework, designed to analyze road accidents on specific routes within the Lubuskie Voivodeship. The system allows users to calculate routes between two points and checks historical accident data within a safety buffer along those paths.

### Key Features
* Route Planning: Integration with the OSRM API to provide primary and alternative routing options.
* Accident Analysis: Automated searching for historical road incidents within a safety buffer along the selected route.
* Interactive Map: Visualization of routes and accident clusters using the Folium library.
* Risk Prediction: Utilizes a Logistic Regression model (scikit-learn) to calculate the probability of an accident based on time of day, day of the week, and month.
* Detailed Data View: Toggleable data tables showing specific incident details such as town, street, and distance from the route.

### Tech Stack
* Language: Python 3.x
* Frontend/Hosting: Streamlit Cloud
* Maps and Geospatial: Folium, Streamlit-Folium
* Data Processing: Pandas, Numpy
* Machine Learning: Scikit-learn (Logistic Regression)
* Data Source: Historical road accident dataset (CSV format)

* To use the dataset for the Lubuskie Voivodeship covering the years 2018–2024, change the filename in e-zagrozenie.py to **dane_wypadki_2018_2024.csv**; however, please note that the application may run slower as a result.
---

## Wersja Polska

## Link do storny internetowej:  https://e-zagrozenie.streamlit.app

### Opis Projektu
e-ZagrożeNIE to aplikacja webowa stworzona w języku Python z wykorzystaniem frameworka Streamlit, służąca do analizy wypadków drogowych na trasach w województwie lubuskim. System pozwala użytkownikowi wyznaczyć trasę między dwoma punktami i sprawdza historyczne dane o zdarzeniach drogowych w buforze bezpieczeństwa wzdłuż wybranej trasy.

### Główne Funkcje
* Wyznaczanie tras: Integracja z API OSRM w celu dostarczenia trasy głównej oraz opcji alternatywnych.
* Analiza wypadków: Automatyczne wyszukiwanie historycznych zdarzeń drogowych wzdłuż wyznaczonej trasy.
* Interaktywna mapa: Wizualizacja tras oraz klastrów wypadków przy użyciu biblioteki Folium.
* Predykcja ryzyka: Wykorzystanie modelu regresji logistycznej (scikit-learn) do obliczenia prawdopodobieństwa wypadku w oparciu o porę dnia, dzień tygodnia oraz miesiąc.
* Szczegółowy widok danych: Możliwość wyświetlenia tabeli z konkretnymi szczegółami zdarzeń (miejscowość, ulica, odległość od trasy).

### Wykorzystane Technologie
* Język: Python 3.x
* Frontend/Hosting: Streamlit Cloud
* Mapy i dane przestrzenne: Folium, Streamlit-Folium
* Przetwarzanie danych: Pandas, Numpy
* Uczenie maszynowe: Scikit-learn (Logistic Regression)
* Źródło danych: Historyczny zbiór danych o wypadkach drogowych (format CSV)

* Aby korzystać z danych dla województwa lubuskiego z lat 2018–2024, należy w pliku e-zagrozenie.py zmienić nazwę wczytywanego pliku na **dane_wypadki_2018_2024.csv**, jednak miej na uwadze, że aplikacja może wtedy działać wolniej.
---

### Local Installation / Instalacja lokalna

1. Clone the repository / Sklonuj repozytorium:
   git clone https://github.com/rafbar4/eZagrozeNIE.git

2. Install dependencies / Zainstaluj biblioteki:
   pip install -r requirements.txt

3. Run the app / Uruchom aplikację:
   streamlit run e-zagrozenie.py

Project developed as part of university coursework at the Faculty of Mathematics and Computer Science (WMI), Adam Mickiewicz University in Poznan.
Projekt przygotowany w ramach zajęć na Wydziale Matematyki i Informatyki (WMI) Uniwersytetu im. Adama Mickiewicza w Poznaniu.
