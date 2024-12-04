# space for structured(!) information about task, progress background info
## To Dos
- Reference Plot (Meike)
- (done) x Axis aline (Marco)
- Fit Bug fix (Tom)
- Description (?)
- Wenn output dir schon vorhanden, dann soll beim drücken von Process Start nicht nochmal die komplette Datei ausgeführt werden (Marco)
- Wenn die Datei nicht vorhanden ist soll kein Error sondern eine Nachricht erscheinen (Marco)
- (done) Fiting Logic optimieren, session_state speichert die Pfade auch beim schließen der Datei (Marco)
- Nur einmal Start Process klicken und Daten einlesen, dann nur plots (future optimierung)
- (done) flip scale as seen in fatimas paper

# Notes from 27.11.
- [DONE] Putting everything in one plot,  Sleect between stacked and overlayed (P1 Marco)
- [DONE] Save the figures as pdf (P2 Marco)

- [DONE] Kinetic Plot, Dots instead of line (without a mean line) (P1 Tom)
- [DONE]Shifted frames by one  (P2 tom)
- [DONE] Caracterize how similar the noise is. (Mean, std of all the noise frames) (P2 Meike)
- [WILLNOTBEDONE] Line width of the lorenzians should not change (maybe 10%) constrain of for the fitting (P3 Tom)
- [DONE]Print the reference value (P1 Meike)
- [DONE] time with s, min h? Time Step instead of time (P1 Meike)
- reference factor mit in die anderen plots (P3 all) 
- [DONE] Font is too small (P1 Marco)
- [DONE] Capitalisation in the labels (P1 Marco)
- [DONE] Use the same label names on e.g y axis  (P1 Marco)
- [Done] file name of sum_fit should be fitted (P1 Tom)
- [DONE] put time to the beginning (in kenitcs.csv) (P1 Tom)
- [DONE] add kinetics in mol - another .csv file (P1 Meike)
- [DONE] add constant to fitting (P2 Tom)
- [DONE] Button to select also meta file and reference file (P1 Marco)
- [DONE] Start Processing at the end of selecting  (P1 Marco)
- [DONE] Progress bar after pressing the button (P2 Tom/Marco)
- [DONE] Select the model  (P1 Tom/Marco)
- [DONE] write instructions (P1 Marco), Monday
- [DONE] performance (Matplotlib, Plotly, Session_state) (P3 All)
- [DONE]Beide Modell zusammen ist aktuell mit bug (P1 Tom)
- [DONE] Save Reference Plot in correct directory (P1 Marcou) ??
- [DONE] Increase Font size for all plots (P1 All)
- [DONE] Writing is too slow (P2 All) -> Does actually not make a big difference. Plotly is slow and makes the biggest difference.
- [DONE] Is first time point of reference/contour plot 0 or 1?(P1 Meike, Marco)
- [DONE] Software beenden Button? [Marco]
- check whether it works with water+yeast  Reference (P2 Meike)
- [NotDOABLE]create streamlit .exe (P1 Meike)
- [DONE] plot titles dependent on file name?(P3 Marco/Tom)
- [DONE] Erstellen einer README Datei zur beschreibung wie die app gestartet werden kann. [Tom/Marco]
- Code verständlich kommentieren (alle)
- gemeinsam durchgehen und Code bereinigen (alle)

# Presentation

## Introduction
- Ziel des Projects (Spectren die wir analysiere möchten, und wir das integral berechnen möchten aber wegen overlaps ist das nicht möglich)
- (ansprechen das Overlaps in den Daten vorhanden sind und eien einfacher Peak detectioon nicht geht)
- Welche Daten & wie sehen sie aus


## Deepl Learning Approich kurz anspechen
- Problem der Umsetzung einfachen Machine Learning modell ansprechen (nicht gelabelt)

## Lorenzian Curve Fitting vorstellen

## Meikes Ableitungsmodel zur peak position finden 

## Diskussion wie das Modell durch Meikes Ansatz besser wird

## App vorstellen


## Questions 24. 09.24
- Do external conditions cause the same shift for each metabolite? Skewing of the spectra also possible?Is skewing the only problem?
- how exactly are the spectra calibrated/normalized? Each by itself or by average of whole time series, where do deviations originate from?
- --> should be normalized one by one, if not check with Fatima
- ---> first peak is taken as reference
- --> we can shift them



- relations of peaks to each other, when there are several characteristic peaks for one metabolite
- --> nicotinamite, peaks should have the same concentration
- --->xy around 1/10th of 

- different 'thresholds' for peaks (e.g. nicotinamide d4) --> why? different number of measurements?? maybe
-   
- more peaks recognized than mentioned in the metadata --> how should we deal with them? Calculate as normal but leave unlabeled?
- --> if truly there leave unlabeled
  
- How can we validate the model in the end?
- --> evaluate fitting of lorentzian curve, constraint is that amplitude for metabolite is the same
- --> dashboard --> visualization with original spectra, 'perfect'spectra with lorentzian curves and 'left over noise'
   
- when fourier transformed, is the data already smoothed? --> several measurements averaged thereby partly smoothed

- Why are .ser files not in metadata.

- concentration
- --> set boundaries for integral, when it is around noise level (so far)?? sigma would be fine
## Questions 17.09.24

### technical background
- How are the spectra produced? Difference MRI MRS? Different frequencies emitted or only different frequences "returned"?  
--> excited with strong frequency pulse (10 to 30 µs) short in time, high in amplitude. Spectra shows frequencies returned in audio file after Fourier transformation
  
### about the data

- What are the audiofiles?
  --> The Raw Files
  --> files how data is returned from MRS,before Fourier transformation; contains more information  
  --> maybe interesting  
---
- what is the goal of the project? --> identificatuion and quantification(via integral)? --> identification just via range of peaks?  
-->  identification of peaks and intergral, time series, show timeline of metabolites present  
--> some metabolites create several peaks with different heights (hopefully in fatima's manuscript)  
-->  ideally also visual output with marked peaks
--> Integral Diagrams of the Kinematics of the Peaks (up or down)

- physical reason for measurement, e.g. resonance frequency/ is identification peak dependent(or =) on reference frequency? (is reference frequency = exciting frequency)  
  --> ??? D4 classical refrence frequency for deuterium spectra, They used heavy water??  
  --> Water always cretes peak at 4.7 ppm  

- Why not use NMRQN, what are the differences? Why not just the existing models. For example BATMAN  
  --> maybe it is an option; not used yet. Works with static data, not time series  
  
---
- (Why ppm on x axis? shouldn't it be the frequency? What is chemical shift?(because y= intensity))  
--> relative shift of reference frequency, what is reference frequency? (Water or something else?)  
  
- what are characterististic curves/values? --> labeled data  
    - (file Data_description.xlsx feature ppm ? ) if multiple values, multiple                   
     states of same metabolite?--> ranges for metabolics ?  
    --> some metabolites cause several peaks (of different heights)  
    - in data, there are four peaks only three labels in data description (marco, Tom explain)  
    - what are possible shifts? what are they caused by and how are they recognized?  
    --> shifts can be caused by external influences such as ph/ temperature etc. shouldn't be more than 0.5ppm  
      
    - do water peaks need to be taken into account?  
      --> water creates strong peak around 4.7, if it is not present, let them know!!!  
 
---

- Fatima said she is calculating integrals. What integrals? what is she doing? -> Further Quantification  


  (- what is a healthy spectrum?? not labeling peaks but abnormalities)


## New questions:
- if every frame is normalized to water= 4.7 can there still be a shift of the peaks (>0.05)


---
## possibly interesting keywords (with info)
- Markov Chain Monte carlo algorithm ( what is it? mentioned in several papers)
- stochastic hill climbing algorithm (-:-)
- streamlit for visualization ( recommended by SuperMarco)


## interesting paper links / videos/ background information
BATMAN:    https://doi.org/10.1093/bioinformatics/bts308   
LCModel:   https://doi.org/10.1002/nbm.698   
NMRQNet:   https://doi.org/10.1101/2023.03.01.530642  
NMR-Onion: https://doi.org/10.1016/j.heliyon.2024.e36998


## Model 1 - Function Fitting
The distribution can be fitted as a linear combination of Lorentzian distributions:

$$
f(x, x_0, \gamma) = \frac{1}{\pi} \cdot \frac{\gamma}{(x - x_0)^2 + \gamma^2}, \quad x \in \mathbb{R}
$$

where the peak occurs at \( x_0 \), and the width is represented by the scale parameter \( \gamma \).

The overall spectrum is then expressed as a weighted sum of these Lorentzian distributions:

$$
\text{Spectrum}(x) = \sum_{i} w_i \cdot f_i(x, x_{0,i}, \gamma_i)
$$

## Model 2 - Peakfinder NumPy


# From workfile:
## Plan
16.11 ist Heute
19.11 Meeting:
    - Absprache wie soll das board aussehen(Welceh Plots? Wie sollten die Plots genau assehen(x, achse, y achse usw.))
    - Veränderung über die Zeit
    - Mindestanzahl datenpunkte für ppm finder(Signifikanz?)
    - Deep Learning Model vorstellen.
    - Wie soll die Applikation verpackt sein. Docker, exe, python file?
    - Metadaten. Wie müssen die Aussehen? Nächste Woche -> also nächsten    
    - Wann könnt Ihr unsere App und das fitting 'korrigiert' haben? -> Termin für 1. version

21.11 Übergabe an Tom spätestens und dann implementieren
24.11 Beta Dashboard 

26.11 Meeting: Dashboard Besprechung. Dauer länger
26.11 Start Dokumentation
26.11 Nächste Consulation(9 Uhr) - Deadline Programmieren, Consultatioj

3.12 

10.12

13.12 MoinCC Projektabgabe mit Präsi
19.12 Präsi FH
22.12 Deadline Abgabe Dokument




# Docker Befehle
sudo docker build -t streamlit-app .

sudo docker run -it --rm -p 8501:8501 \
  -v /home/generalpegasus/Documents/Programme/MoinCC-AI4metabolomics/Data:/app/input \
  -v /home/generalpegasus/Documents/Programme/MoinCC-AI4metabolomics/app/output:/app/output \
  streamlit-app


## Plan
16.11 ist Heute
19.11 Meeting:
    - Absprache wie soll das board aussehen(Welceh Plots? Wie sollten die Plots genau assehen(x, achse, y achse usw.))
    - Veränderung über die Zeit
    - Mindestanzahl datenpunkte für ppm finder(Signifikanz?)
    - Deep Learning Model vorstellen.
    - Wie soll die Applikation verpackt sein. Docker, exe, python file?
    - Metadaten. Wie müssen die Aussehen? Nächste Woche -> also nächsten    
    - Wann könnt Ihr unsere App und das fitting 'korrigiert' haben? -> Termin für 1. version

21.11 Übergabe an Tom spätestens und dann implementieren
24.11 Beta Dashboard 

26.11 Meeting: Dashboard Besprechung. Dauer länger
26.11 Start Dokumentation
26.11 Nächste Consulation(9 Uhr) - Deadline Programmieren, Consultatioj

3.12 

10.12

13.12 MoinCC Projektabgabe mit Präsi
19.12 Präsi FH
22.12 Deadline Abgabe Dokument


Fitting Outputs:
**app/**
- **README.md**  
  | Documentation for the application.

- **app.py**  
  | Main script to run the application.

- **.config**  
  | Configuration file for application settings.

- **curve_fitting.py**  
  | Script handling the curve fitting algorithms.

- **DataLoader.py**  
  | Script for loading and processing data.

- **output_dir/**  
  | Directory containing all output results.

  - **File_Name/** (A directory for each processed file)
    |-- **fitted_spectra/**
    |   |-- `file_name_1.csv`
    |   |-- `file_name_2.csv`
    |   |-- ...
    |
    |-- **difference_spectra/**
    |   |-- `file_name_1.csv`
    |   |-- `file_name_2.csv`
    |   |-- ...
    |
    |-- **individual_curves/**
    |   |-- `file_name_1.csv`
    |
    |-- `file_spectra_params.csv`
    |
    |-- `fitted_spectra_params_error.csv`
    |
    |-- `integral_spectra_over_time.csv`
