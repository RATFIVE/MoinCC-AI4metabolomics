## space for structured(!) information about task, progress background info

test

## Questions

### technical background
- How are the spectra produced? Difference MRI MRS? Different frequencies emitted or only different frequences "returned"?  
--> excited with strong frequency pulse (10 to 30 µs) short in time, high in amplitude. Spectra shows frequencies returned in audio file after Fourier transformation
  
### about the data

- What are the audiofiles?  
  --> files how data is returned from MRS,before Fourier transformation; contains more information  
  --> maybe interesting  
---
- what is the goal of the project? --> identificatuion and quantification(via integral)? --> identification just via range of peaks?  
-->  identification of peaks and intergral, time series, show timeline of metabolites present  
--> some metabolites create several peaks with different heights (hopefully in fatima's manuscript)  
-->  ideally also visual output with marked peaks  

- physical reason for measurement, e.g. resonance frequency/ is identification peak dependent(or =) on reference frequency? (is reference frequency = exciting frequency)  
  --> ??? D4 classical refrence frequency for deuterium spectra, They used heavy water??  
  --> Water always cretes peak at 4.7 ppm  

- Why not use NMRQN, what are the differences? Why not just the existing models. For example BATMAN  
  --> maybe it is an option; not used yet. Works with static data, not time series  
  
---
- (Why ppm on x axis? shouldn't it be the frequency? What is chemical shift?(because y= intensity))  
--> relative shift of reference frequency, what is reference frequency?  
  
- what are characterististic curves/values? --> labeled data  
    - (file Data_description.xlsx feature ppm ? ) if multiple values, multiple                   
     states of same metabolite?--> ranges for metabolics ?  
    --> some metabolites cause several peaks (of different heights)  
    - in data, there are four peaks only three labels in data description (marco, Tom explain)  
    - 
    - what are possible shifts? what are they caused by and how are they recognized?  
    --> shifts can be caused by external influences such as ph/ temperature etc. shouldn't be more than 0.5ppm  
      
    - do water peaks need to be taken into account?  
      --> water creates strong peak around 4.7, if it is not present, let them know!!!  
 
---

- Fatima said she is calculating integrals. What integrals? what is she doing? -> Further Quantification  


  (- what is a healthy spectrum?? not labeling peaks but abnormalities)





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



