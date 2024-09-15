# Data Description (us)
This directory contains 2 different types of data. The metadata in the file 'Data_description.xlsx' contains information about the individual experiments. All other csv files correspond to one experiment. The csv files contains the time dependent data.

## Metadata: Datadescription.xlsx
The meaning of the contained columns in that csv:


| Column Name                          | Description                                                                                          |
|--------------------------------------|------------------------------------------------------------------------------------------------------|
| **ID**                               | A unique identifier for each record or experiment.                                                  |
| **File**                             | The name or identifier of the file where the MRI or NMR data is stored.                             |
| **Expt_name**                        | The name of the experiment, detailing the type or conditions of the experiment.                     |
| **TR[s]**                            | Repetition Time in seconds, the time interval between successive pulse sequences. Multiple scans are combined to reduce variance.                   |
| **NS**                               | Number of Scans, indicating how many scans or measurements were taken for one colum in the data files. This is done for an averaging process to reduce variation                            |
| **TRtotal[s]**                       | Total Repetition Time in seconds, calculated as `TR[s] * NS`. This should be the average point in time for one column?                                      |
| **Substrate_name**                   | The name of the substrate used in the experiment.                                                   |
| **Substrate_N_D**                    |?      |
| **Substrate_mM**                     | Concentration of the substrate in millimolar (mM).                                                   |
| **Substrate_ppm**                    | Concentration of the substrate in parts per million (ppm).                                           |
| **pH_before**                        | pH level of the solution before the experiment or treatment.                                         |
| **pH_after**                         | pH level of the solution after the experiment or treatment.                                          |
| **Reaction temperature (Kelvin)**    | Temperature at which the reaction or experiment was conducted, measured in Kelvin.                  |
| **Yeast_suspension**                 | Information on the concentration or volume of yeast suspension used.                                |
| **Substrate_solvent**                | Solvent in which the substrate was dissolved (e.g. PBS).                                 |
| **Substrate_mM_added**               | Amount of substrate added, measured in millimolar (mM).                                             |
| **Water_ppm**                        | Concentration of water in the solution, measured in parts per million (ppm).                        |
| **Metabolite_1** to **Metabolite_5** | Names or identifiers of the identified metabolites                                   |
| **Metabolite_1_ppm** to **Metabolite_5_ppm** | Concentration of each metabolite in parts per million (ppm).                                  |

## Data Files
Metadata can be extracted from the just mentioned file. The first column shows the Chemical shift in ppm and therefore represents the x axis. The following columns represent the measured spectras for different points in time. The time difference between the columns and measurements can be extracted from the metadata file. 

| Chemical Shift (ppm) | Time 1 | Time 2 | Time 3 | ... |
|----------------------|--------|--------|--------|-----|
| X1                   | Y1,1   | Y1,2   | Y1,3   | ... |
| X2                   | Y2,1   | Y2,2   | Y2,3   | ... |
| X3                   | Y3,1   | Y3,2   | Y3,3   | ... |
| ...                  | ...    | ...    | ...    | ... |

The spectra are averaged NS times, with repetition TR and the time between middle of spectra is TR*NS which is provided in Data_description.xlsx.
