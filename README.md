# Task-agnostic

## ReadMe file for Task-agnostic subsequence-based representation learning project:

### Title: Learning task-agnostic and interpretable subsequence-based representation of time series and its applications in fMRI analysis


#### Demanded library:

 timesynth(for data creation); skimage;
 
 Keras >=2.0; Tensorflow >= 2.0; Numpy;

Files:

  A. Simulation_data.py
  
  This script is made for creating simulation time series data (1st half as training, 2nd half as testing time series) for time series reconstruction and de-noising task.
      
  B. Model.py
  
  This main code snippet contains the core programme on how we compute the lower-bound of Mutual information derived in Eq.3 (with tunable alpha).
      
  C. HCP.py
  
  [To be uploaded soon]
