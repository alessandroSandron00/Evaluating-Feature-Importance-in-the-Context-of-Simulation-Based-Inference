# Feature Importance in Simulation-Based Inference of Cortical Circuit Parameters  

## 📌 Overview  
This project explores **feature importance** in **simulation-based inference (SBI)** for predicting **cortical circuit parameters** using **Local Field Potentials (LFPs)**. By leveraging **SHAP values**, we analyze how individual features contribute to SBI model predictions.  

## 🧠 Abstract  
Extracellular electrophysiology recordings, such as **LFPs**, provide key insights into **cortical dynamics**. However, understanding how these signals relate to **underlying neural populations** remains a challenge.  

In this study, we:  
✅ Generated a **large-scale dataset** (1M simulations) using a **spiking cortical microcircuit model**.  
✅ Applied **biophysics-based causal filters** to generate realistic LFP signals.  
✅ Extracted key features from simulated LFP data.  
✅ Trained an **SBI algorithm** to infer cortical circuit parameters.  
✅ Used **SHAP values** to assess the contribution of each feature to the model’s predictions.  

Our results demonstrate the effectiveness of **feature selection** in improving **SBI performance** for **neuroscientific inference**.  

## 🛠 Technologies & Tools  
- **Python**  
- **NumPy / SciPy**  
- **Brian2 / NEST** (Neural simulation frameworks)  
- **PyTorch** (for SBI training)  
- **SHAP** (for feature importance analysis)  
