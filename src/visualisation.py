import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def importance_graph(feature_names, importances):

  model_coefficients = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
  model_coefficients = model_coefficients.sort_values(by='Importance', ascending=False)   
  plt.figure(figsize=(10, 6))
  sns.barplot(x='Importance', y='Feature', data=model_coefficients) 
