import pandas as pd
import numpy as np
import pickle

# For predicting today's high change and today's low change use today's open change

with open("models/pred_high.pkl","rb") as file:
    pred_high = pickle.load(file)

with open("models/pred_low.pkl","rb") as file:
    pred_low = pickle.load(file)

# For predicting today's close change use today's open change, predicted today's high change , low change 
with open("models/pred_close.pkl","rb") as file:
    pred_close = pickle.load(file)

open_val = np.array([[-0.25]])


predicted_high = float(pred_high.predict(open_val))
predicted_low = float(pred_low.predict(open_val))
close_val = np.array([[predicted_low, predicted_high, open_val[0, 0]]])
predicted_close = float(pred_close.predict(close_val))

print("Predicted High : ",predicted_high)
print("Predicted Low : ",predicted_low)
print("Predicted Close : ",predicted_close)