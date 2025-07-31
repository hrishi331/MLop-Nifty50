# This gives prediction based on change in open and each prediction further depends upon previous
# e.g. it predicts high and low change from open then predicts 
# change in close from predicted high, predicted low and change in open 

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

open_val = np.array([[-0.12]])


predicted_high = float(pred_high.predict(open_val))
pred_high_range = (predicted_high-0.63,predicted_high+0.63)
predicted_low = float(pred_low.predict(open_val))
pred_low_range = (predicted_low-0.64,predicted_low+0.64)
close_val = np.array([[predicted_low, predicted_high, open_val[0, 0]]])
predicted_close = float(pred_close.predict(close_val))
pred_close_range = (predicted_close-0.6,predicted_close+0.6)

print("Predicted High Range : ",pred_high_range)
print("Predicted Low Range : ",pred_low_range)
print("Predicted Close Range : ",pred_close_range)