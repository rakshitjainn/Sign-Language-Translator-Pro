import pandas as pd
data = pd.read_csv('hand_data.csv')
print(data['class_id'].value_counts().sort_index())