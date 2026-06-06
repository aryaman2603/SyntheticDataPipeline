from folktables import ACSDataSource, ACSIncome

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["CA"], download=True)

features, label, _ = ACSIncome.df_to_numpy(acs_data)

import pandas as pd
import numpy as np

df = pd.DataFrame(features, columns=ACSIncome.features)
df['PINCP'] = label  # income label (binary: >50k or not)

df.to_csv('data/raw/acs_income_ca_2018.csv', index=False)