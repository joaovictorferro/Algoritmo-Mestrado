import pandas as pd
import Completo_AG_IBOVESPA as CPI

data=pd.read_csv('https://raw.githubusercontent.com/joaovictorferro/DataSet-IBOVESPA/main/IBOVESPA/DataSet_IBOVESPA.csv')

CPI.start(data)