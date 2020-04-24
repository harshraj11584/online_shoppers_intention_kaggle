import pandas as pd

d = pd.read_csv('/home/harsh/Desktop/Untitled spreadsheet - Sheet1.csv')

d['HasRevenue'] = d['HasRevenue']/4.0


d.to_csv('ok.csv')