import matplotlib.pyplot as plt
import seaborn as sns
from data_proccessing import *

# WIZUALIZACJA DANYCH
proc_data = GetData().processed_data
col_data_type = []

for col in proc_data.columns:
    if proc_data[col].dtype in ['int64','float64']:
        col_data_type.append('dane numeryczne')
    else:
        col_data_type.append('dane kategoryczne')


# plt.figure(figsize=(7,5))
# plt.grid()
# sns.countplot(x=col_data_type)
# plt.ylabel('liczba kolumn')
# plt.title('Zależność danych liczbowych i kategorycznych')
# plt.show()


# OUTLIERS
# train = proc_data[proc_data["TYPE"] == 'train']
train = GetData().processed_train
print(train)
top_features = train.corr()[['PRICE']].sort_values(by=['PRICE'],ascending=False)


plt.figure(figsize=(7,7))
sns.heatmap(top_features,cmap='rainbow',annot=True,annot_kws={"size": 10},vmin=-0.5)
plt.show()


# WARTOSCI UNIKALNE