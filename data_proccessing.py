import os

import pandas as pd
import numpy as np
import re
import copy
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')

os.path.dirname(__file__)
sc = StandardScaler()
encode = LabelEncoder()

class GetData:

    def __init__(self):
        # upload data
        self.raw_train = self.initial_proccessing(pd.read_excel('train.xlsx'))
        self.raw_test = self.initial_proccessing(pd.read_excel('test.xlsx'))
        self.raw_train['TYPE'] = 'train'
        self.raw_test['TYPE'] = 'test'
        self.merged_data = pd.concat([self.raw_train, self.raw_test])
        self.processed_data = self.column_proccessing(self.merged_data)
        self.encoded_data = self.encode_data(self.processed_data)
        self.processed_train, self.processed_test = self.split_merged(self.encoded_data)
        self.X_train, self.X_test, self.y_train = self.scale_data(self.processed_train, self.processed_test)

    # def get_train(self):
    #     print("train getter called")
    #     return self.raw_train

    # def get_test(self):
    #     print("test getter called")
    #     return self.raw_test

    # def get_merged(self):
    #     print("merged getter called")
    #     return self.merged_data

    # def get_processed(self):
    #     print("process getter called")
    #     return self.processed_data

    # def get_encoded(self):
    #     print("encode getter called")
    #     return self.encoded_data

    # # def get_merged(self):
    # #     print("merged getter called")
    # #     return self.merged_data

    # def set_train(self, new_data):
    #     print("train setter called")
    #     self.raw_train = new_data

    # def set_test(self, new_data):
    #     print("test setter called")
    #     self.raw_test = new_data

    # def set_merged(self, new_data):
    #     print("merged setter called")
    #     self.merged_data = new_data   

    # def set_processed(self, new_data):
    #     print("process setter called")
    #     self.processed_data = new_data

    # def set_encoded(self, new_data):
    #     print("encoded setter called")
    #     self.encoded_data = new_data 

    # train = property(get_train, set_train)
    # test = property(get_test, set_test)
    # merged = property(get_merged, set_merged)


    def initial_proccessing(self, data: pd.DataFrame):        
        # replace characters
        data.columns = data.columns.str.upper().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

        # Delete synopsisi since it will not be used and can confuse outcome
        return data.drop(['SYNOPSIS'], axis=1)


    def column_proccessing(self, set: pd.DataFrame):

        dataset = copy.copy(set)

        # PUBLICATION & YEAR
        dataset['YEAR'] = dataset['EDITION'].str[-4:]
        # ed_year = [int(m_y[i][1].strip()) if m_y[i][1].isdigit() else 0 for i in range(len(m_y))]

        # Random publication year for some books
        dataset['YEAR'] = dataset['YEAR'].apply(lambda x: re.sub("[^0-9]", 'NA', x))
        dataset['YEAR'] = dataset['YEAR'].apply(lambda x: x.replace('NA', '0'))
        dataset['YEAR'] = dataset['YEAR'].astype(np.int16)

        dataset['AGE'] = 2022 - dataset['YEAR']

        dataset.loc[(dataset['YEAR'] == 0), 'YEAR'] = np.NaN
        avg_age = dataset['AGE'].mean()
        dataset.loc[(dataset['YEAR'].isnull()), 'AGE'] = avg_age

        # REVIEWS
        dataset['REVIEWS'] = dataset['REVIEWS'].apply(lambda x: x.split(' ')[0])
        dataset['REVIEWS'] = dataset['REVIEWS'].astype('float')

        # RATINGS
        dataset['RATINGS'] = dataset['RATINGS'].apply(lambda x: x.split(' ')[0])
        dataset['RATINGS'] = dataset['RATINGS'].apply(lambda x: int(x.replace(',', '')))

        # TITLE
        dataset['TITLE'] = dataset['TITLE'].str.upper()

        # EDITION
        dataset['EDITION'] = dataset['EDITION'].apply(lambda x: x.split(",")[0].strip().upper())

        # AUTHOR
        dataset['AUTHOR'] = dataset['AUTHOR'].str.upper()
        # max_aut = max([len(x.split(",")) for x in list(dataset['AUTHOR'])])

        authors = list(dataset['AUTHOR'])

        A1 = []
        A2 = []
        A3 = []
        A4 = []
        A5 = []
        A6 = []
        A7 = []

        for i in authors:
        
            try :
                A1.append(i.split(',')[0].strip().upper())
            except :
                A1.append('NONE')
                
            try :
                A2.append(i.split(',')[1].strip().upper())
            except :
                A2.append('NONE')
                
            try :
                A3.append(i.split(',')[2].strip().upper())
            except :
                A3.append('NONE')
                
            try :
                A4.append(i.split(',')[3].strip().upper())
            except :
                A4.append('NONE')
                
            try :
                A5.append(i.split(',')[4].strip().upper())
            except :
                A5.append('NONE')
                
            try :
                A6.append(i.split(',')[5].strip().upper())
            except :
                A6.append('NONE')
                
            try :
                A7.append(i.split(',')[6].strip().upper())
            except :
                A7.append('NONE')

        dataset["AUTHOR1"] = A1
        dataset["AUTHOR2"] = A2
        dataset["AUTHOR3"] = A3
        dataset["AUTHOR4"] = A4
        dataset["AUTHOR5"] = A5
        dataset["AUTHOR6"] = A6
        dataset["AUTHOR7"] = A7
        dataset = dataset.drop(['AUTHOR'], axis=1)

        # GENRE
        max_gen = max([len(x.split(",")) for x in list(dataset['GENRE'])])
        max_gen

        genres = list(dataset['GENRE'])

        G1 = []
        G2 = []

        for i in genres:

            try :
                G1.append(i.split(',')[0].strip().upper())
                
            except :
                G1.append('NONE')
                
            try :
                G2.append(i.split(',')[1].strip().upper())
            except :
                G2.append('NONE')

        dataset['GENRE1'] = G1
        dataset['GENRE2'] = G2
        dataset = dataset.drop(['GENRE'], axis=1)

        # CATEGORY
        max_cat = max([len(x.split(",")) for x in list(dataset['BOOKCATEGORY'])])
        max_cat

        category = list(dataset['BOOKCATEGORY'])

        C1 = []
        C2 = []

        for i in category:

            try :
                C1.append(i.split(',')[0].strip().upper())
                
            except :
                C1.append('NONE')
                
            try :
                C2.append(i.split(',')[1].strip().upper())
            except :
                C2.append('NONE')

        dataset['CATEGORY1'] = C1
        dataset['CATEGORY2'] = C2
        dataset = dataset.drop(['BOOKCATEGORY'], axis=1)

        # self.set_processed(dataset)
        return dataset

    def encode_data(self, dataset: pd.DataFrame):

        new_dataset = copy.copy(dataset)
        
        new_dataset['TITLE'] = encode.fit_transform(list(new_dataset['TITLE']))
        new_dataset['EDITION'] = encode.fit_transform(list(new_dataset['EDITION']))

        new_dataset['AUTHOR1'] = encode.fit_transform(list(new_dataset['AUTHOR1']))
        new_dataset['AUTHOR2'] = encode.fit_transform(list(new_dataset['AUTHOR2']))
        new_dataset['AUTHOR3'] = encode.fit_transform(list(new_dataset['AUTHOR3']))
        new_dataset['AUTHOR4'] = encode.fit_transform(list(new_dataset['AUTHOR4']))
        new_dataset['AUTHOR5'] = encode.fit_transform(list(new_dataset['AUTHOR5']))
        new_dataset['AUTHOR6'] = encode.fit_transform(list(new_dataset['AUTHOR6']))
        new_dataset['AUTHOR7'] = encode.fit_transform(list(new_dataset['AUTHOR7']))

        new_dataset['GENRE1'] = encode.fit_transform(list(new_dataset['GENRE1']))
        new_dataset['GENRE2'] = encode.fit_transform(list(new_dataset['GENRE2']))
        new_dataset['CATEGORY1'] = encode.fit_transform(list(new_dataset['CATEGORY1']))
        new_dataset['CATEGORY2'] = encode.fit_transform(list(new_dataset['CATEGORY2']))

        # self.set_encoded(new_dataset)
        return new_dataset

    def split_merged(self, dataset: pd.DataFrame):

        train = dataset[dataset["TYPE"] == 'train']
        test = dataset[dataset["TYPE"] == 'test']

        # self.set_train(train.drop(['TYPE'], axis=1))
        # self.set_test(test.drop(['TYPE'], axis=1))

        return train.drop(['TYPE'], axis=1), test.drop(['TYPE'], axis=1)


    def scale_data(self, train, test):

        X_train = train.drop(['PRICE'],axis = 1)
        y_train = train['PRICE'] #.values
        X_test = test.drop(["PRICE"], axis=1)
        # X_train.describe(include = 'all')

        # SCALE
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        # reshaping
        y_train = y_train.values.reshape((len(y_train), 1))
        y_train = sc.fit_transform(y_train)
        y_train = y_train.ravel()

        return X_train, X_test, y_train

if __name__ == "__main__":
    SetData = GetData()
    print(SetData.encoded_data)
    # print(SetData.column_proccessing(pd.concat([SetData.raw_train, SetData.raw_test])))