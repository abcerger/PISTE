
import csv
from torch.utils.data.sampler import  WeightedRandomSampler
import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
import copy
import random
import os


# split data
def split_data(data):
    data_len = data['y'].count()
    split1 = int(data_len * 0.45)
    split2 = int(data_len * 0.55)
    print('split1', split1)
    print('split2', split2)

    train_data = data[:split1]
    test_data = data[split1:split2]
    shadow_data = data[split2:]
    return train_data,  test_data,  shadow_data

def credit_split_data(data):
    data_len = data['Y'].count()
    split1 = int(data_len * 0.45)
    split2 = int(data_len * 0.55)
    train_data = data[:split1]
    test_data = data[split1:split2]
    shadow_data = data[split2:]
    return train_data,  test_data, shadow_data

def census_split_data(data):
    data_len = len(data)
    print('data_len', data_len)
    split1 = int(data_len * 0.5)
    train_data = data[:split1]
    shadow_data = data[split1:]
    return train_data,  shadow_data


# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))



def records_path(acti, dataset, number_client, num_cutlayer):
    csvFile_1 = open(f'Results/{dataset}/num_client{number_client}/VFL_client1_{acti}_c{num_cutlayer}.csv', 'w+')
    writer_1 = csv.writer(csvFile_1)
    csvFile_2 = open(f'Results/{dataset}/num_client{number_client}/VFL_client2_{acti}_c{num_cutlayer}.csv', 'w+')
    writer_2 = csv.writer(csvFile_2)
    return writer_1, writer_2



def load_data(dataset, batch_size):
    if dataset == 'bank_marketing':
        processed_data = 'Data/processed_data/bank-additional-full.csv'
        data = pd.read_csv(processed_data)
        data['age']=(data['age']-data['age'].min())/(data['age'].max()-data['age'].min())
        data['duration']=(data['duration']-data['duration'].min())/(data['duration'].max()-data['duration'].min())
        data['campaign']=(data['campaign']-data['campaign'].min())/(data['campaign'].max()-data['campaign'].min())
        data['pdays']=(data['pdays']-data['pdays'].min())/(data['pdays'].max()-data['pdays'].min())
        data['previous']=(data['previous']-data['previous'].min())/(data['previous'].max()-data['previous'].min())
        data['emp.var.rate']=(data['emp.var.rate']-data['emp.var.rate'].min())/(data['emp.var.rate'].max()-data['emp.var.rate'].min())
        data['cons.price.idx']=(data['cons.price.idx']-data['cons.price.idx'].min())/(data['cons.price.idx'].max()-data['cons.price.idx'].min())
        data['cons.conf.idx']=(data['cons.conf.idx']-data['cons.conf.idx'].min())/(data['cons.conf.idx'].max()-data['cons.conf.idx'].min())
        data['euribor3m']=(data['euribor3m']-data['euribor3m'].min())/(data['euribor3m'].max()-data['euribor3m'].min())
        data['nr.employed']=(data['nr.employed']-data['nr.employed'].min())/(data['nr.employed'].max()-data['nr.employed'].min())
        for i in range(len(data)):
          if (data['poutcome_nonexistent'][i]):
            data['poutcome_failure'][i]=2
          if(data['poutcome_success'][i]):
            data['poutcome_failure'][i]=3
          if (data['job_blue-collar'][i]):
            data['job_admin.'][i]=2
          if(data['job_entrepreneur'][i]):
            data['job_admin.'][i]=3
          if (data['job_housemaid'][i]):
            data['job_admin.'][i]=4
          if(data['job_management'][i]):
            data['job_admin.'][i]=5
          if(data['job_retired'][i]):
            data['job_admin.'][i]=6
          if(data['job_self-employed'][i]):
            data['job_admin.'][i]=7
          if(data['job_services'][i]):
            data['job_admin.'][i]=8
          if(data['job_student'][i]):
            data['job_admin.'][i]=9
          if(data['job_technician'][i]):
            data['job_admin.'][i]=10
          if (data['job_unemployed'][i]):
            data['job_admin.'][i]=11
          if(data['marital_married'][i]):
            data['marital_divorced'][i]=2
          if(data['marital_single'][i]):
            data['marital_divorced'][i]=3
          if(data['contact_telephone'][i]):
            data['contact_cellular'][i]=2
          if (data['month_aug'][i]):
            data['month_apr'][i]=2
          if(data['month_dec'][i]):
            data['month_apr'][i]=3
          if (data['month_jul'][i]):
            data['month_apr'][i]=4
          if(data['month_jun'][i]):
            data['month_apr'][i]=5
          if(data['month_mar'][i]):
            data['month_apr'][i]=6
          if(data['month_may'][i]):
            data['month_apr'][i]=7
          if(data['month_nov'][i]):
            data['month_apr'][i]=8
          if(data['month_oct'][i]):
            data['month_apr'][i]=9
          if(data['month_sep'][i]):
            data['month_apr'][i]=10
          if (data['day_of_week_mon'][i]):
            data['day_of_week_fri'][i]=2
          if(data['day_of_week_thu'][i]):
            data['day_of_week_fri'][i]=3
          if (data['day_of_week_tue'][i]):
            data['day_of_week_fri'][i]=4
          if(data['day_of_week_wed'][i]):
            data['day_of_week_fri'][i]=5
        data['default']=(data['default']+1)
        data['housing']=(data['housing']+1)
        data['loan']=(data['loan']+1)
        train_data, test_data, shadow_data = split_data(data)

        numeric_attrs = ["nr.employed", "pdays",  "previous",  "duration", "campaign",                  "job_admin.",  "loan",  "education", "housing", "contact_cellular",
                         "emp.var.rate", "euribor3m",   "age",   "cons.price.idx","cons.conf.idx",   "month_apr",  "marital_divorced", "poutcome_failure", "default", "day_of_week_fri"]

        trainfeatures = torch.tensor(np.array(train_data[numeric_attrs])).float()
        trainlabels = torch.tensor(np.array(train_data['y'])).long()
        testfeatures = torch.tensor(np.array(test_data[numeric_attrs])).float()
        testlabels = torch.tensor(np.array(test_data['y'])).long()

        shadowfeatures = torch.tensor(np.array(shadow_data[numeric_attrs])).float()
        shadowlabels = torch.tensor(np.array(shadow_data['y'])).long()
    
    if dataset == 'credit':
        processed_data = 'Data/default of credit card clients.xls'
        data = pd.read_excel(processed_data)
        data.drop(index=[0], inplace=True)
        data.drop(0, axis=1, inplace=True)
        data['X1'] = (data['X1'] -data['X1'].min())/(data['X1'].max()-data['X1'].min())
        data['X5'] = (data['X5'] -data['X5'].min())/(data['X5'].max()-data['X5'].min())
        data['X12'] = (data['X12'] -data['X12'].min())/(data['X12'].max() -data['X12'].min())
        data['X13'] = (data['X13'] -data['X13'].min())/(data['X13'].max() -data['X13'].min())
        data['X14'] = (data['X14'] -data['X14'].min())/(data['X14'].max() -data['X14'].min())
        data['X15'] = (data['X15'] -data['X15'].min())/(data['X15'].max() -data['X15'].min())
        data['X16'] = (data['X16'] -data['X16'].min())/(data['X16'].max() -data['X16'].min())
        data['X17'] = (data['X17'] -data['X17'].min())/(data['X17'].max() -data['X17'].min())
        data['X18'] = (data['X18'] -data['X18'].min())/(data['X18'].max() -data['X18'].min())
        data['X19'] = (data['X19'] -data['X19'].min())/(data['X19'].max() -data['X19'].min())
        data['X20'] = (data['X20'] -data['X20'].min())/(data['X20'].max() -data['X20'].min())
        data['X21'] = (data['X21'] -data['X21'].min())/(data['X21'].max() -data['X21'].min())
        data['X22'] = (data['X22'] -data['X22'].min())/(data['X22'].max() -data['X22'].min())
        data['X23'] = (data['X23'] -data['X23'].min())/(data['X23'].max() -data['X23'].min())
        # relabel EDUCATION information
        fil = (data.X3 == 5) | (data.X3 == 6) | (data.X3 == 0)
        data.loc[fil, 'X3'] = 4
        # relabel MARRIAGE information
        data.loc[data.X4 == 0, 'X4'] = 3
        data['X6']=data['X6']+3
        data['X7']=data['X7']+3
        data['X8']=data['X8']+3
        data['X9']=data['X9']+3
        data['X10']=data['X10']+3
        data['X11']=data['X11']+3
        for i in range(len(data)):
            if (data['X3'][i+1]==4):
                data.drop(i+1,inplace=True)
        train_data, test_data, shadow_data = credit_split_data(data)
        numeric_attrs = ['X18', 'X17', 'X12', 'X16', 'X20', 'X19', 'X15', 'X2', 'X3', 'X4', 'X8', 'X9',
                         'X1',  'X5',  'X13', 'X21', 'X14', 'X23', 'X22', 'X10',  'X6', 'X7', 'X11' ]
        trainfeatures = torch.tensor(np.array(train_data[numeric_attrs]).astype(float)).float()
        trainlabels = torch.tensor(np.array(train_data['Y']).astype(int)).long()
        testfeatures = torch.tensor(np.array(test_data[numeric_attrs]).astype(float)).float()
        testlabels = torch.tensor(np.array(test_data['Y']).astype(int)).long()

        shadowfeatures = torch.tensor(np.array(shadow_data[numeric_attrs]).astype(float)).float()
        shadowlabels = torch.tensor(np.array(shadow_data['Y']).astype(int)).long()

    if dataset == 'census':
        processed_data = 'Data/census/census_income_train.csv'
        train_data = pd.read_csv(processed_data)
        processed_testdata = 'Data/census/census_income_test.csv'
        test_data = pd.read_csv(processed_testdata)
        train_data, shadow_data = census_split_data(train_data)

        numeric_attrs=['age','capital gains','capital losses','divdends from stocks','num persons worked for employer','wage per hour', 
                       'tax filer status', 'veterans benefits', 'marital status',
                        'sex', 'full or part time employment stat', 'enrolled in edu inst last wk', 'race', 'own business or self employed',
                       'member of a labor union', 'citizenship', 'live in this house 1 year ago', 'fill inc questionnaire for veterans admin', 'year', 

        'weeks worked in year', 'occupation code', 'education', 'major occupation code', 'industry code', 'major industry code', 'detailed household and family stat', 'detailed household summary in household',  'class of worker', 'family members under 18',
        'country of birth father', 'country of birth mother', 'hispanic origin', 'country of birth self', 
        'migration code-change in msa', 'migration code-change in reg',
        'migration code-move within reg', 'state of previous residence','reason for unemployment', 'migration prev res in sunbelt', 'region of previous residence']

        trainfeatures = torch.tensor(np.array(train_data[numeric_attrs]).astype(float)).float()
        trainlabels = torch.tensor(np.array(train_data['income']).astype(int)).long()
        testfeatures = torch.tensor(np.array(test_data[numeric_attrs]).astype(float)).float()
        testlabels = torch.tensor(np.array(test_data['income']).astype(int)).long()
        shadowfeatures = torch.tensor(np.array(shadow_data[numeric_attrs]).astype(float)).float()
        shadowlabels = torch.tensor(np.array(shadow_data['income']).astype(int)).long()
    
    # Sample data
    num_train_0 = sum(i == 0 for i in trainlabels)
    num_train_1 = sum(i == 1 for i in trainlabels)

    num_test_0 = sum(i == 0 for i in testlabels)
    num_test_1 = sum(i == 1 for i in testlabels)
    
    num_shadow_0 = sum(i == 0 for i in shadowlabels)
    num_shadow_1 = sum(i == 1 for i in shadowlabels)

    a = int(num_train_0/num_train_1)
    b = int(num_test_0/num_test_1)
    c = int(num_shadow_0/num_shadow_1)
    print('a', a)
    print('b', b)
    print('c', c)
    dataset_train = Data.TensorDataset(trainfeatures, trainlabels)
    weights_train = [a if label == 1 else 1 for data, label in dataset_train]
    sampler_train = WeightedRandomSampler(weights_train, num_samples=int(num_train_1)*2, replacement=False)
    index_train =copy.deepcopy(list(sampler_train))
    train_iter = Data.DataLoader(
      dataset=dataset_train,  
      batch_size=batch_size,  
      sampler=index_train,
      drop_last=True,
      shuffle=False,
    )

    dataset_test = Data.TensorDataset(testfeatures, testlabels)
    weights_test = [b if label == 1 else 1 for data, label in dataset_test]
    sampler_test = WeightedRandomSampler(weights_test, num_samples=int(num_test_1)*2, replacement=False)
    index_test = copy.deepcopy(list(sampler_test))
    test_iter = Data.DataLoader(
      dataset=dataset_test, 
      batch_size=batch_size, 
      sampler=index_test,
      drop_last=True,
      shuffle=False,
    )

    dataset_shadow = Data.TensorDataset(shadowfeatures, shadowlabels)
    weights_shadow = [c if label == 1 else 1 for data, label in dataset_shadow]
    sampler_shadow = WeightedRandomSampler(weights_shadow, num_samples=int(num_shadow_1)*2, replacement=False)
    index_shadow = copy.deepcopy(list(sampler_shadow))
    shadow_iter = Data.DataLoader(
      dataset=dataset_shadow, 
      batch_size=batch_size, 
      sampler=index_shadow,
      drop_last=True,
      shuffle=False,
    )

    size_train = len(sampler_train)
    size_test = len(sampler_test)
    size_shadow = len(sampler_shadow)

    return train_iter, test_iter, shadow_iter, size_train, size_test, size_shadow



def records_path(save_path, acti, dataset,  num_cutlayer):

    csvFile_1 = open(os.path.join(save_path, f'VFL_client1_{acti}_{dataset}_c{num_cutlayer}.csv'), 'w+')
    writer_1 = csv.writer(csvFile_1)

    csvFile_2 = open(os.path.join(save_path, f'VFL_client2_{acti}_{dataset}_c{num_cutlayer}.csv'), 'w+')
    writer_2 = csv.writer(csvFile_2)

    return writer_1, writer_2
def records_shadow_path(save_path, acti, dataset,num_cutlayer):

  csvFile_1 = open(os.path.join(save_path, f'VFL_client1_{acti}_{dataset}_c{num_cutlayer}_shadow.csv'), 'w+')
  writer_1 = csv.writer(csvFile_1)

  csvFile_2 = open(os.path.join(save_path, f'VFL_client2_{acti}_{dataset}_c{num_cutlayer}_shadow.csv'), 'w+')
  writer_2 = csv.writer(csvFile_2)

  return writer_1, writer_2