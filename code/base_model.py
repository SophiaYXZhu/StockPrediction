import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import time
import pickle
import os
import torch.nn.init as init

torch.autograd.set_detect_anomaly(True)


def normalize_and_multiply_100(df):
    normalized_df = (df - df.mean()) / df.std()
    normalized_df *= 100
    return normalized_df

def prepare_data():
    print("Start...")
    st = time.time()
    if os.path.exists("filtered_rets.pkl"):
        df_30 = pickle.load(open("filtered_df_30.pkl", "rb"))
        df_daily = pickle.load(open("filtered_df_daily.pkl", "rb"))
        label = pickle.load(open("filtered_rets.pkl", "rb"))
        print("load from pkl: {}, {}, {}".format(len(df_30), len(df_daily), len(label)))
        return df_30, df_daily, label

    file_path = './data/stock_daily_adjusted_data_20160104_20231215.h5' # daily
    df_daily = normalize_and_multiply_100(pd.read_hdf(file_path)).astype('float32')
    print("df_daily", len(df_daily))
    
    file_path = './data/stock_adjvwap_data_20160104_20231215.h5'
    df = pd.read_hdf(file_path).astype('float32')
    df = df[["stock_adjvwap"]].copy()
    df.dropna(inplace=True)
    df = df.unstack()
    df.fillna(method="ffill",limit=20,inplace=True)
    rets = df.shift(-11) / df.shift(-1) - 1
    rets = rets.stack()
    rets = 100*rets
    rets = rets['stock_adjvwap'] # labels
    print("rets", len(rets))
    
    common_index = df_daily.index.intersection(rets.index)
    print("common_index", len(common_index))
    
    file_path = './data/stock_30min_adjusted_data_20160104_20231215.h5' 
    df_30 = normalize_and_multiply_100(pd.read_hdf(file_path)).astype('float32')
    print("df_30", len(df_30))
    df_30.dropna(inplace=True)
    print("after dropna df_30", len(df_30))
    df_30['date_only'] = df_30.index.get_level_values('trade_date').date
    filtered_df = df_30.groupby(["date_only", "stock_code"]).filter(lambda x: len(x) == 8)
    print("filtered_df_30 with 8 data points", len(filtered_df))
    filtered_df["date_only"]=pd.to_datetime(filtered_df['date_only'])
    filtered_df = filtered_df.reset_index()
    filtered_df.set_index(['date_only', 'stock_code'], inplace=True)
    common_index_date_only= common_index.intersection(filtered_df.index)
    print("common_index_date_only", len(common_index_date_only))
    filtered_df_daily = df_daily.loc[common_index_date_only]
    filtered_rets = rets.loc[common_index_date_only]
    filtered_df_30 = filtered_df.loc[common_index_date_only]
    print("After preprocessing: {}, {}, {}".format(len(filtered_df_30), len(filtered_df_daily), len(filtered_rets)))
    print("time taken : {}".format(time.time()-st))
    with open("filtered_df_30_with_date.pkl", "wb") as fw:
        pickle.dump(filtered_df_30, fw)
    with open("filtered_df_daily.pkl", "wb") as fw:
        pickle.dump(filtered_df_daily, fw)
    with open("filtered_rets.pkl", "wb") as fw:
        pickle.dump(filtered_rets, fw)
    filtered_df_30_final = filtered_df_30.reset_index()
    filtered_df_30_final = filtered_df_30_final.drop(columns=['date_only'])
    filtered_df_30_final.set_index(['trade_date', 'stock_code'], inplace=True)
    with open("filtered_df_30.pkl", "wb") as fw:
        pickle.dump(filtered_df_30_final, fw)    
    return filtered_df_30_final, filtered_df_daily, filtered_rets

class StockPredictionDataset(Dataset):
    def __init__(self, data1, data2, label, seq_length=20):
        st = time.time()

        # each instance is (,)
        # dataframe is (num_days, num_stocks_each_day, )
        self.label = []
        stocks = label.index.get_level_values('stock_code').unique()
        label = label.swaplevel('trade_date', 'stock_code').sort_index()
        print(len(stocks), len(label))
        for stock_code in stocks: 
            stock_df = label.loc[stock_code, :]
            dates = label.loc[stock_code, :].index.get_level_values('trade_date').unique()
            dates = dates[-250:]
            dates = dates[0::5]
            if len(dates) <= 30:
                continue
            for date_idx in range(seq_length, len(dates)): # TEST
                profit = stock_df.iloc[date_idx]
                self.label.append(profit)
        print("finished label, time taken:{}".format(time.time()-st))
        with open("dataset_label.pkl", "wb") as fw:
            pickle.dump(self.label, fw)
            
        # each instance is (20, 6)
        # dataframe is (num_days, num_stocks_each_day, 20, 6)
        self.data_daily = []
        stocks = data2.index.get_level_values('stock_code').unique()
        data2 = data2.swaplevel('trade_date', 'stock_code').sort_index()
        print(len(stocks), len(data2))
        for stock_code in stocks: 
            stock_df = data2.loc[stock_code, :]
            dates = data2.loc[stock_code, :].index.get_level_values('trade_date').unique()
            dates = dates[-250:]
            dates = dates[0::5]
            if len(dates) <= 30:
                continue
            for date_idx in range(seq_length, len(dates)):
                twenty_days = stock_df.iloc[date_idx - seq_length: date_idx, :]
                twenty_days = twenty_days.values.tolist()
                self.data_daily.append(twenty_days)
        print("finished df_daily, time taken:{}".format(time.time()-st))
        with open("dataset_daily.pkl", "wb") as fw:
            pickle.dump(self.data_daily, fw)
            
        # each instance is (20 * 8, 6)
        # dataframe is (num_days * 8, num_stocks_each_day, 20 * 8, 6)
        self.data_30 = []
        stocks = data1.index.get_level_values('stock_code').unique()
        data1 = data1.swaplevel('trade_date', 'stock_code').sort_index()
        print(len(stocks), len(data1))
        for stock_code in stocks: 
            stock_df = data1.loc[stock_code, :]
            dates = data1.loc[stock_code, :].index.get_level_values('trade_date').unique()
            dates = dates[-2000:]
            d = []
            for i in range(0, len(dates), 5*8):
                d.extend(dates[i:i+8])
            dates = d
            if len(dates) <= 30*8:
                continue
            for date_idx in range(seq_length*8, len(dates), 8): # eight 30mins's in a day
                twenty_days = stock_df.iloc[date_idx - seq_length * 8: date_idx, :]
                twenty_days = twenty_days.values.tolist()
                self.data_30.append(twenty_days)
        print("finished df_30, time taken:{}".format(time.time()-st))
        with open("dataset_30.pkl", "wb") as fw:
            pickle.dump(self.data_30, fw)
        print(len(self.data_daily), len(self.data_30), len(self.label))

        self.data_30, self.data_daily, self.label = torch.tensor(self.data_30).reshape(-1, seq_length*8, 6), torch.tensor(self.data_daily).reshape(-1, seq_length, 6), torch.tensor(self.label)
        print("complete stockdataset: {}".format(time.time()-st))
        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data_30[idx], self.data_daily[idx], self.label[idx]
    
class BiAGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiAGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(2 * hidden_size, 1, bias=False)
        self.fc = nn.Linear(2 * hidden_size, output_size)
        self.init_weights()
        
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param.data)
            elif 'bias' in name:
                init.constant_(param.data, 0.0)

    def forward(self, x):
        gru_output, _ = self.gru(x)
        attention_scores = self.attention(gru_output)
        attention_weights = torch.softmax(attention_scores, dim=1)
        attended_output = torch.sum(attention_weights * gru_output, dim=1)
        output = self.fc(attended_output)
        return output

class LargeModel(nn.Module):
    def __init__(self, input_size = 6, mid_size = 5, output_size = 1, hidden_size = 64):
        super(LargeModel, self).__init__()
        self.agru_30 = BiAGRU(input_size=input_size, hidden_size = hidden_size, output_size=mid_size)
        self.agru_daily = BiAGRU(input_size=input_size, hidden_size = hidden_size, output_size=mid_size)
        self.fnn = nn.Linear(2*mid_size, output_size)

    def forward(self, data_30, data_daily):
        output_30 = self.agru_30(data_30) # output = (batch_size, 5)
        # print('out_30:', output_30)
        output_daily = self.agru_30(data_daily) # output = (batch_size, 5)
        # print('out_daily:', output_daily)
        x = torch.cat((output_30, output_daily), dim=1)
        output = self.fnn(x) # output = (batch_size, 1)
        return output


if __name__ == "__main__":
    input_size = 6 
    mid_size = 5
    output_size = 1
    hidden_size = 64  
    batch_size = 64
    try:
        print("Start loading StockPredictionDataset")
        data=pickle.load(open("StockPredictionDataset.pkl", "rb"))
    except:
        print("Start prepare_data")
        df_30, df_daily, rets = prepare_data()
        print("start to generate StockPredictionDataset")
        st = time.time()
        data = StockPredictionDataset(df_30, df_daily, rets)
        print("Got StockPredictionDataset. time taken:{}".format(time.time()-st))
        with open("StockPredictionDataset.pkl", "wb") as fw:
            pickle.dump(data, fw)

    print("after loading dataset", len(data))
    model = LargeModel(input_size, mid_size, output_size, hidden_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    total_size = len(data)
    train_size = int(0.75 * total_size)
    val_size = total_size - train_size
    print(total_size, train_size, val_size)
    train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    print("Start training")
    num_epochs = 50
    best_val_loss = np.inf
    start_time = time.time()
    for epoch in range(num_epochs):
        print(epoch)
        st = time.time()
        # Training
        model.train()
        total_loss = 0
        for batch_num, train_batch in enumerate(train_loader):
            input_30, input_daily, target = train_batch
            input_30, input_daily, target = input_30.to(device), input_daily.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(input_30, input_daily).reshape(-1,)
            loss = criterion(output, target)
            # print(loss)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_loss / (batch_num + 1)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_batch in val_loader:
                input_30, input_daily, target = val_batch
                input_30, input_daily, target = input_30.to(device), input_daily.to(device), target.to(device)
                output = model(input_30, input_daily)
                val_loss += criterion(output, target).item()
            avg_val_loss = val_loss / len(val_loader)

        # Logging
        print(f"At epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, time taken:{time.time()-st}")
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "./model/best_biAGRU.pth")
            print(f"At epoch {epoch}, saving best model with Validation Loss: {avg_val_loss:.4f}")
    print("Total time taken:{}".format(time.time()-start_time))