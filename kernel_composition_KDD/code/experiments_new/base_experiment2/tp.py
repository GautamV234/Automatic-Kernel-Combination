# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns

# # # Load the group_activity.csv
# # df = pd.read_csv('/home/gautam.pv/nlim/kernel_composition_KDD/code/experiments_new/Group_activity.csv')
# # print(df.head())
# # randomized_tratment = df['randomized_treatment']
# # purchase_decision = df['purchase_decision']
# # print(randomized_tratment)
# # purchase_counts = df.groupby('randomized_treatment')['purchase_decision'].sum()
# # total_counts = df.groupby('randomized_treatment')['purchase_decision'].count()
# # purchase_percentages = purchase_counts / total_counts * 100
# # recoder = {1:'Underdog',2:'Proficient Worker',3:'Chatbot W/O',4:'Chatbot WBD', 5:'Chatbot WAD',6:'Chatbot ADD'}
# # # Create a bar plot of the purchase decisions for each treatment group
# # plt.bar(purchase_percentages.index, purchase_counts.values)
# # # plt.xticks(purchase_percentages.index, [recoder[i] for i in purchase_percentages.index])
# # plt.xlabel('Treatment Group')
# # plt.ylabel('Purchase Counts')
# # plt.title('Purchase Decision by Treatment Group')
# # #  add legend and beautify the plot
# # plt.tight_layout()
# # # beautify
# # plt.grid()
# # plt.legend()


# # # plt.show()
# # plt.savefig('/home/gautam.pv/nlim/kernel_composition_KDD/code/experiments_new/base_experiment2/tp2.png')


# #  USE LESS CODE BELOW ONLY FOR TESTING DELETE LATER AND UNCOMMENT ABOVE CODE
# import torch
# import torch.nn as nn

# # fix seed
# # torch.manual_seed(0)
# # m = nn.Linear(20,30)
# # input = torch.randn(128,40, 20)
# # output = m(input)
# # # print(output.size())
# # seq_len = 5
# # N = 8
# # # print(torch.arange(0,seq_len).shape)
# # pos = torch.arange(0,seq_len).expand(N,seq_len).float()
# # # print(pos.shape)
# # # print(pos)
# # v = torch.arange(0,10)
# # # add a dimension
# # v = v.unsqueeze(1)
# # v = v.unsqueeze(1)
# # print(f"v shape: {v.shape}")
# # # print(v)
# # x = nn.Embedding(10, 3)
# # print(x(v).shape)
# # # print(x(v))


# max_len = 50
# embed_size = 7
# src_vocab_size = 5
# seq_len = 2
# N = 3
# # pos_emb = nn.Embedding(max_len,embed_size)
# word_emb = nn.Embedding(src_vocab_size,embed_size)
# positions = torch.arange(0,seq_len).expand(N,seq_len).long()
# x = torch.randn(N,seq_len).long()
# # clip x to be within the vocab size
# print(x)
# x = x % src_vocab_size
# print(x.shape)
# print(x)
# out1 = word_emb(x)
# # out2 = pos_emb(positions)
# # out = nn.Dropout((out1 + out2))
# # print(out.shape)
# input_size = 28
# hidden_size = 256
# num_layers =2
# sequence_length = 28
# rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
# fully_connected = nn.Linear(sequence_length*hidden_size,10)
# x = None
# # h0 = torch.zeros(num_layers,N,hidden_size) 
# # out,h_final = rnn(x,h0)
# # print(h0.shape)

# x = torch.randn(3,4,5)
# y = nn.Linear(5,1)
# z = y(x)
# print(z.shape)


import torch.nn as nn
import torch
# Define an LSTM with 1 input feature, 64 hidden units, and 1 output feature
lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=3, batch_first=True)
linear = nn.Linear(64,1)

# Generate a random sequence of length 10 with 1 feature per time step
input_seq = torch.randn(5, 10, 1)

# Pass the input sequence through the LSTM
output, (h_n, c_n) = lstm(input_seq)

# Print the output and hidden states
print(output.shape)
print(h_n.shape)
print(c_n.shape)