# import torch
# import torch.nn as nn
# import torch.optim as optim

# # Define the input and output sequences
# input_seq = [['hello', 'world', 'how', 'are', 'you'],['Good','Morning']]
# output_seq = [['bonjour', 'monde', 'comment', 'allez', 'vous'],['Bon','Appetit']]

# # Define the vocabulary
# vocab = set()
# for sent in input_seq:
#     for word in sent:
#         vocab.add(word)
# for sent in output_seq:
#     for word in sent:
#         vocab.add(word)
# print(vocab)
# vocab_size = len(vocab)
# word_to_idx = {word: i for i, word in enumerate(vocab)}
# idx_to_word = {i: word for i, word in enumerate(vocab)}
# # add <sos> to the dictionary
# word_to_idx['<sos>'] = 14
# idx_to_word[14] = '<sos>'
# vocab.add('<sos>')
# print(word_to_idx)
# # Define the LSTM architecture
# input_size = 100 # size of input word embeddings
# hidden_size = 256 # number of hidden units in each LSTM layer
# num_layers = 2 # number of LSTM layers
# dropout = 0.2 # dropout probability
# batch_size = 32 # batch size for training
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use GPU if available

# class Seq2SeqLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, dropout):
#         super(Seq2SeqLSTM, self).__init__()
#         self.encoder = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
#         self.decoder = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
#         self.fc = nn.Linear(hidden_size, vocab_size)

#     def forward(self, input_seq, target_seq):
#         # Encode the input sequence into a fixed-length vector
#         print(f"input seq shape => {input_seq.shape}")
#         _, (h_n, c_n) = self.encoder(input_seq)

#         # Initialize the decoder hidden state with the encoder hidden state
#         hidden = (h_n, c_n)

#         # Initialize the decoder input with the start-of-sequence token
#         sos = torch.zeros(target_seq.shape[0], 1, input_size).to(device)
#         sos[:, :, word_to_idx['<sos>']] = 1
#         print("1")
#         # Decode the fixed-length vector into the output sequence
#         output, _ = self.decoder(sos, hidden)            
#         print(f"output shape 1-> {output.shape}")

#         print("2")
#         for i in range(1, target_seq.shape[1]):
#             print(f"output shape-> {output.shape}")
#             output_i, hidden = self.decoder(output, hidden)
#             print("3")
#             output = torch.cat((output, output_i), dim=1)

#         # Convert the output sequence into logits and return
#         logits = self.fc(output)
#         return logits

# # Define the hyperparameters
# learning_rate = 0.001
# num_epochs = 100

# # Define the model, optimizer, and loss function
# model = Seq2SeqLSTM(input_size, hidden_size, num_layers, dropout).to(device)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.CrossEntropyLoss()

# # Define the training loop
# for epoch in range(num_epochs):
#     total_loss = 0
#     for i in range(0, len(input_seq), batch_size):
#         batch_input_seq = input_seq[i:i+batch_size]
#         print(f"batch_input_seq is {batch_input_seq}")
#         print(f"Its shape : {len(batch_input_seq)},{(len(batch_input_seq[0]))}")
#         batch_target_seq = output_seq[i:i+batch_size]
#         print(f"batch_output_sequence is {batch_target_seq}")

#         # Convert the input and target sequences to tensors and move to device
#         input_tensor = torch.zeros(len(batch_input_seq), len(batch_input_seq[0]), input_size).to(device)
#         print(input_tensor.shape)
#         target_tensor = torch.zeros(len(batch_target_seq), len(batch_target_seq[0]), vocab_size).to(device)
#         print(target_tensor.shape)
#         for j, (input_word_seq, target_word_seq) in enumerate(zip(batch_input_seq, batch_target_seq)):
#             print(f"input_word_seq is {input_word_seq}")
#             for k, input_word in enumerate(input_word_seq):
#                 print(f"input word is {input_word}")
#                 input_tensor[j, k, word_to_idx[input_word]] = 1
#             for k, target_word in enumerate(target_word_seq):
#                 target_tensor[j, k, word_to_idx[target_word]] = 1
#         print(f"final input_tensor.shape=> {input_tensor.shape}")
#         print(f"final target_tensor.shape=> {target_tensor.shape}")

#         # Zero the gradients and forward pass
#         optimizer.zero_grad()
#         logits = model(input_tensor, target_tensor[:, :-1, :])

#         # Compute the loss and backward pass
#         loss = criterion(logits.view(-1, vocab_size), target_tensor[:, 1:, :].argmax(dim=-1).view(-1))
#         loss.backward()

#         # Clip the gradients and update parameters
#         nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
#         optimizer.step()

#         # Add the batch loss to the total loss
#         total_loss += loss.item()

#     # Print the epoch loss
#     epoch_loss = total_loss / (len(input_seq) / batch_size)
#     print(f'Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f}')

# With square kernels and equal stride

import torch
import torch.nn as nn
# m = nn.Conv2d(16, 33, 3, padding='same')
# non-square kernels and unequal stride and with padding
# m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# non-square kernels and unequal stride and with padding and dilation
# m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
# m = nn.Conv2d(16, 33, kernel_size=1, padding=0)
# input = torch.randn(20, 16, 240, 240)
# l = nn.LSTM(240,100,1)
# input2 = torch.randn(20, 200, 240)
# out , (h,c) = l(input2)
# output = m(input)
# print(output.shape)
# mean, log_variance = torch.chunk(input2, 2, dim=1)
# print(input2.shape)
# print(mean.shape)
# print(log_variance.shape)
input_latents = torch.randn(32,4,28,28)
input_latents = input_latents.repeat(2, 1, 1, 1)
print(input_latents.shape)