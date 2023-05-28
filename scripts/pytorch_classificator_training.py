# This script is used to train and save a model
# after each epoch. Each epoch model is then 
# tested on the testing set and required statistics
# are calculated.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages as PDF
import pandas as pd
from pathlib import Path
import os

# Defining the directory paths.
parent_directory = str(Path(__file__).absolute().parent.parent)
tensor_dir = Path(f'{parent_directory}/data/final_data')

# Loading training and testing data as X and y variables.
training_X = torch.load(tensor_dir.joinpath("training_X.pt")).to(dtype=torch.float32)
training_y = torch.load(tensor_dir.joinpath("training_y.pt")).to(dtype=torch.float32)
validation_X = torch.load(tensor_dir.joinpath("validation_X.pt")).to(dtype=torch.float32)
validation_y = torch.load(tensor_dir.joinpath("validation_y.pt")).to(dtype=torch.float32)

# Switching to CUDA if possible.
if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')
print(device)


# ==================================
# CLASSIFICATOR
# ==================================
class BinaryClassifier(nn.Module):
    def __init__(self, 
                 input_size=1281, 
                 intermediate_size=380, 
                 output_size=1, 
                 hidden_sizes=[], 
                 activation_fc=nn.ReLU(),
                 dropout_rate=0.5):
        super().__init__()
        
        self.input_size = input_size
        self.intermediate_size = intermediate_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation_fc = activation_fc
        self.dropout_rate = dropout_rate
        
        layers = []
        
        # Adding the input layer.
        layers.append(nn.Linear(input_size, hidden_sizes[0]) \
                      if len(hidden_sizes) > 0 else nn.Linear(input_size, intermediate_size))
        layers.append(activation_fc)
        
        layers.append(nn.Dropout(dropout_rate))

        # Adding the hidden layers.
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(activation_fc)
            
        # Adding the output layer.
        layers.append(nn.Linear(hidden_sizes[-1] \
                                if len(hidden_sizes) > 0 else intermediate_size, output_size))
        layers.append(nn.Sigmoid())
        
        # Creating the sequential model.
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


# ==================================
# HYPERPARAMETERS
# ==================================
# Model parameter setting.
input_size = 1281
intermediate_size = 640
output_size = 1
hidden_sizes = []
activation_fc = nn.ReLU()
dropout_rate = 0.2 # Default is 0.5

# Model creation.
model = BinaryClassifier(input_size, intermediate_size, output_size, hidden_sizes, activation_fc)
model = model.to(device)
print(model)

# Hyperparameter setting.
batch_size = 20
epochs = 100
learning_rate = 0.001
threshold = 0.5
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.BCELoss()

# Model name.
model_name = 'ESM-1'

# PATH for saving statistics and creating a directory for this model.
trained_model_dir = str(Path(f'{parent_directory}/data/trained_models/{model_name}'))

if not os.path.exists(trained_model_dir):
    os.makedirs(trained_model_dir)
    os.makedirs(f'{trained_model_dir}/models')

# ==================================
# DATASETS
# ==================================
# Creating a Tensor Dataset and loading data into a DataLoader to fragment data into batches.
training_data = TensorDataset(training_X, training_y)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
validation_data = TensorDataset(validation_X, validation_y)
val_dataloader = DataLoader(validation_data, batch_size=1, shuffle=True)

# Counting the 1 and 0 class for weight application. Because the dataset
# is imbalanced. There are less 1s than 0s.
total = training_y.size()[0] 
class_1_count = torch.count_nonzero(training_y).item()
class_0_count = total - class_1_count
ratio = class_0_count / class_1_count

# Checking the structure of the first inexed batch in the test dataloader.
# (X, y) refers to the batch, where X are the features and y are the targets.
for batch_index, (X, y) in enumerate(train_dataloader):
    print('Batch index: ', batch_index)
    print('X size: ', X.size())
    print('y size: ', y.size())
    print('Device:', y.device)
    break


# ==================================
# TRAINING PREPERATION
# ==================================
# Defining arrays for dataframes.
epoch_array = []
matthews_corrcoef_array = []
confusion_matrix_array = []
f1_score_array = []
accuracy_score_array = []
false_positives = []

# Opening PDF documents.
confusion_matrix_pdf = PDF(f'{trained_model_dir}/{model_name}_confusion_matrix.pdf')
ROC_pdf = PDF(f'{trained_model_dir}/{model_name}_ROC.pdf')
precission_recall_pdf = PDF(f'{trained_model_dir}/{model_name}_precission_recall.pdf')


# ==================================
# METHOD FOR CALCULATING STATISTICS
# ==================================
from sklearn.metrics import confusion_matrix, f1_score, \
matthews_corrcoef, accuracy_score, roc_curve, roc_auc_score, \
precision_recall_curve, average_precision_score

def statistics(y_true, y_predicted, y_predicted_binary, epoch):
  # Matthew correlation coefficient.
  matt_coe = matthews_corrcoef(y_true, y_predicted_binary) 

  # Confusion matrix
  # [[True negative, False positive]    
  #  [False negative, True positive]]
  conf_matrix = confusion_matrix(y_true, y_predicted_binary)

  # True Negative, False Positive, False Negative, True Positive.
  TN, FP, FN, TP = conf_matrix.ravel()
  false_positives.append(FP)

  # The F1 score.
  f1_scr = f1_score(y_true, y_predicted_binary)

  # Accuracy.
  acc_scr = accuracy_score(y_true, y_predicted_binary)

  # Confusion matrix plot.
  cf_matrix = confusion_matrix(y_true, y_predicted_binary)
  fig = plt.figure(figsize=(8, 5))
  sns.heatmap(cf_matrix, annot=True, cbar=False, fmt="g")
  plt.xlabel('Predicted label')
  plt.ylabel('True label')
  plt.title(f'Confusion matrix for epoch {epoch}')
  confusion_matrix_pdf.savefig()
  plt.close()

  # ROC curve.
  # Calculate ROC curve and AUC
  fpr, tpr, thresholds = roc_curve(y_true, y_predicted)
  auc = roc_auc_score(y_true, y_predicted)
  # Plot ROC curve
  plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
  # Plot random classifier
  plt.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)')
  # Add labels and legend
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(f'ROC Curve for epoch {epoch}')
  plt.legend()
  ROC_pdf.savefig()
  plt.close()

  # Preccision-recall curve.
  # Calculate precision, recall, and threshold values
  precision, recall, thresholds = precision_recall_curve(y_true, y_predicted)
  ap = average_precision_score(y_true, y_predicted)
  # Plot PR curve
  plt.plot(recall, precision, label=f'AP = {ap:.2f}')
  # Add labels and legend
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title(f'Precision-Recall Curve for epoch {epoch}')
  plt.legend()
  precission_recall_pdf.savefig()
  plt.close()  

  return matt_coe, conf_matrix, f1_scr, acc_scr


# ==================================
# METHOD FOR PLOTTING FALSE NEGATIVE
# VALUES PER EPOCH
# ==================================
def plot_false_positive():
  false_positive_epoch_pdf = PDF(f'{model_name}_false_positive_rate.pdf')
  plt.plot(false_positives)
  plt.title('False Positives per Epoch')
  plt.xlabel('Epochs')
  plt.ylabel('False Positive Values')
  false_positive_epoch_pdf.savefig()
  plt.close()
  false_positive_epoch_pdf.close()


# ==================================
# METHOD FOR TESTING THE TRAINED
# MODEL ON A TESTING SET
# ==================================
def validation_set(model_file, epoch):
  model.load_state_dict(torch.load(model_file))
  y_true = []
  y_predicted = []

  # Batch_size set to 1.
  with torch.no_grad():
    for X, y in val_dataloader:
        X = X.to(device)
        y = y.to(device)
        outputs = model(X)
        
        y = y.cpu().squeeze().numpy()
        outputs = outputs.cpu().squeeze().numpy()
        y_true.append(float(y))
        y_predicted.append(float(outputs))
  
  y_predicted_binary = np.where(np.array(y_predicted) > threshold, 1, 0)

  matthews_coeficient, confusion_matrix, f1_score, accuracy_score = statistics(y_true, y_predicted, y_predicted_binary, epoch)

  epoch_array.append(epoch)
  matthews_corrcoef_array.append(matthews_coeficient)
  confusion_matrix_array.append(confusion_matrix)
  f1_score_array.append(f1_score)
  accuracy_score_array.append(accuracy_score)


# ==================================
# TRAINING MODELS FOR EACH EPOCH
# ==================================
# Array for losses.
global_losses = []

for i in range(epochs):
  epoch_loss = []
  for batch_index, (X, y) in enumerate(train_dataloader):
    # Switching to GPU.
    X = X.to(device)
    y = y.to(device)

    # Calculating output.
    output = model(X)

    # Caclulating loss.
    weight = torch.ones_like(y) / ratio + (1.0 - 1.0 / ratio) * y
    loss = F.binary_cross_entropy(output, y, weight=weight)
    epoch_loss.append(loss.item())

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  # Epoch over.
  loss_avg = np.mean(epoch_loss)
  global_losses.append(loss_avg)
  print(f'Epoch: {i+1} | Loss: {loss_avg}')
  model_file = f'{trained_model_dir}/models/epoch_{i+1}.pt'
  torch.save(model.state_dict(), model_file)
  validation_set(model_file, i+1)

statistics_df = pd.DataFrame()
statistics_df['epoch'] = epoch_array
statistics_df['matthews_corrcoef'] = matthews_corrcoef_array
statistics_df['confusion_matrix'] = confusion_matrix_array
statistics_df['f1_score'] = f1_score_array
statistics_df['accuracy_score'] = accuracy_score_array
statistics_df['loss'] = global_losses
statistics_df.to_csv(f'{trained_model_dir}/{model_name}_statistics.csv', index=False)

# Opening a PDF for loss and model.
loss_model_pdf = PDF(f'{trained_model_dir}/{model_name}_model_and_loss.pdf')

# Plotting the model.
modelPage = plt.figure(figsize=(8, 5))
modelPage.clf()
txt = f'{model}'
modelPage.text(0.5,0.4,txt, transform=modelPage.transFigure, size=16, ha="center")
loss_model_pdf.savefig()
plt.close()

# Plotting the loss
plt.plot(global_losses)
plt.title('Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
loss_model_pdf.savefig()
plt.close()

# Plotting the false positive rate per epoch.
plot_false_positive()

# Closing the PDF documents.
loss_model_pdf.close()
confusion_matrix_pdf.close()
ROC_pdf.close()
precission_recall_pdf.close()


# ==================================
# HYPERPARAMETER SAVING
# ==================================
# Saving hyperparameters.
hidden_layers = ""
if (len(hidden_sizes) == 1):
  hidden_layers = hidden_sizes[0]
elif (len(hidden_sizes) > 1):
  for i in hidden_sizes:
    hidden_layers += f'{i} '
else:
  hidden_layers = str(intermediate_size)

hyperparameter_df = pd.DataFrame()
hyperparameter_df['model_name'] = [f'{model_name}']
hyperparameter_df['input_size'] = [f'{input_size}']
hyperparameter_df['hidden_layers'] = [hidden_layers]
hyperparameter_df['output_size'] = [f'{output_size}']
hyperparameter_df['batch_size'] = [f'{batch_size}']
hyperparameter_df['epochs'] = [f'{epochs}']
hyperparameter_df['learning_rate'] = [f'{learning_rate}']
hyperparameter_df['threshold'] = [f'{threshold}']
hyperparameter_df['dropout_rate'] = [f'{dropout_rate}']
hyperparameter_df['optimizer'] = [f'{optimizer.__class__.__name__}']
hyperparameter_df['loss_function'] = [f'{loss_fn.__class__.__name__}']
hyperparameter_df['activation_function'] = [f'{activation_fc.__class__.__name__}']

hyperparameter_df.to_csv(f'{trained_model_dir}/{model_name}_hyperparameters.csv', index=False)