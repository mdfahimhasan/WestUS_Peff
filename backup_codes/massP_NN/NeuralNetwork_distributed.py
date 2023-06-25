import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from Codes.utils.system_ops import makedirs


##########################
class NeuralNetwork(torch.nn.Module):
    """
    A Neural Network Class for nonlinear regression type model. Creates model with user defined feed forward networks.

    Methods:
        initialize_weights(): Initializes weight for the Neural Network model.
        forward(): Calculates outputs of each layer given inputs in X.
    """

    def __init__(self, n_inputs, n_hiddens_list, n_outputs, activation_func='tanh', device='cpu'):
        """
        Creates a neural network with the given structure.

        :param n_inputs: int. 
                         Number of attributes/predictors that will be used in the model.
        :param n_hiddens_list: list. 
                               A list of number of units in each hidden layer. Each member of the list represents one
                               hidden layer.
        :param n_outputs: int. 
                          Number of output/prediction. Generally 1.
        :param activation_func: str. 
                                Name of the activation function. Can take 'tanh'/'relu'/'leakyrelu'.
        :param device: str. 
                       Name of the device to run the model. Either 'cpu'/'cuda'.
        """
        # Call parent class (torch.nn.Module) constructor
        super().__init__()

        self.device = device
        print(f'Model running on {device}....')

        # For printing
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_layers = n_hiddens_list

        # To build list of layers, must use torch.nn.ModuleList, not plain list
        self.hidden_layers = torch.nn.ModuleList()

        # Defining activation functions
        if activation_func == 'tanh':
            self.activation_func = torch.nn.Tanh()
        elif activation_func == 'relu':
            self.activation_func = torch.nn.ReLU()
        elif activation_func == 'leakyrelu':
            self.activation_func = torch.nn.LeakyReLU()
        else:
            raise Exception("Activation function should be 'tanh'/'relu'/'leakyrelu'")

        # Building the neural network structure
        for nh in n_hiddens_list:
            # one hidden layer and one activation func added in each loop
            self.hidden_layers.append(torch.nn.Sequential(
                torch.nn.Linear(n_inputs, nh), self.activation_func))

            n_inputs = nh  # output of each hidden layer will be input of next hidden layer

        self.output_layer = torch.nn.Linear(n_inputs, n_outputs)  # output layer doesn't have activation func

        self.initialize_weights()

        self.to(self.device)  # transfers the whole thing to 'cuda' if device='cuda'

    def __repr__(self):
        return 'NeuralNetwork({}, {}, {}, activation func={})'.format(self.n_inputs, self.n_hidden_layers,
                                                                      self.n_outputs, self.activation_func)

    def __str__(self):
        s = self.__repr__()
        if self.n_epochs > 0:  # self.total_epochs
            s += '\n Trained for {} epochs.'.format(self.n_epochs)
        return s

    def initialize_weights(self):
        """
        Initializes weight for the Neural Network model. For 'tanh', initializing optimization is 'xavier_normal'.
        For 'relu' and 'leakyrelu', initialization optimization is 'kaiming_normal'.
        """
        for m in self.modules():
            # self.modules() returns an iterable to the many layers or "modules" defined in the model class
            if isinstance(m, torch.nn.Linear):
                if isinstance(self.activation_func, torch.nn.Tanh):
                    torch.nn.init.xavier_normal_(m.weight)
                elif isinstance(self.activation_func, torch.nn.ReLU):
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(self.activation_func, torch.nn.LeakyReLU):
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, X):
        """
        Calculates outputs of each layer given inputs in X.

        :param X: torch.Tensor. 
                  Standardized input array representing attributes as columns and samples as row.

        :return: torch.Tensor. 
                 Standardized output array representing model prediction.
        """
        Y = X
        for hidden_layers in self.hidden_layers:
            Y = hidden_layers(Y)  # going through hidden layers

        # Final output
        Y = self.output_layer(Y)

        return Y


##########################
def calculate_rmse(Y_pred, Y_obsv):
    """
    Calculates RMSE value of model prediction vs observed data.

    :param Y_pred: prediction array or panda series object.
    :param Y_obsv: observed array or panda series object.

    :return: RMSE value.
    """
    if isinstance(Y_pred, np.ndarray):
        Y_pred = Y_pred.reshape(-1, 1)
        Y_obsv = Y_obsv.reshape(-1, 1)
        rmse_val = np.sqrt(np.mean((Y_obsv - Y_pred) ** 2))
    else:  # in case of pandas series
        rmse_val = np.sqrt(np.mean((Y_obsv - Y_pred) ** 2))
    return rmse_val


def calculate_r2(Y_pred, Y_obsv):
    """
    Calculates R2 value of model prediction vs observed data.

    :param Y_pred: prediction array or panda series object.
    :param Y_obsv: observed array or panda series object.

    :return: R2 value.
    """
    if isinstance(Y_pred, np.ndarray):
        Y_pred = Y_pred.reshape(-1, 1)
        Y_obsv = Y_obsv.reshape(-1, 1)
        r2_val = r2_score(Y_obsv, Y_pred)
    else:  # in case of pandas series
        r2_val = r2_score(Y_obsv, Y_pred)
    return r2_val


def scatter_plot(Y_pred, Y_obsv, savedir='../Model_Run/Plots'):
    """
    Makes scatter plot of model prediction vs observed data.

    :param Y_pred: flattened prediction array.
    :param Y_obsv: flattened observed array.
    :param savedir: filepath to save the plot.

    :return: A scatter plot of model prediction vs observed data.
    """
    fig, ax = plt.subplots()
    ax.plot(Y_obsv, Y_pred, 'o')
    ax.plot([0, 1], [0, 1], '-r', transform=ax.transAxes)
    ax.set_xlabel('GW Observed (mm)')
    ax.set_ylabel('GW Predicted (mm)')

    r2_val = round(r2(Y_pred, Y_obsv), 3)
    ax.text(0.1, 0.9, s=f'R2={r2_val}', transform=ax.transAxes)

    makedirs([savedir])
    fig_loc = savedir + '/scatter_plot.jpeg'
    fig.savefig(fig_loc, dpi=300)


def ddp_setup(rank, world_size, backend='gloo'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def to_torch(M, torch_type=torch.FloatTensor, device='cuda'):
    """
    Convert numpy array to torch Tensor.

    :param M: numpy array.
    :param torch_type: torch data type. Default set to Float.

    :return: A torch.Tensor.
    """
    if not isinstance(M, torch.Tensor):
        M = torch.from_numpy(M).type(torch_type).to(device)
    return M


def standardize(M):
    """
    Standardizes an input torch Tensor/numpy array.

    :param M: torch.tensor or numpy array.

    :return: Standardized torch Tensor, mean, and standard deviation values.
    """
    if isinstance(M, torch.Tensor):
        M_means = torch.mean(M, dim=0, keepdim=False)
        M_stds = torch.std(M, dim=0, keepdim=False)

    elif isinstance(M, np.ndarray):
        M_means = np.mean(M, axis=0, keepdims=False)
        M_stds = np.std(M, axis=0, keepdims=False)

    Ms = (M - M_means) / M_stds

    return Ms, M_means, M_stds


def execute_dataloader(dataset, rank, world_size, num_workers=0, batch_size=None, pin_memory=False, shuffle=False):
    """
    Execute the Dataloader process to create batches of data for each subprocess (for multiple GPU/CPU).
    In case of batch_size=None, Dataloader passes the whole dataset to a single GPU/CPU.

    :param dataset: torch.Tensor.
                    Input data array representing attributes as columns and samples as row.
    :param rank: int.
                 Within the process group, each process is identified by its rank, from 0 to K-1.
    :param world_size: int.
                       The number of processes in the group i.e. gpu number——K.
    :param num_workers: int.
                        How many subprocesses to use for data loading. Default 0 means that the data will be loaded
                        in the main process.
    :param batch_size: int.
                       How many samples per batch to load. Default is None, meaning the whoe dataset will be laoded at once.
                       (for single GPU/CPU).
    :param pin_memory: Boolean. Default False.
                       If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them.
    :param shuffle: Boolean. Default False.
                    Set to True to have the data reshuffled at every epoch.
                    
    :return: Returns an iterable over the given dataset.
    """
    # Removing one-hot encoded columns before standardizing. 
    # data_num and data_cat represents numeric and categorical variables, respectively.
    ####### change it later based on need
    categ_col_start_indx = 6
    disaggregated_obsv_data_col = -3
    fips_years_col = -1
    #######

    # Separating observed (training) data and fips_years columns
    fips_years = dataset[:, fips_years_col:]
    obsv_data = dataset[:, disaggregated_obsv_data_col:fips_years_col]

    data_numr = dataset[:, :categ_col_start_indx]  # getting rid of fips_years from predictors
    data_cat = dataset[:, categ_col_start_indx:disaggregated_obsv_data_col]

    # Standardization of numeric variable
    Xs, X_means, X_stds = standardize(data_numr)

    # Adding one-hot encoded columns with standardized variables
    Xs = torch.hstack((Xs, data_cat, obsv_data, fips_years))  # Xs is standardized dataset (categoricals not stansardized)

    distributed_sampler = DistributedSampler(Xs, num_replicas=world_size, rank=rank, shuffle=shuffle,
                                             drop_last=True)

    dataloader = None
    if batch_size is not None:
        dataloader = DataLoader(Xs, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
                                drop_last=True, shuffle=shuffle, sampler=distributed_sampler)
    else:
        batch_size = len(Xs)
        dataloader = DataLoader(Xs, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
                                drop_last=True, shuffle=shuffle, sampler=distributed_sampler)

    return dataloader, X_means, X_stds


def cleanup():
    dist.destroy_process_group()


def model_train_distributed_T(train_data, n_epochs, n_inputs, n_hiddens_list,
                              rank, world_size, batch_size=None, num_workers=0,
                              n_outputs=1, activation_func='tanh', device='cuda',
                              optimization='adam_betas', learning_rate=0.01, adam_betas=(0.5, 0.99), sgd_momentum=0.3,
                              verbose=True, fips_years_col=-1, epochs_to_print=100, setup_ddp=False):
    """
    Trains the model with given training and observed data in a distributed approach (Observed data is distributed
    to each pixel/sample in each epoch before initiating backpropagation).

    :param train_data: numpy array.
                       Standardized input array representing attributes as columns and samples as row.
    :param n_epochs: int.
                     Number of passes to take through all samples.
    :param n_inputs: int.
                     Number of attributes/predictors that will be used in the model.
    :param n_hiddens_list: list of int.
                           A list of number of units in each hidden layer. Each member of the list represents one
                           hidden layer.
    :param rank: int.
                 Within the process group, each process is identified by its rank, from 0 to K-1.
    :param world_size: int.
                       The number of processes in the group i.e. gpu number——K.
    :param batch_size: int.
                       How many samples per batch to load. Default 1 means data will be loaded to one CPU/GPU.
    :param num_workers: int.
                        How many subprocesses to use for data loading. Default 0 means that the data will be loaded
                        in the main process.
    :param n_outputs: int.
                      Number of output/prediction. Default 1.
    :param activation_func: str. 
                            Name of the activation function. Can take 'tanh'/'relu'/'leakyrelu'.
    :param device: str. 
                   Name of the device to run the model. Either 'cpu'/'cuda'.
    :param optimization: str.
                         Optimization algorithm. Can take 'adam_betas'/'sgd'.
    :param learning_rate: float.
                          Controls the step size of each update, only for sgd and adam_betas.
    :param adam_betas: tuple.
                  adam_betas hyperparameter for the adam_betas optimizer.
    :param sgd_momentum: float.
                         momentum parameter of sgd optimizer.
    :param verbose: boolean.
                    If True, prints training progress statement.
    :param fips_years_col: int.
                           Column index in X (torch.Tensor) array holding 'fips_years' attribute.
                           This column is removed before standardizing and forward pass.
    :param epochs_to_print: int.
                            If verbose is True, training progress will be printed after this number of epochs.
    :param setup_ddp: Boolean.
                      For running the model first time (in one kernel initialization) set to True and set False
                      rest of the times.

    :return: A trained NN model along with rmse_trace, train_means, train_stds, obsv_mean, obsv_std.
    """
    # Initialize the process groups
    if setup_ddp:
        ddp_setup(rank, world_size)

    # If already not torch.Tensor converts X and T. If device=='cuda' will transfer to GPU.
    train_data = to_torch(train_data, device=device)  # is a torch tensor now

    # Dataloader
    train_dataloader, train_means, train_stds = \
        execute_dataloader(dataset=train_data, rank=rank, world_size=world_size, num_workers=num_workers,
                           batch_size=batch_size, pin_memory=False, shuffle=False)

    ##############
    nn_model = None
    # Instantiating model, wrapping it with DDP, and moving it to the right device (cuda/gpu)
    if device == 'cpu':
        nn_model = NeuralNetwork(n_inputs, n_hiddens_list, n_outputs, activation_func, device)
    elif device == 'cuda':
        nn_model = NeuralNetwork(n_inputs, n_hiddens_list, n_outputs, activation_func, device).to(rank)

    # wrap the model with DDP
    # device_ids tell DDP where is the model
    # output_device tells DDP where to output, in this case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module 
    # in the model
    if device == 'cpu':
        nn_model = DDP(nn_model, device_ids=None, output_device=None, find_unused_parameters=True)
    elif device == 'cuda':
        nn_model = DDP(nn_model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    ##############

    # Call the requested optimizer optimization to train the weights.
    if optimization == 'sgd':
        optimizer = torch.optim.SGD(nn_model.parameters(), lr=learning_rate, momentum=sgd_momentum)
    elif optimization == 'adam':
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate, betas=adam_betas, weight_decay=0.01)
    else:
        raise Exception("optimization must be 'sgd', 'adam'")

    # mse function
    mse_func = torch.nn.MSELoss()
    rmse_trace = []  # records standardized calculate_rmse records per epoch

    disaggregated_obsv_data_col = -3
    total_obsv_data_col = -2
    n_epoch = list(range(n_epochs))
    Y_interim = {}  # a dictionary keeping track of updated observed GW values (at pixels)

    for epoch in n_epoch:
        # if using DistributedSampler, have to tell it which epoch this is
        train_dataloader.sampler.set_epoch(epoch)

        fips_years_track = {}  # a dictionary keeping track of regressed/predicted values for each fips_years
        for step, Xs in enumerate(train_dataloader):  # Xs has been standardized in execute_dataloader()
            fips_years = Xs[:, fips_years_col:]
            if epoch == 0:
                # Separating training (observed) data
                Y_obsv_pixel = Xs[:, disaggregated_obsv_data_col:total_obsv_data_col]
                shape = Y_obsv_pixel.shape
            else:
                # Updating observed data (at pixel) for training using outputs from last iteration/epoch
                Y_update = Y_interim[step].cpu().detach().numpy().squeeze().tolist()
                fips_years_step = fips_years.cpu().detach().numpy().squeeze().tolist()

                # Creating the interim_df to help with merging for updating training data (at pixel)
                interim_df = pd.DataFrame({'Y_update': Y_update, 'fips_years': fips_years_step})
                interim_df = interim_df.merge(fips_years_Y_regress_df, on='fips_years', how='left')
                interim_df['Y_obsv_pixel_updated'] = interim_df['Y_update'] / interim_df['Y_obsv_county_new']

                # Moving new training data to torch and reshaping it
                Y_obsv_pixel = to_torch(interim_df['Y_obsv_pixel_updated'].values, device=device).reshape(shape)

            # Separating training (observed) data at county level
            Y_obsv_county = Xs[:, total_obsv_data_col:fips_years_col]

            # Removing observed data and fip_years columns from predictors
            Xs = Xs[:, :disaggregated_obsv_data_col]

            # gradients need to be computed for Tensor Xs (the standardized predictors)
            Xs.requires_grad_(True)

            # Forward pass
            Y_pred = nn_model(Xs)

            # backpropagation # converting mse to calculate_rmse loss; unstandardized obsv and predicted output
            rmse_loss = torch.sqrt(mse_func(Y_pred, Y_obsv_pixel))

            # Backpropagate the loss
            rmse_loss.backward()

            # using optimizer
            optimizer.step()
            optimizer.zero_grad()  # Reset the gradients to zero

            # Creating dictionary to track model's interim predictions for each county
            Y_pred_copy = Y_pred.cpu().detach()  # Detaching and creating copy to avoid "second backward graph" error
            fips_years_copy = fips_years.cpu().detach()

            if len(fips_years_track) == 0:
                fips_years_track['fips_years'] = fips_years_copy.squeeze().tolist()
                fips_years_track['Y_regressed'] = Y_pred_copy.squeeze().tolist()
            else:
                fips_years_track['fips_years'].extend(fips_years_copy.squeeze().tolist())
                fips_years_track['Y_regressed'].extend(Y_pred_copy.squeeze().tolist())

            # Updating Y_obsv_pixel values. In each dataloader batch, "original county_sum" and model's interim
            # prediction are kept in a dictionary which are updated during next epoch using "new county_sum".
            # "original county_sum" = Y_obsv_county_main # "new county_sum" = Y_obsv_county_new

            # converting to numpy and transferring to torch again as Y_obsv_county_main i a tensor in cuda device
            Y_pred_copy = to_torch(Y_pred_copy.numpy(), device=device)
            Y_obsv_county_main = Y_obsv_county
            Y_interim[step] = Y_pred_copy * Y_obsv_county_main

        # Creating dataframe with each epochs model's prediction and fips_years records
        fips_years_Y_regress_df = pd.DataFrame(fips_years_track)
        fips_years_Y_regress_df = fips_years_Y_regress_df.groupby(by=['fips_years'])['Y_regressed'].sum().reset_index()
        fips_years_Y_regress_df.columns = ['fips_years', 'Y_obsv_county_new']

        # printing unstandardized calculate_rmse loss in training
        if verbose & (((epoch + 1) % epochs_to_print) == 0):
            print(f'{optimization}: Epoch={epoch + 1} RMSE={rmse_loss.item():.5f}')

        rmse_trace.append(rmse_loss)

    cleanup()

    return nn_model, rmse_trace, train_means, train_stds


def predict(X, fips_years_arr, trained_model, train_means, train_stds):
    # obsv_mean, obsv_std):
    """
    Uses trained model to predict on given data.

    :param X: numpy array. 
              Input array representing attributes as columns and samples as row. Must not have fips_years column.
    :param fips_years_arr: numpy array. 
                           An array representing fips_year attribute that represents county and year of data.

    :return: A numpy array of prediction (aggregated at county level).
    """
    # Removing 1-dimensions from fips_years_arr
    fips_years = fips_years_arr.squeeze()

    # Moving to torch (cpu)
    X = to_torch(X)

    # Removing one-hot encoded columns before standardizing. 
    # data_num and data_cat represents numeric and categorical variables, respectively.
    ####### change it later based on need
    categ_col_start_indx = 6
    disaggregated_obsv_data_col = -3
    fips_years_col = -1
    #######

    X_num = X[:, :categ_col_start_indx]  # getting rid of fips_years from predictors
    X_cat = X[:, categ_col_start_indx:disaggregated_obsv_data_col]

    # Standardization
    Xs = (X_num - train_means) / train_stds

    # Adding one-hot encoded columns with standardized variables
    Xs = torch.hstack((Xs, X_cat))  # Xs is standardized dataset (categoricals not stansardized)

    # Ys
    Y = trained_model(Xs)  # standardized result

    # Y = Ys * obsv_mean + obsv_std  # Unstandardizing
    Y = Y.cpu().detach()  # prediction as numpy array for each pixel  

    df = pd.DataFrame(Y, columns=['gw prediction (mm)'])  # dataframe created to store pixel-wise results

    df['fips_years'] = pd.Series(
        fips_years)  # adding fips_years to dataframe for aggregating result to county level
    prediction_df = df.groupby(by=['fips_years'])[
        'gw prediction (mm)'].sum().reset_index()  # prediction aggregated to county level

    return prediction_df  # predicted result as a dataframe
