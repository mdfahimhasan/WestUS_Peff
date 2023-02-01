
import torch
import numpy as np
import pandas as pd

class NeuralNetwork(torch.nn.Module):
    """
    A Neural Network Class for nonlinear regression type model. Creates model with user defined feed forward networks.

    Methods:
        initialize_weights(): Initializes weight for the Neural Network model.
        to_torch(): Convert numpy array to torch.Tensor.
        standardize(): Standardizes an input torch Tensor.
        _forward(): Calculates outputs of each layer given inputs in X.
        train(): Trains the model with given training and observed data.
                 ** This method() will not be used in our model training.
        distribute_T_for_backprop(): Distributes observed data to each pixel/sample for backpropagation purpose.
        train_with_distributed_T(): Trains the model with given training and observed data in a distributed approach
                                    (Observed data is distributed to each pixel/sample in each epoch before
                                    initiating backpropagation).
        predict(): Uses trained model to predict on given data.
        get_performance_trace(): Provides model loss for each epoch.
    """
    
    def __init__(self, n_inputs, n_hiddens_list, n_outputs, activation_func='tanh', device='cpu'):
        """
        Creates a neural network with the given structure.

        :param n_inputs: int. Number of attributes/predictors that will be used in the model.
        :param n_hiddens_list: list. A list of number of units in each hidden layer. Each member of the list represents one
                               hidden layer.
        :param n_outputs: int. Number of output/prediction. Generally 1.
        :param activation_func: str. Name of the activation function. Can take 'tanh'/'relu'/'leakyrelu'.
        :param device: str. Name of the device to run the model. Either 'cpu'/'cuda'.
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

        if activation_func == 'tanh':
            self.activation_func = torch.nn.Tanh()
        elif activation_func == 'relu':
            self.activation_func = torch.nn.ReLU()
        elif activation_func == 'leakyrelu':
            self.activation_func = torch.nn.LeakyReLU()
        else:
            raise Exception("Activation function should be 'tanh'/'relu'/'leakyrelu'")

        for nh in n_hiddens_list:
            self.hidden_layers.append(torch.nn.Sequential(
                torch.nn.Linear(n_inputs, nh),
                self.activation_func))

            n_inputs = nh  # output of each hidden layer will be input of next hidden layer

        self.output_layer = torch.nn.Linear(n_inputs, n_outputs)
        self.initialize_weights()
        
        self.performance_trace = []  # records standardized rmse records per epoch  
        self.to(self.device)  # transfers the whole thing to 'cuda' if device='cuda'
        self.fips_years = None  # modified in the train_with_distributed_T() function
        self.n_epochs = None
        
    def __repr__(self):
        return 'NeuralNetwork({}, {}, {}, activation func={})'.format(self.n_inputs, self.n_hidden_layers,
                                                                      self.n_outputs, self.activation_func)
    def __str__(self):
        s = self.__repr__()
        if self.n_epochs > 0:  # self.total_epochs
            s += '\n Trained for {} epochs.'.format(self.n_epochs)
            s += '\n Final standardized training error {:.4g}.'.format(self.performance_trace[-1])
        return s
    
    def initialize_weights(self):
        """
        Initializes weight for the Neural Network model. For 'tanh' initializing method is 'xavier_normal'. For 'relu' and
        'leakyrelu' initialization method is 'kaiming_normal'.
        """
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                if isinstance(self.activation_func, torch.nn.Tanh):
                    torch.nn.init.xavier_normal_(m.weight)
                elif isinstance(self.activation_func, torch.nn.ReLU):
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(self.activation_func, torch.nn.LeakyReLU):
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def to_torch(self, M, torch_type=torch.FloatTensor):
        """
        Convert numpy array to torch Tensor.

        :param M: numpy array.
        :param torch_type: torch data type. Default set to Float.

        :return: A torch.Tensor.
        """
        if not isinstance(M, torch.Tensor):
            M = torch.from_numpy(M).type(torch_type).to(self.device)
        return M
    
    def standardize(self, M):
        """
        Standardizes an input torch Tensor.

        :param M: torch.tensor.

        :return: Standardized torch Tensor, mean, and standard deviation values.
        """
        M_means = torch.mean(M, dim=0, keepdim=False)
        M_stds = torch.std(M, dim=0, keepdim=False)

        Ms = (M - M_means) / M_stds

        return Ms, M_means, M_stds
    
    def _forward(self, X):
        """
        Calculates outputs of each layer given inputs in X.

        :param X: torch.Tensor. Standardized input array representing attributes as columns and samples as row.

        :return: torch.Tensor. Standardized output array representing model prediction.
        """
        Y = X
        for hidden_layers in self.hidden_layers:
            Y = hidden_layers(Y)  # going through hidden layers

        # Final output
        Y = self.output_layer(Y)

        return Y

    def train(self, X, T, n_epochs, method='adam', learning_rate=None, verbose=True):
        """
        Trains the model with given training and observed data.
        ** This method() will not be used in our model training.

        :param X: torch.Tensor. Standardized input array representing attributes as columns and samples as row.
        :param T: torch.Tensor. Standardized observed data (as array) to compare model prediction and initiate
                  backpropagation.
        :param n_epochs: int. Number of passes to take through all samples.
        :param method: str. Optimization algorithm. Can take 'adam'/'sgd'.
        :param learning_rate: float. Controls the step size of each update, only for sgd and adam.
        :param verbose: boolean. If True, prints training progress statement.

        :return: self. A trained NN model.
        """
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

        # If already not torch.Tensor converts X and T. If device=='cuda' will transfer to GPU. 
        X = self.to_torch(X)
        T = self.to_torch(T)

        # Standardization
        Xs, self.X_means, self.X_stds = self.standardize(X)
        Ts, self.T_means, self.T_stds = self.standardize(T)

        Xs.requires_grad_(True)

        # Call the requested optimizer method to train the weights.
        if method == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        elif method == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0)
        else:
            raise Exception("method must be 'sgd', 'adam'")

        mse_func = torch.nn.MSELoss()  # mse function

        for epoch in range(self.n_epochs):
            Ys = self._forward(Xs)
            rmse_loss = torch.sqrt(mse_func(Ys, Ts))  # converting mse to rmse loss
            print(rmse_loss)
            rmse_loss.backward()  # Backpropagates the loss function

            # using optimizer
            optimizer.step()
            optimizer.zero_grad()   # Reset the gradients to zero

            # printing standardized rmse loss in training
            epochs_to_print = 1000
            if verbose & (((epoch + 1) % epochs_to_print) == 0):
                print(f'{method}: Epoch={epoch + 1} RMSE={rmse_loss.item():.5f}')

            self.performance_trace.append(rmse_loss)

        # Returning neural network object to allow applying other methods
        # after training, such as:    Y = nnet.train(X, T, 100, 0.01).predict(X)
        return self
    
    @staticmethod
    def distribute_T_for_backprop(input_torch_stack, observedGW_data_csv):
        """
        Distributes observed data to each pixel/sample for backpropagation purpose.

        :param input_torch_stack: torch.Tensor. A stacked array generated after each epoch with prediction in
                                  column 1 and fips_years in columns 2.
        :param observedGW_data_csv: csv. Observed data (at county level).

        :return: A numpy array of distributed T (observed) values for each pixel/sample.
        """
        # input_torch_stack is a numpy array having Ys output from self._forward() and 'fips_years' records stacked
        predicted_df = pd.DataFrame(input_torch_stack, columns=['Ys', 'fips_years'])

        # calculating total gw_withdrawal predicted (Ys_sum) using groupby on Ys
        predicted_grp_df = predicted_df.groupby(by=['fips_years'])['Ys'].sum().reset_index()
        predicted_grp_df = predicted_grp_df.rename(columns={'Ys': 'Ys_sum'})

        # processing observed grounwater use data for counties
        observed_df = pd.read_csv(observedGW_data_csv)

        # 1st Merge
        # merging grouped groundwater pumping dataframe with observed pumping dataframe.
        merged_df = predicted_grp_df.merge(observed_df, on=['fips_years'], how='left')
        merged_df = merged_df.sort_values(by=['fips_years']).reset_index(drop=True)
        merged_df = merged_df[['fips_years', 'Ys_sum', 'total_gw_observed']]
        #         print(merged_df.head(5))

        # 2nd Merge
        # merging the merged dataframe with pixel-wise dataframe
        # needed for distributing observed data to each pixel based on percentage share on Ys_sum.
        # required for backpropagation.
        distributed_df = predicted_df.merge(merged_df, on=['fips_years'], how='left').reset_index()
        #         print('distributed_df')
        #         print(distributed_df.head(5))

        distributed_df['share_of_totalgw'] = (distributed_df['Ys'] / distributed_df['Ys_sum']) \
                                             * distributed_df['total_gw_observed']

        T_share = distributed_df[
            ['share_of_totalgw']].to_numpy()  # this is not standardized. Standardize it in train function

        return T_share

    def train_with_distributed_T(self, X, T_csv, n_epochs, method='sgd', learning_rate=None, verbose=True,
                                 fips_years_col=-1, epochs_to_print=100):
        """
        Trains the model with given training and observed data in a distributed approach (Observed data is distributed
        to each pixel/sample in each epoch before initiating backpropagation).

        :param X: numpy array. Standardized input array representing attributes as columns and samples as row.
        :param T_csv: csv. Observed data (at county level).
        :param n_epochs: int. Number of passes to take through all samples.
        :param method: str. Optimization algorithm. Can take 'adam'/'sgd'.
        :param learning_rate: float. Controls the step size of each update, only for sgd and adam.
        :param verbose: boolean. If True, prints training progress statement.
        :param fips_years_col: int. Column index in X (torch.Tensor) array holding 'fips_years' attribute.
                               This column is removed before standardizing and forward pass.
        :param epochs_to_print: int. If verbose is True, training progress will be printed after this number of epochs.

        :return: self. A trained NN model using distributed observed data approach.
        """
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

        # Creating fips_years unique list 
        fips_years = X[:, fips_years_col:]
        self.fips_years = fips_years
        fips_years_unq = np.unique(X[:, fips_years_col])
        X = X[:, :fips_years_col]  # getting rid of fips_years from predictors

        # If already not torch.Tensor converts X and T. If device=='cuda' will transfer to GPU. 
        X = self.to_torch(X)  # is a torch tensor now

        # Standardization X
        Xs, self.X_means, self.X_stds = self.standardize(X)

        Xs.requires_grad_(True)

        # Call the requested optimizer method to train the weights.
        if method == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        elif method == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0)
        else:
            raise Exception("method must be 'sgd', 'adam'")

        mse_func = torch.nn.MSELoss()  # mse function

        for epoch in range(self.n_epochs):
            Ys = self._forward(Xs)

            # grouping Ys by fips_years codes
            Ys_stack = np.hstack((Ys.cpu().detach(), self.fips_years))  # is a numpy arrary

            # T (unstandardized, numpy array) is countywise observed gw pumping data distributed among each pixel.
            # This distribution will be performed in each epoch, based on prediction share of Ys in Ys_sum.
            T = self.distribute_T_for_backprop(Ys_stack, T_csv)

            # converting to torch tensor and standardizing
            T = self.to_torch(T)
            Ts, self.T_means, self.T_stds = self.standardize(T)

            # backpropagation
            rmse_loss = torch.sqrt(mse_func(Ys, Ts))  # coverting mse to rmse loss
            rmse_loss.backward()  # Backpropagates the loss function

            # using optimizer
            optimizer.step()
            optimizer.zero_grad()  # Reset the gradients to zero

            # printing standardized rmse loss in training
            if verbose & (((epoch + 1) % epochs_to_print) == 0):
                print(f'{method}: Epoch={epoch + 1} RMSE={rmse_loss.item():.5f}')

            self.performance_trace.append(rmse_loss)

        return self

    def predict(self, X, fips_years_arr=None):
        """
        Uses trained model to predict on given data.

        :param X: numpy array. Standardized input array representing attributes as columns and samples as row. Must not
                  have fips_years column.
        :param fips_years_arr: numpy array. If the function/method is used on validation/testing data, must have this
                               a fips_years array. Default set to None for training data (fips_years will come form
                               self.fips_years)

        :return: A numpy array of prediction (aggregated at county level).
        """
        if fips_years_arr is not None:
            fips_years = fips_years_arr.squeeze()
        else:
            fips_years = self.fips_years.squeeze()

        # Moving to torch
        X = self.to_torch(X)

        # Standardization
        Xs = (X - self.X_means) / self.X_stds

        Ys = self._forward(Xs)  # standardized result
        Y = Ys * self.T_stds + self.T_means  # Unstandardizing
        Y = Y.cpu().detach()  # prediction as numpy array for each pixel  

        df = pd.DataFrame(Y, columns=['Y_predicted'])  # dataframe created to store pixel-wise results
        df['fips_years'] = pd.Series(
            fips_years)  # adding fips_years to dataframe for aggregating result to county level
        df = df.groupby(by=['fips_years'])['Y_predicted'].mean().reset_index()  # prediction aggregated to county level
        Y_predicted = df['Y_predicted'].to_numpy()

        return Y_predicted  # predicted result as numpy array

    def get_performance_trace(self):
        """
        Provides model loss for each epoch.

        :return: A list of values of model error for all epoch.
        """
        performance = [loss.item() for loss in self.performance_trace]
        return performance
