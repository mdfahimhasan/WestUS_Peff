# from https://github.com/jaymody/backpropagation/blob/master/nn.ipynb
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
# import ipdb
from sklearn.metrics import f1_score
from rasterio.plot import show

plt.close('all')

def one_hot_encode(y, num_classes):
    # ipdb.set_trace()
    y_one_hot = np.zeros((y.shape[0], num_classes))
    # idx = [np.arange(y.shape[0]), y]
    for kk in range(y.shape[0]):
        y_one_hot[kk,y[kk]] = 1
    return y_one_hot

# def mse(y_hat, y_one_hot,weights=None):
#     # ipdb.set_trace()
#     if weight!=None:
#         for kk,weight in enumerate(weights):
#             y_one_hot[:,kk]=

#     return np.mean((y_one_hot - y_hat)**2)

# def mse(y_hat, y_one_hot):
#     # ipdb.set_trace()
#     mse=0
#     for kk in range(y_hat.shape[1]):
#         mse0 = np.mean((y_one_hot[:,kk] - y_hat[:,kk])**2)
#         mse=mse+mse0
#     return mse

def mse(model_ph,x_ph,y_ph,y_one_hot_ph,model_as,x_as,y_as,y_one_hot_as,col_inds):
    yhat_ph = model_ph.forward(x_ph)
    yhat_ph_as = model_ph.forward(x_as[:,col_inds])
    
    # ipdb.set_trace() 
    
    prob_ph_as = yhat_ph_as[:,1].reshape(len(yhat_ph_as),1)
    
    # ipdb.set_trace()
    
    x_as = np.hstack((x_as,prob_ph_as))
    
    yhat_as = model_as.forward(x_as)
    
    mse_as = np.mean((y_one_hot_as-yhat_as)**2)
    mse_ph = 1*np.mean((y_one_hot_ph-yhat_ph)**2)
    
    mse_tot = np.mean([mse_as,mse_ph])
    return(mse_tot)

# def d_mse(y_hat, y_one_hot):
#     # ipdb.set_trace()
#     mse=0
#     for kk in range(y_hat.shape[1]):
#         mse0 = np.mean((y_one_hot[:,kk] - y_hat[:,kk])**2)
#         mse=mse+mse0
#     return mse

def d_mse(y_hat, y_one_hot):
    # ipdb.set_trace()
    return (1 / y_hat.shape[0]) * (1/y_hat.shape[-1]) * -2 * np.sum(y_one_hot - y_hat, axis=0)

def linear(x):
    return x


def d_linear(x):
    return np.ones_like(x)


def relu(x):
    return np.where(x > 0, x, 0)


def d_relu(x):
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            self.weights.append(np.random.randn(n_in, n_out))
            self.biases.append(np.random.randn(1, n_out))

        self.tape = [None for _ in range(len(self.weights) + 1)]
        self.activations = [linear] + [relu for _ in range(len(self.weights) - 1)] + [sigmoid]
        self.d_activations = [d_linear] + [d_relu for _ in range(len(self.weights) - 1)] + [d_sigmoid]

    def forward(self, x, grad=False):
        if grad and self.tape[0] is not None:
            raise ValueError("Cannot call forward with grad without calling backwards")

        if grad:
            self.tape[0] = x
        
        for i in range(len(self.weights)):
            x_hat = x @ self.weights[i] + self.biases[i]
            x = self.activations[i + 1](x_hat)
            if grad:
                self.tape[i + 1] = x_hat

        return x

    def backward(self, d_loss):
        assert d_loss.shape == self.weights[-1].shape[-1:]

        weights_grad = [None for w in self.weights]
        biases_grad = [None for b in self.biases]

        d_activation = self.d_activations[-1]
        error = d_loss * d_activation(self.tape[-1]) # (n_out) * (n_out)
        error = error.reshape(1, -1)

        for i in reversed(range(len(self.weights))):
            # error = (1, n_out)
            # tape[i] = (n_in)
            # weights[i] = (n_in, n_out)
            x = self.tape[i]
            activation = self.activations[i]
            d_activation = self.d_activations[i]

            weights_grad[i] = error * activation(x.reshape(-1, 1)) # (1, n_out) * (n_in, 1) -> (n_in, n_out) * (n_in, n_out) via broadcast -> (n_in, n_out)
            biases_grad[i] = error * 1 # derivative of bias is vector of ones, which we represent as 1 to be explicit
            
            error = error @ self.weights[i].T  # (1, n_out) @ (n_out, n_in) -> (1, n_in)
            error = error * d_activation(x).reshape(1, -1) # (1, n_in) * (1, n_in)

        self.tape = [None for _ in range(len(self.weights) + 1)]

        return weights_grad, biases_grad

    def predict(self, x):
        y = self.forward(x)
        preds = np.argmax(y, axis=1)
        return preds
    
# validate forward prop works as expected
model = NeuralNetwork(2, [8], 3)

n_classes = 3
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = one_hot_encode(np.array([0, 1, 2, 1]), n_classes)

# loss = (targets - sigmoid(relu(inputs @ model.weights[0] + model.biases[0]) @ model.weights[1] + model.biases[1]))**2
# assert np.allclose(loss.mean(), mse(model.forward(inputs), targets))

def compute_numerical_gradient(model_ph, x_ph, y_ph, model_as, x_as, y_as, col_inds, batch_size, num_classes, delta=1e-8):
    # batches = list(zip(X, Y))

    def compute_loss():
        loss = 0
        y_one_hot_ph = one_hot_encode(y_ph, num_classes)
        y_one_hot_as = one_hot_encode(y_as,num_classes)
        
        loss = mse(model_ph,x_ph,y_ph,y_one_hot_ph,
                   model_as,x_as,y_as,y_one_hot_as,col_inds)
        return loss
        # for i in range(0, len(batches), batch_size):
        #     batch = batches[i : i + batch_size]
        #     x, y = zip(*batch)
        #     x = np.array(x)
        #     y = np.array(y)
        #     y_one_hot = one_hot_encode(y, num_classes)
        #     y_hat = model.forward(x)
        #     loss += mse(y_hat, y_one_hot) * len(x)
        # return loss / len(X)

    base_loss = compute_loss()

    numerical_weights_grad_as = [np.zeros(w.shape) for w in model_as.weights]
    numerical_biases_grad_as = [np.zeros(b.shape) for b in model_as.biases]
    for l in range(len(model_as.weights)):
        for ix, iy in np.ndindex(model_as.weights[l].shape):
            old_w = model_as.weights[l][ix][iy]
            new_w = old_w + delta
            model_as.weights[l][ix][iy] = new_w

            new_loss = compute_loss()
            dL = base_loss - new_loss
            dw = old_w - new_w
            numerical_weights_grad_as[l][ix][iy] = dL / dw

            model_as.weights[l][ix][iy] = old_w

        for ii in range(len(model_as.biases[l][0])):
            old_b = model_as.biases[l][0][ii]
            new_b = old_b + delta
            model_as.biases[l][0][ii] = new_b

            new_loss = compute_loss()
            dL = base_loss - new_loss
            db = old_b - new_b
            numerical_biases_grad_as[l][0][ii] = dL / db

            model_as.biases[l][0][ii] = old_b
            
    numerical_weights_grad_ph = [np.zeros(w.shape) for w in model_ph.weights]
    numerical_biases_grad_ph = [np.zeros(b.shape) for b in model_ph.biases]
    for l in range(len(model_ph.weights)):
        for ix, iy in np.ndindex(model_ph.weights[l].shape):
            old_w = model_ph.weights[l][ix][iy]
            new_w = old_w + delta
            model_ph.weights[l][ix][iy] = new_w

            new_loss = compute_loss()
            dL = base_loss - new_loss
            dw = old_w - new_w
            numerical_weights_grad_ph[l][ix][iy] = dL / dw

            model_ph.weights[l][ix][iy] = old_w

        for ii in range(len(model_ph.biases[l][0])):
            old_b = model_ph.biases[l][0][ii]
            new_b = old_b + delta
            model_ph.biases[l][0][ii] = new_b

            new_loss = compute_loss()
            dL = base_loss - new_loss
            db = old_b - new_b
            numerical_biases_grad_ph[l][0][ii] = dL / db

            model_ph.biases[l][0][ii] = old_b

    return numerical_weights_grad_as, numerical_biases_grad_as, numerical_weights_grad_ph, numerical_biases_grad_ph

predict_as=True

if predict_as:

# load arsenic data
    import pandas as pd
    as_data = pd.read_csv('training_datasets/arsenic_class.csv')
    as_data = as_data.rename(columns={'avg_welldepth':'depth'})
    ph_data = pd.read_csv('training_datasets/pH_class.csv')
    
    # use fewer predictors for pH
    ph_data = ph_data[['DMppt8110','LP6','Longitude','Latitude','depth','lay1_h0','pHclass']]
    X_as = (as_data.drop(['Asclass','lay1_dh','lay3_dh'],1))
    y_as = (as_data['Asclass'])
    
    X_ph = (ph_data.drop(['pHclass'],1))
    X_ph_array = np.array(X_ph)
    y_ph = (ph_data['pHclass'])
    
    col_inds = list(X_as.columns.get_loc(col) for col in X_ph.columns)
    
    scaler = StandardScaler()
    X_as1 = scaler.fit_transform(X_as)
    scale = scaler.scale_
    mean = scaler.mean_
    
    X_as = pd.DataFrame(X_as1,columns = list(X_as.columns.values))
    
    for kk, (scalec, meanc) in enumerate(zip(scale[col_inds],mean[col_inds])):
        print(scalec)
        print(meanc)
        X_ph_array[:,kk] = (X_ph_array[:,kk]-meanc)/scalec
        
    
    X_ph = pd.DataFrame(X_ph_array,columns = list(X_ph.columns.values))
    
    input_size = X_as.shape[1]+1
    hidden_sizes = [8]
    output_size = np.max(y_as)+1
    n_classes = output_size
    model_as = NeuralNetwork(input_size, hidden_sizes, output_size)
    model_ph = NeuralNetwork(X_ph.shape[1], hidden_sizes, output_size)
    x_as=np.array(X_as)
    y_as=np.array(y_as)
    
    x_ph = np.array(X_ph)
    y_ph = np.array(y_ph)
    
else:
    
    input_size = 2
    hidden_sizes = [8]
    output_size = 4
    model = NeuralNetwork(input_size, hidden_sizes, output_size)
    
    n_examples = 1000
    n_classes = 4
    x = np.random.randn(n_examples, input_size)
    y = np.random.randint(0, n_classes, size=(n_examples))

batch_size = 1

numerical_weights_grad_as, numerical_biases_grad_as,numerical_weights_grad_ph, numerical_biases_grad_ph = compute_numerical_gradient(
    model_ph, x_ph, y_ph, model_as, x_as, y_as, col_inds, 
    batch_size, n_classes, delta=1e-8
)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

def generate_area_map(features, points_per_int=10, alpha=0.2):
    xstart = int((features[:, 0].min() - 1) * points_per_int)
    xrang = int((features[:, 0].max() + 1) * points_per_int - xstart)

    ystart = int((features[:, 1].min() - 1) * points_per_int)
    yrang = int((features[:, 1].max() + 1) * points_per_int - ystart)

    area_map_set = np.array(
        [[x + xstart, y + ystart] for x in range(xrang) for y in range(yrang)]
    )
    area_map_set = area_map_set / points_per_int

    return area_map_set


def area_map_plot(network, area_map_set, features, targets, path="", alpha=0.1):
    pred = network.predict(area_map_set)

    plt.scatter(features[:, 0], features[:, 1], c=targets, cmap="jet")
    plt.scatter(area_map_set[:, 0], area_map_set[:, 1], c=pred, alpha=alpha, cmap="jet")

    if path == "":
        plt.show()
    else:
        plt.savefig(path)
        plt.close()


X_train_as, X_test_as, y_train_as, y_test_as = train_test_split(
    x_as, y_as, test_size=0.25, stratify=y_as,random_state=12345,
)

X_train_ph, X_test_ph, y_train_ph, y_test_ph = train_test_split(
    x_ph, y_ph, test_size=0.25, stratify=y_ph,random_state=12345,
)

print("train set")
_ = plt.scatter(X_train_as[:, 0], X_train_as[:, 1], c=y_train_as, cmap="jet")

print("test set")
_ = plt.scatter(X_test_as[:, 0], X_test_as[:, 1], c=y_test_as, cmap="jet")

# model = NeuralNetwork(input_size, hidden_sizes, output_size)

# area_map_set = generate_area_map(X_test, points_per_int = 7)
# area_map_plot(model, area_map_set, X_test, y_test, alpha = 0.15)

# print("Accuracy on Train Set:", np.mean(model.predict(X_train_as) == y_train_as))
# print("Accuracy on Test Set:", np.mean(model.predict(X_test_as) == y_test_as))
def numerical_gradient_descent(model_ph, x_ph, y_ph, model_as, x_as, y_as, 
                               col_inds, lr, batch_size, num_classes):
    numerical_weights_grad_as, numerical_biases_grad_as, numerical_weights_grad_ph, numerical_biases_grad_ph = compute_numerical_gradient(
        model_ph, x_ph, y_ph, model_as, x_as, y_as, col_inds, 
        batch_size, num_classes
    )
    for l in range(len(model_as.weights)):
        model_as.weights[l] += lr * -numerical_weights_grad_as[l]
        model_as.biases[l] += lr * -numerical_biases_grad_as[l]
        
    for l in range(len(model_ph.weights)):
        model_ph.weights[l] += lr * -numerical_weights_grad_ph[l]
        model_ph.biases[l] += lr * -numerical_biases_grad_ph[l]
    
    y_one_hot_ph = one_hot_encode(y_ph,num_classes)
    y_one_hot_as = one_hot_encode(y_as,num_classes)
    
    loss = mse(model_ph,x_ph,y_ph,y_one_hot_ph,
               model_as,x_as,y_as,y_one_hot_as,col_inds)

    return loss

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score

loss_history = []
acc_history = []
auc_mean=0.5
for epoch in range(500):
    # if auc_mean<=0.5:
    #     lr=2
    # if np.logical_and(auc_mean>0.5,auc_mean<=0.6):
    #     lr=1
    # if auc_mean>0.6:
    #     lr=0.5
    loss = numerical_gradient_descent(model_ph, X_train_ph, y_train_ph, 
                                      model_as, X_train_as, y_train_as, 
                                   col_inds, lr=0.5, batch_size=1, num_classes=output_size)
    # loss = numerical_gradient_descent(model, X_train, y_train, lr=1.0, batch_size=1, num_classes=output_size)
    
    yhat_ph_as = model_ph.forward(X_test_as[:,col_inds])
    yprob_ph_as = yhat_ph_as[:,1].reshape(len(yhat_ph_as),1)
    
    X_test_as_update = np.hstack((X_test_as,yprob_ph_as))
    
    acc = np.mean(model_as.predict(X_test_as_update) == y_test_as)
    auc_as = roc_auc_score(y_test_as,model_as.forward(X_test_as_update)[:,1])
    
    auc_ph = roc_auc_score(y_test_ph,model_ph.forward(X_test_ph)[:,1])
    
    auc_mean=np.mean([auc_as,auc_ph])
    
    loss_history.append(loss)
    acc_history.append(acc)
    if epoch % 10 == 0:
        print(f"epoch: {epoch:<6} loss: {loss:.6f}     auc (as): {auc_as:.4f}   auc (ph): {auc_ph:.4f}")


yhat_ph_as = model_ph.forward(X_test_as[:,col_inds])
yprob_ph_as = yhat_ph_as[:,1].reshape(len(yhat_ph_as),1)

X_test_as_update = np.hstack((X_test_as,yprob_ph_as))

yhat = model_as.predict(X_test_as_update)
yprobs = model_as.forward(X_test_as_update)

scores = np.zeros((50,2))
threshold = np.linspace(0,1,50)

for kk in range(scores.shape[0]):
    yhat = yprobs[:,1]>threshold[kk]

    auc= roc_auc_score(y_test_as,yprobs[:,1])
    kappa = cohen_kappa_score(yhat,y_test_as)
    f1 = f1_score(y_test_as,yhat)
    scores[kk,0]=kappa
    scores[kk,1]=f1
    
plt.figure();plt.plot(threshold,scores[:,0])
plt.plot(threshold,scores[:,1])
plt.legend(['kappa','f1'])

ind = np.argmax(scores[:,1])
print('f1 score: '+str(scores[ind,1]))
print('kappa score: '+str(scores[ind,0]))
print('auc: '+str(auc))

yhat = yprobs[:,1]>threshold[ind]
c = confusion_matrix(y_test_as,yhat)
print(c)

# make raster predictions ----------------------------------------------------
import rasterio
import geopandas as gpd
from rasterio.mask import mask

df_rast = pd.read_csv('raster_prediction/raster_dataframe.csv')
rast_array = np.array(df_rast)

as_cols = list(X_as.columns)
col_inds_as = list(df_rast.columns.get_loc(col) for col in as_cols)

rast_array = rast_array[:,col_inds_as]

for kk, (scalec, meanc) in enumerate(zip(scale,mean)):
    rast_array[:,kk] = (rast_array[:,kk]-meanc)/scalec

df_rast = pd.DataFrame(rast_array,columns = list(X_as.columns.values))

r0 = rasterio.open('C:/Users/smithrg/OneDrive - Colostate/Research/NIH/geospatial_data/Lombard/inputs_slv/p_c.tif')

def mask_raster_with_geometry(raster, transform, shapes, **kwargs):
    """Wrapper for rasterio.mask.mask to allow for in-memory processing.

    Docs: https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html

    Args:
        raster (numpy.ndarray): raster to be masked with dim: [H, W]
        transform (affine.Affine): the transform of the raster
        shapes, **kwargs: passed to rasterio.mask.mask

    Returns:
        masked: numpy.ndarray or numpy.ma.MaskedArray with dim: [H, W]
    """
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=raster.shape[1],
            width=raster.shape[2],
            count=raster.shape[0],
            dtype=raster.dtype,
            transform=transform,
        ) as dataset:
            dataset.write(raster)
        with memfile.open() as dataset:
            output, transform = mask(dataset, shapes, **kwargs)
    return output, transform

def saveraster_with_transform(data,fname,transform,crs='',drivername='GTiff',epsg='',datatype='float32',bands=1):
    # data_new=np.reshape(data,(bands,data.shape[0],data.shape[1]))
    if crs !='':
        crs=rasterio.crs.CRS.from_proj4(crs)
    if epsg!='':
        crs=rasterio.crs.CRS.from_epsg(epsg)
    with rasterio.open(
            fname,'w',driver=drivername,
            height=data.shape[0],
            width=data.shape[1],
            count=bands,
            dtype=datatype,
            crs=crs,
            transform=transform) as dst:
        dst.write(np.float32(data),bands)

def predict_raster(model,df,raster,pred_name,shp_to_clip=pd.Series([]),probability=True,col_inds=None):
    if col_inds!=None:
        # ipdb.set_trace()
        df_new = df.iloc[:,col_inds]
    else:
        df_new = df
    filt = df_new.isnull().any(1)
    df_new[filt] = 0
    pred = model.forward(np.array(df_new))
    # pred = np.array(predict_dataset(df_new,network))
    pred[filt] = np.nan
    df[pred_name] = pred[:,1]
    pred1d = np.array(df[pred_name]).reshape((len(df),1))
    # ipdb.set_trace()
    pred_rast = np.reshape(pred1d,raster.shape)
    tr = raster.transform
    
    if shp_to_clip.empty:
        print('no clipping')
        
    else:
        pred_rast,tr = mask_raster_with_geometry(np.reshape(pred_rast,(1,pred_rast.shape[0],pred_rast.shape[1])),
                                                      raster.transform,shp_to_clip,nodata=np.nan,crop=True)
    return(pred_rast,tr,df)


slv = gpd.read_file('C:/Users/smithrg/OneDrive - Colostate/Research/arsenic/san_luis_valley/model_layers/bound.shp')
slv_tr = slv.to_crs(crs = '+proj=aea +lat_0=37.5 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')
slv_geom = slv_tr.geometry

prob_pH_rast,tr,df_rast = predict_raster(model_ph,df_rast,r0,'pH',col_inds=col_inds)
df_rast['pH'] = prob_pH_rast.flatten()
prob_as_rast,tr,df_rast = predict_raster(model_as,df_rast,r0,'as',shp_to_clip=slv_geom)

prob_pH_clipped,tr = mask_raster_with_geometry(np.reshape(prob_pH_rast,(1,prob_pH_rast.shape[0],prob_pH_rast.shape[1])),
                                              r0.transform,slv_geom,nodata=np.nan,crop=True)

ph_shp = gpd.read_file('C:/Users/smithrg/OneDrive - Colostate/Research/NIH/geospatial_data/arsenic/machine_learning/training_datasets/shapefiles/ph_class.shp')
ph_shp2 = ph_shp.to_crs(crs = '+proj=aea +lat_0=37.5 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')

as_shp = gpd.read_file('C:/Users/smithrg/OneDrive - Colostate/Research/NIH/geospatial_data/arsenic/arsenic_medians_kathy_min2.shp')
as_shp = as_shp.set_crs(epsg='4326')
as_shp2 = as_shp.to_crs(crs = '+proj=aea +lat_0=37.5 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')
as_shp2['As_class'] = as_shp2['Median']>10

fig,ax=plt.subplots()
show(prob_pH_clipped,transform=tr,ax=ax)
ph_shp2.plot(edgecolor='black',ax=ax,markersize=8)
ph_shp2.plot(column='pHclass',ax=ax,markersize=7)

fig,ax=plt.subplots()
show(prob_as_rast,transform=tr,ax=ax)
as_shp2.plot(edgecolor='black',ax=ax,markersize=8)
as_shp2.plot(column='As_class',ax=ax,markersize=5)

saveraster_with_transform(np.squeeze(prob_pH_clipped), 'ph_joint_backprop.tif', tr,crs = '+proj=aea +lat_0=37.5 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')
saveraster_with_transform(np.squeeze(prob_as_rast), 'as_joint_backprop.tif', tr,crs = '+proj=aea +lat_0=37.5 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')
