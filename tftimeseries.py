import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from statsmodels.tsa.stattools import adfuller
import datetime
from sklearn.preprocessing import StandardScaler


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
        train_df, val_df, test_df,
        label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

WindowGenerator.split_window = split_window

def make_dataset(self, data, batch_size=32,):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=batch_size)

    ds = ds.map(self.split_window)

    return ds

WindowGenerator.make_dataset = make_dataset

@property
def train(self):
    return self.make_dataset(self.train_df)

WindowGenerator.train = train

@property
def val(self):
    return self.make_dataset(self.val_df)

WindowGenerator.val = val

@property
def test(self):
    return self.make_dataset(self.test_df)

WindowGenerator.test = test

@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result

WindowGenerator.example = example

def plot(self, model=None, plot_col=None, max_subplots=3):
    # get an example batch
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_n, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                         label='Inputs', marker='.', zorder=-10)

        if self.label_columns:    
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                                edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:     
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                                    marker='X', edgecolors='k', label='Predictions',
                                    c='#ff7f0e', s=64)      

        if n == 0:
            plt.legend()

    plt.xlabel('Time [h]')

WindowGenerator.plot = plot

class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]



def compile_and_fit(model, window, patience=2, epochs=20):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError(),
                        tf.metrics.MeanAbsolutePercentageError()])

    history = model.fit(window.train, epochs=epochs,
                      validation_data=window.val,
                      callbacks=[early_stopping])
    return history


def stationary_test(series, name=None, verbose=False, test=False):
    adft = adfuller(series,autolag="AIC")
    if verbose:
        print(f"Results of dickey fuller test - {name}")
        # output for dft will give us without defining what the values are.
        #hence we manually write what values does it explains using a for loop
        output = pd.Series(adft[0:4],index=["Test Statistics","p-value","No. of lags used","Number of observations used"])
        for key,values in adft[4].items():
            output["critical value (%s)"%key] = values
        print(output)
        
    if test:
        # extract p-value from test object
        if adft[1] > 0.05:
            return False # non-stationary
        else:
            return True # stationary.

def remove_seasonality(col, skip_cols, drop=False):
    """skip_cols (list)
    """
    
    if col.name in skip_cols:
        return col
    
    # check if col is stationary, if stationary skip col.
    if stationary_test(col, test=True) is True:
        return col
    
    # if column not number, skip col
    if not np.issubdtype(col.dtype, np.number): 
        return col

    if (col < 0).any(): # check if contains neg values
        # add constant, make values positive 
        col = col + col.min()
        
    """Spaghetti Code Alert:
    Can't find a way to invert the np.log(col + 1)
    """
    np.seterr(divide = 'ignore') 
    
    log_values = np.log(col + 1) # add 1 to avoid log(0)
    # bad code: fill NaN with 0
    log_values = np.nan_to_num(log_values)
    
    diff = np.diff(log_values, prepend=np.nan)
    
    if drop: 
        diff = diff.dropna()
    
    return diff

def invert_seasonality(seasonality_vals, last_cum_val):
    # log and add one 
    invert_vals = np.insert(seasonality_vals, 0, np.log(last_cum_val + 1)).cumsum()
    invert_vals = np.exp(invert_vals) - 1
    
    return invert_vals

def date_periodicity(df, date_col='date'):
    """Perform sinsoidal transformation on date column.
    """
    date_ts = df.pop(date_col).map(datetime.datetime.timestamp)

    day = 24*60*60
    year = (365.2425)*day

    # time of year : solve for model to interept periodicity.
    df['Year sin'] = np.sin(date_ts * (2 * np.pi / year))
    df['Year cos'] = np.cos(date_ts * (2 * np.pi / year))
    
    return df

def move_column_inplace(df, col, pos):
    col = df.pop(col)
    df.insert(pos, col.name, col)

def split_df(df):
    """Splits dataframe into standard size (70-20-10)
    training, validation and test datasets and returns
    them.
    """
    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]
    
    return train_df, val_df, test_df

def scale_datasets(train_df, val_df, test_df, y):

    if y is not None:
        train_price = train_df.pop(y)
        val_price = val_df.pop(y)
        test_price = test_df.pop(y)
    
    scaler = StandardScaler().fit(train_df)    
    
    # Scale all Independent variables
    train_scaled_vals = scaler.transform(train_df.values)
    val_scaled_vals = scaler.transform(val_df.values)
    test_scaled_vals = scaler.transform(test_df.values)

    train_df = pd.DataFrame(train_scaled_vals, columns=train_df.columns, index=train_df.index)
    val_df = pd.DataFrame(val_scaled_vals, columns=val_df.columns, index=val_df.index)
    test_df = pd.DataFrame(test_scaled_vals, columns=test_df.columns, index=test_df.index)
    
    if y is not None:
        # Merge price back onto Independent variables
        train_df = pd.merge(train_price, train_df, left_index=True, right_index=True)
        val_df = pd.merge(val_price, val_df, left_index=True, right_index=True)
        test_df = pd.merge(test_price, test_df, left_index=True, right_index=True)
    
    # return scaler for inverse_transform later?
    return train_df, val_df, test_df, scaler

def trim_drop_df(df_raw):
    """Reduce Dataframe by Timeperiods (future dates and dates outside of API limits)
    Drop high amounts of empty columns
    """
    df = df_raw.copy() # keep raw data
    df = df.rename(columns={'t': 'date'})
    df = df.select_dtypes(exclude=['object'])

    df['date'] = pd.to_datetime(df['date'])

    # 2010-07-17, no pricing data from API before this date.
    df = df[df['date'].dt.date > datetime.date(2010,7,17)]
    # Data set includes Tier 3 Metric that are only current to a year ago.
    date_of_scrape = datetime.date(2021,5,23)
    df = df[df['date'].dt.date < date_of_scrape - datetime.timedelta(weeks=55)]

    # # keep columns with no more than n% NA values
    df = df.dropna(thresh=0.80*len(df), axis=1)
    df = df.sort_values(by=['date'])
    df = df.reset_index(drop=True)

    # spaghetti code to remove columns with heaps of 0's
    # contradicts studio time frame
    lots_of_zeros = (df == 0).sum().to_frame().sort_values(by=0, ascending=False).head(8).index.to_list()
    df = df.drop(columns=lots_of_zeros)
    
    return df

def write_pickle(obj, fn):
    with open(fn, 'wb') as file:
        pickle.dump(obj, file)

def load_pickle(fn):
    with open(fn, 'rb') as file:
        return pickle.load(file)

def rand_slice_indicies(df, sample_size=100):
    """Create a start and end index to be used for 
    Slicing for input df. Limitation as last n-sample_size ... n, will
    be excluded.
    
    Args:
        df (Dataframe): Input dataframe
        sample_size (int): Width of slice range (default 100)
        
    Returns:
        start_index (int): Start of slice
        end_index (int): End of slice
    """
    start_index = np.random.randint(low=0, high=df.shape[0]-sample_size)
    end_index = start_index + sample_size

    return start_index, end_index


def fit_predict_mae(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    mae_train = mean_absolute_error(y_train, preds_train)
    mae_test = mean_absolute_error(y_test, preds_test)

    return preds_train, preds_test, mae_train, mae_test
