# WeatherInterpolation
## Small Research Project for a Client.

### Overview
A client deals with various different Metereological data streams, these streams suffer occassionally from sensor error/irregularities. The client was interested in the use of Machine Learning/Deep Learning methods and there ability to interpolate the missing data caused from sensor malfunctions and how they performed against conventional techniques such as Linear and Polynomial methods.

**Structure of Analysis**<br>
Meterological data is sourced from a kaggle dataset found [here](https://www.kaggle.com/selfishgene/historical-hourly-weather-data).
This dataset was chosen due to it's hourly resolution, this was the closest resolution to the clients description in the brief. Various Meterological data features have been joined for the city of *Los Angeles* including; Temperature, Humidity, Pressure, Wind Direction and Wind Speed. I selected to interpolate the temperature feature. 

**Artificial Sensor Error Creation**<br>
As this problem is concerned with different interpolation techniques performances, it was then essential to create an artificially noisy dataset to simulate sensor error. We, are then able to validate our results between the artificial *gap* dataset and the *complete* dataset(original dataset). 

In order to distribute noise throughout the dataset, I have created two methods to generate artificial gaps in a series both; `create_variable_gap_df` and `create_fixed_gap_df`. `create_variable_gap_df` is arguably a more realistic simulation of sensor error wheras `create_fixed_gap_df` is more effective for experimenting as the gap size is fixed.

For `variable_gap_df`, a sine transformation is applied to the *complete* data, the sinusoisal distrubution is then fed into ```np.choice```. The intention of this distribution was to simulate error occuring in groups opposed to a uniform distribution which would create random *salt and pepper-like* gaps. Below you can see a random subset of the data plotted with artificial gaps agaisnt the complete data, note the red *Missing* points are often neighbouring.

![create_gap_df](/img/create_gap_df.png)

**Benchmarking Conventional Interpolation Methods** <br>
Various different interpolation techniques provided by ```pandas.Dataframe.interpolate``` where then benchmarked using Mean Squared Error(MSE) between the *noisy* dataset and the *complete* dataset. ```pandas.Dataframe.interpolate``` interpolation techniques that were tested include: ```'linear', 'index', 'pad', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'spline', 'polynomial', 'piecewise_polynomial', 'spline', 'pchip', 'akima', 'cubicspline', 'from_derivatives'```.

Depending on the random state of Numpy, ```'akima'``` and ```quadratic``` often reported the lowest Mean Squared Error(MSE). It's important to note MSE is comparing the test split of the data against the test split of data with artificial gaps filled in by interpolation. Therefore, MSE results are considerably low as all of the non-gap values are identical. If we refer back to our previous picture you can notice that even though four consequective *Missing* points can easily be interpolated by a line or with a slight curvature. Therefore, methods such as `akima` score an MSE close to 0. 

Unsurprisingly, ML/DL methods will struggle to outperform with these conventional methods due to the nature of the `interpolate` method (or interpolation in general), it notices a `NaN`\s grabs a point A prior to the `Nan` and the next non `NaN` value, B. Then a function is then applied, a line/curve is fitted and the series is filled between A and B. I am unaware of any ML/DL methods that operate under this way, traditionally ML/DL models take in historical data and predict into the future, however there is no consideration of B. Regardless, these experiments have been conducted and can be viewed in `WeatherInterpolation.ipynb`.

#### Hmm... 
What about if the sensor was down for 6 hours. Then what would a conventional method do? 


![spline6](/img/spline_gap_size_6.png)

**Can a DL model beat this?**

Unfortuneatly, not with my implementations, MSE's even from sophisticated models employing Gated Recurrent Units, and 1D Convolutional layers were unable to beat the best performing conventional methods. As mentioned previously, this I believe is mainly due to the fact the model comparison between conventional methods and Machine Learning methods is realtively incomparable as the conventional method is able to seek forward an arbitrary amount of values and fit a curve, a ML method is predictly relatively blindly! Hence, curves being fit like so:

![conv_model](/img/tf_conv_gap_size_6.png)

For a sensor error of size 6, the Mean Squared Errors are as follows:

![mses](/img/error_bars_6.png)
