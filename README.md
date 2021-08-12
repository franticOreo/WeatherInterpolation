# WeatherInterpolation
## Small Research Project for a Client.

### Overview
A client deals with various different Metereological data streams, these streams suffer occassionally from sensor error/irregularities. The client was interested in the use of Machine Learning/Deep Learning methods and there ability to interpolate the missing data caused from sensor malfunctions and how they performed against conventional techniques such as Linear and Polynomial methods.

**Structure of Analysis**<br>
Meterological data is sourced from a kaggle dataset found [here](https://www.kaggle.com/selfishgene/historical-hourly-weather-data).
This dataset was chosen due to it's hourly resolution, this was the closest resolution to the clients description in the brief. Various Meterological data features have been joined for the city of *Los Angeles* including; Temperature, Humidity, Pressure, Wind Direction and Wind Speed. I selected to interpolate the temperature feature. 

**Artificial Sensor Error Creation**<br>
As this problem is concerned with different interpolation techniques performances, it was then essential to create an artificially noisy dataset to simulate sensor error. We, are then able to validate our results between the artificial *gap* dataset and the *complete* dataset(original dataset). In order to distribute noise throughout the dataset, I created a sinusoidal-like distrubution which was then fed into ```np.choice```. The intention of this distribution was to simulate error occuring in groups opposed to a uniform distribution which would create random *salt and pepper-like* gaps. Below you can see a random subset of the data plotted with artificial gaps agaisnt the complete data.

![create_gap_df](/img/create_gap_df.png)

**Benchmarking Conventional Interpolation Methods** <br>
Various different interpolation techniques provided by ```pandas.Dataframe.interpolate``` where then benchmarked using Mean Squared Error(MSE) between the *noisy* dataset and the *complete* dataset. ```pandas.Dataframe.interpolate``` interpolation techniques that were tested include: ```'linear', 'index', 'pad', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'spline', 'polynomial', 'piecewise_polynomial', 'spline', 'pchip', 'akima', 'cubicspline', 'from_derivatives'```.

Depending on the random state of Numpy, ```'akima'``` and ```quadratic``` often reported the lowest MSE.

**Deep Learning Models** <br>
**TBA**
