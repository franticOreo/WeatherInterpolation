# WeatherInterpolation
## Small Research Project for a Client.

### Overview
Client deals with various different Metereological data streams, these streams suffer occassionally from sensor error/irregularities. Client was interested in the use of Machine Learning/Deep Learning techniques ability to interpolate the missing data caused from sensor malfunctions, opposed to conventional techniques such as Linear and Polynomial methods.

**Structure of Analysis**<br>
Meterological data is sourced from a kaggle dataset found [here](https://www.kaggle.com/selfishgene/historical-hourly-weather-data).
This dataset was chosen due to it's hourly resolution, this was the closest resolution to the clients description in the brief. Various Meterological data features have been joined for the city of *Los Angeles* including; Temperature, Humidity, Pressure, Wind Direction and Wind Speed.

**Artifically Noisy Data Creation**<br>
As this problem is concerned with different interpolation techniques performances, it was then essential to create an artificially noisy dataset to simulate sensor error. We, are then able to validate our results between the *noisy* dataset and the *complete* dataset(original dataset). In order to distribute noise throughout the dataset, I created a sinusoidal-like distrubution which was then fed into ```np.choice```. The intention of this distribution was to simulate error occuring in groups opposed to the default uniform distribution provided by NumPy.

**Benchmarking Conventional Interpolation Methods** <br>
Various different interpolation techniques provided by ```pandas.Dataframe.interpolate``` where then benchmarked using Mean Squared Error(MSE) between the *noisy* dataset and the *complete* dataset. ```pandas.Dataframe.interpolate``` interpolation techniques that were tested include: ```'linear', 'index', 'pad', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'spline', 'polynomial', 'piecewise_polynomial', 'spline', 'pchip', 'akima', 'cubicspline', 'from_derivatives'```.

Depending on the random state of Numpy, ```'akima'``` and ```quadratic``` often reported the lowest MSE.

**Deep Learning Models** <br>
**TBA**
