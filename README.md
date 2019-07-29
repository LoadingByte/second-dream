# Second Dream

This project provides all the tools you need to let neural networks dream deeply on time series data.
It comes with a neat demo webapp.

## Usage

All of the core code for running dreams is contained in the top-level `second_dream.py` module.
Just import the module and use it according to the provided documentation.

The `notebook.ipynb` file contains a Jupyter notebook that demonstrates how to use the module.
Also look there if you want to see exemplary dream time series and the interesting insights they can give.
The notebook draws models and base time series data from `auxiliary/models/` and `auxiliary/testsets/`, respectively.

Finally, the `auxiliary/webapp/` folder holds a Flask webapp which is excellent for toying around with Second Dream and building intuition.
Just like the notebook, this webapp draws its data from the `auxiliary/` folder (by default).
You can launch a development instance by running the `webapp.py` script.
The config file is located at `auxiliary/webapp/settings.cfg`.
If you don't want to change this file directly, you can supply the path to an override config file in the environment variable `SD_SETTINGS`.
A production instance of the webapp is permanently hosted on https://loadingbyte.com/second-dream. Check it out!

## References

This code is roughly based on [this example](https://github.com/keras-team/keras/blob/master/examples/deep_dream.py) from the Keras project, but has been heavily modified.
The demo datasets are contributed by the [UCR Time Series Classification Archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/).
The code for the demo models is provided by [Fawaz et al. (2019)](https://github.com/hfawaz/dl-4-tsc), you can reference their accompanying survey paper at [this DOI](https://doi.org/10.1007/s10618-019-00619-1).
