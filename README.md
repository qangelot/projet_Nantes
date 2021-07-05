
# projet_Nantes

Machine learning package to predict canteens attendance 2-3 weeks ahead in Nantes Metropole. The package focuses on predicting 

## Architecture

Data folder stores all the raw data used in this data science project.

Package folder stores all the config and python files necessary to build the machine learning package.

Research folder stores all the notebooks used to explore the available data files, build ETL, process the data, elaborate the best possible predictive model and analyze its results.

API folder stores all files necessary to build a functional Rest API to serve predictions of the Machine learning model in an application.



## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install attendance_model.

```bash
pip install attendance_model
```

## Usage

```python
import attendance_model

# returns trained pipeline
attendance_model.train_pipeline.run_training()

# returns predictions for the provided input data
attendance_model.predict.make_prediction(input_data)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)