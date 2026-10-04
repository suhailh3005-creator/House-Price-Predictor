# House Price Predictor

This project is a small, beginner-friendly machine-learning example. It teaches the full workflow in a way that is easy to follow: load data, look at it, clean it, engineer a couple of useful features, split it into training and test sets, train a regression model, and make a prediction from a new set of house details.

The goal is not to build a real-estate product. The goal is to help someone understand the basic idea of ML without getting lost in unnecessary complexity.

## What is this?

This app learns from a small set of example houses and estimates the price of a new house using a few simple characteristics such as area, number of bedrooms, bathrooms, age, location score, and parking spaces.

It is a supervised learning problem called regression: instead of predicting a category like “cat” or “dog,” it predicts a number, which in this case is the sale price of a home.

## Why I built it

I wanted a project that demonstrates the actual machine-learning workflow without pretending to be bigger or more advanced than it is. The best beginner projects are small enough to inspect by hand, explain clearly, and run quickly.

This one is intentionally modest: it teaches a beginner how to think through data, model choice, evaluation, and prediction in a realistic but manageable way.

## What you'll learn

- What a dataset is and how to inspect it
- How to clean a small CSV dataset
- What a feature is and what a target is
- Why train/test splitting matters
- How to engineer simple features from existing data
- How a linear regression model learns relationships
- How to evaluate predictions with understandable metrics
- How to save a trained model and reuse it
- How a simple interface can turn a model into a usable tool

## What the model is actually doing

The model is looking for patterns in the data. If larger homes, more bedrooms, and better locations tend to have higher prices, the model learns that pattern and uses it when it sees a new house it has not seen before.

This is not magic. It is a statistical shortcut for noticing patterns in past examples.

A regression model is just a way of drawing a line or a more general curve that best fits the relationship between the house features and the house price. In this project, we use linear regression because it is simple, interpretable, and beginner-friendly.

## Project structure

```text
house-price-predictor/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── houses.csv
├── models/
│   └── house_price_model.joblib
├── notebooks/
│   └── house_price_eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
└── tests/
    └── test_project.py
```

## Getting started

### Installing the dependencies

Create a virtual environment if you want one, then install the project requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the project

Train the model:

```bash
python src/train.py
```

Launch the prediction interface:

```bash
streamlit run app.py
```

Run the tests:

```bash
pytest
```

## Understanding the dataset

The dataset in `data/houses.csv` is intentionally small and human-readable. It contains examples like these fields:

- `area_sqft`: total living area in square feet
- `bedrooms`: number of bedrooms
- `bathrooms`: number of bathrooms
- `age_years`: age of the house in years
- `location_score`: a simple score representing location quality
- `parking_spaces`: number of parking spaces
- `house_price`: the target we want to predict

This project uses a synthetic dataset, which means it is created for teaching rather than coming from a real estate listing service. That is helpful when you are learning because the numbers are easy to understand. It also means the results are educational, not real-world valuations.

## EDA: what we found

The notebook in `notebooks/house_price_eda.ipynb` walks through the project in a logical order:

1. Load the dataset.
2. Look at a few rows.
3. Check the shape and columns.
4. Look for missing values.
5. Summarize the columns.
6. Plot a price histogram.
7. Plot area against price.
8. Ask what patterns seem obvious.

A good beginner takeaway is that larger, newer, better-located homes generally cost more. That is exactly the kind of pattern a model learns from.

## Feature engineering, explained simply

A feature is a piece of information about a house. The target is the thing we want to predict.

This project adds two simple engineered features:

- `area_per_bedroom`: area divided by bedroom count. This helps the model compare how spacious each bedroom is in a home.
- `house_age_group`: house age grouped into a few categories such as new, middle-aged, and older. This makes the age information easier to interpret than a raw number alone.

Feature engineering is simply creating a more useful way to represent information that is already in the data.

## Why Linear Regression?

Linear regression is used as the baseline model because it is easy to explain. It learns a simple relationship between the house features and the target price.

For example, it might learn that higher area, more bathrooms, and better location are linked to higher prices. It does not need complicated machinery to show a beginner how the system learns.

## How the model is evaluated

After training, the model is checked on homes it has not seen before. This is important because a model that memorizes the training examples may look good on the training data but fail on new houses.

We evaluate the model using:

- MAE (Mean Absolute Error): on average, how far off the predictions are from the actual prices.
- RMSE (Root Mean Squared Error): similar to MAE, but it gives extra weight to larger mistakes.
- R2: how much of the variation in house prices is explained by the model.

These metrics tell us whether the model is learning a useful pattern rather than just memorizing examples.

## What the prediction means

When the app estimates a home price, it is giving a rough, educational guess based on sample houses it learned from. It is not a professional property valuation and it should not be treated as one.

This project is valuable because it teaches the thinking behind prediction, not because it is a polished real-estate tool.

## Limitations

This project has several important limitations:

- It uses a small synthetic dataset, not real market data.
- It ignores many real factors such as neighborhood, school quality, renovation status, and local market conditions.
- It is intentionally simple so beginners can understand it.
- The model can only work with the information it has been given.

A real house price depends on far more than six features.

## Things I would improve next

If I kept building this project, the next useful steps would be:

- add more realistic housing data
- test different models such as random forest regressor
- use a richer dataset with more neighborhood information
- improve the interface with better explanations and validation
- add a proper notebook walkthrough with more charts

But for a beginner project, this version is intentionally lean and easier to understand.

## Common beginner questions

### What is regression?

Regression means predicting a number. In this project, the number is the sale price of a house.

### Why do we split the data into train and test sets?

We hide some houses from the model during training so we can later check whether it can handle homes it has not seen before. That is a much more honest test than checking only the data used to train the model.

### What is a feature?

A feature is just a piece of information about a house, like area or number of bedrooms.

### What is the target?

The target is the value we want the model to predict, which here is the house price.

### Why is memorizing a bad idea?

A model that memorizes training examples can look impressive on the data it saw during training, but it often fails when it meets a new house. Learning means noticing general patterns, not copying exact examples.

## Learning resources

If you want to keep going, these topics are worth learning next:

- Python for data analysis with pandas
- Introductory statistics for data science
- Basic machine learning concepts like bias, variance, and overfitting
- Data visualization with matplotlib and seaborn
- Model evaluation and validation

Good beginner starting points include:

- Kaggle Learn
- scikit-learn documentation
- The Python Data Science Handbook
- The official pandas user guide

## Final note

This project is small on purpose. It is designed to teach the idea clearly: the model learns from examples, sees patterns, and then makes a rough estimate for something new. That is the heart of many practical machine-learning workflows.
