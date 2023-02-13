preprocessor = make_column_transformer(
    (OneHotEncoder(), ['Sex']),
    (OrdinalEncoder(), ['Pclass']),
    (StandardScaler(), ['Fare'])
)
preprocessor.fit(with_age)
preprocessor.transform(with_age)
