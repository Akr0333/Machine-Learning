from src.generate_data import generate_dataset
from src.train import build_pipeline


def test_generate_dataset_shape_and_columns():
    df = generate_dataset(n_samples=100)
    assert len(df) == 100
    assert set(df.columns) == {
        "area_sqft", "bedrooms", "bathrooms", "age_years",
        "distance_km", "location", "parking", "furnished", "price"
    }
    assert (df["price"] > 0).all()


def test_pipeline_can_fit_and_predict():
    df = generate_dataset(n_samples=120)
    features = [
        "area_sqft", "bedrooms", "bathrooms", "age_years",
        "distance_km", "location", "parking", "furnished"
    ]
    pipeline = build_pipeline()
    pipeline.fit(df[features], df["price"])
    prediction = pipeline.predict(df[features].head(1))
    assert len(prediction) == 1
    assert prediction[0] > 0
