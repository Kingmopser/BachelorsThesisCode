import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from bde import BdeClassifier
from bde.loss import CategoricalCrossEntropy

iris = load_iris()
X = iris.data.astype("float32")
y = iris.target.astype("int32").ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build the classifier
classifier = BdeClassifier(
    n_members=4,
    hidden_layers=[16, 16],
    seed=0,
    loss=CategoricalCrossEntropy(),
    activation="relu",
    epochs=1000,
    validation_split=0.15,
    lr=1e-3,
    warmup_steps=400,  # very few steps required for this simple dataset
    n_samples=100,
    n_thinning=1,
    patience=10,
)

# Fit the classifier
classifier.fit(x=X_train, y=y_train)

# Get results from classifier
preds = classifier.predict(X_test)
probs = classifier.predict_proba(X_test)
score = classifier.score(X_train, y_train)
raw = classifier.predict(X_test, raw=True) # (ensemble members, n_samples/n_thinning, n_test_data, n_classes)