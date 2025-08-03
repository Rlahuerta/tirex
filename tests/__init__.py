import os
from pathlib import Path

# Define the project root relative to this file
project_path = Path(__file__).parent

# Define paths for test-related directories
unit_test_path = project_path / "tests"
unit_test_plots_path = project_path / "plots"
test_data_path = project_path / "data"

# Define a specific path for a test data file
full_timeseries_data = test_data_path / "btcusd_2022-06-01.joblib"

# Create the plots directory if it doesn't exist.
# Note: It is often better to handle this in a pytest fixture.
os.makedirs(unit_test_plots_path, exist_ok=True)
