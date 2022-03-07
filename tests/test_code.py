import pandas as pd

from pathlib import Path

from code.dat_to_json import make_dataset
from tests import __input_path__ as input_dir
from tests import __output_path__ as output_dir


# run the code: "pytest -s -v tests/test_code.py::test_make_dataset"
def test_make_dataset():
  load_dir = input_dir.joinpath("physionet.org/files/autonomic-aging-cardiovascular/1.0.0")
  save_dir = output_dir.joinpath("data")

  master_df = pd.read_csv(load_dir.joinpath("subject-info.csv"))
  
  make_dataset(load_dir, save_dir, master_df)
