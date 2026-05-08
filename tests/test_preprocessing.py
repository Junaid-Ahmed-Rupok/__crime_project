import pytest
import pandas as pd
import numpy as np
import sys
sys.path.append('..')

def test_outlier_capping():
    """Test IQR outlier capping"""
    df = pd.DataFrame({'value': [1, 2, 3, 4, 100]})
    # Your preprocessing function
    # assert ...
