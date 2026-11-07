from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from common import load_data, prepare_features

def test_dataset_and_target():
    df=load_data()
    assert len(df)>100
    assert "Attrition" in df.columns
    X,y=prepare_features(df)
    assert len(X)==len(y)
    assert set(y.unique()).issubset({0,1})
