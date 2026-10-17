import numpy as np
import pandas as pd
import pytest

from src.evaluation.walkforward import walk_forward_splits


def _df():
    seasons = np.repeat([2018, 2019, 2020, 2021, 2022], 10)
    return pd.DataFrame({"season": seasons, "date": pd.to_datetime("2018-08-01")
                         + pd.to_timedelta(np.arange(50), unit="D")})


def test_expanding_window():
    splits = list(walk_forward_splits(_df(), min_train_seasons=3))
    assert [s for s, _, _ in splits] == [2021, 2022]
    _, tr, te = splits[0]
    assert len(tr) == 30 and len(te) == 10


def test_train_strictly_before_test_and_no_overlap():
    df = _df()
    for _, tr, te in walk_forward_splits(df, min_train_seasons=2):
        assert set(tr).isdisjoint(te)
        assert df.iloc[tr]["date"].max() < df.iloc[te]["date"].min()


def test_not_enough_seasons():
    with pytest.raises(ValueError):
        list(walk_forward_splits(_df(), min_train_seasons=5))
