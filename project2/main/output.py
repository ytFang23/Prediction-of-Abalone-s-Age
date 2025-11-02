import pandas as pd


def dicts_to_pd(dicts, index):
    if len(dicts) == 0:
        return None

    common_keys = set(dicts[0].keys())

    for d in dicts[1:]:
        common_keys &= set(d.keys())

    for i, each in enumerate(dicts):
        dicts[i] = {key: each[key] for key in common_keys}

    df = pd.DataFrame(dicts, index=index).T
    return df


def df_expand(df):
    df_expanded = None
    for col in df.columns:
        if isinstance(df[col].iloc[0], dict):
            _df = pd.json_normalize(df[col])
            _df.index = df.index
        else:
            _df = pd.DataFrame(df[col])
        # print(_df)
        if df_expanded is None:
            df_expanded = _df
        else:
            df_expanded = pd.concat([df_expanded, _df], axis=1)
    return df_expanded


def list_to_csv(lists, index, file_path):
    df = pd.DataFrame(lists,index = index).T
    df_expanded = df_expand(df)
    df_expanded.to_csv(file_path)


if __name__ == '__main__':
    pass





