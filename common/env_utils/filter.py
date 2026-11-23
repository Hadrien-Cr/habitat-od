import pandas as pd

sizes_array = pd.read_csv("common/env_utils/procthor100_sizes.csv")
occurence_array = pd.read_csv("common/env_utils/procthor100_occurences.csv")


def dict_name_to_key(array, key: str):
    print(sizes_array.head())
    int2name = sizes_array["class"].to_dict()
    int2key = sizes_array[key].to_dict()
    return {v: int2key[k] for k, v in int2name.items()}    

d = dict_name_to_key(sizes_array, "max_largest_edge")
thresh_value = sorted(d.values())[40]
filter_fn = lambda x: d[x] <= thresh_value
print([k for k, v in d.items() if filter_fn(k)]) 
# ['apple', 'basket ball', 'bottle', 'bowl', 'bread', 'butter knife', 'candle', 'cell phone', 'cloth', 'credit card', 'cup', 'dish sponge', 'dumbbell', 'egg', 'fork', 'kettle', 'key chain', 'knife', 'ladle', 'lettuce', 'mug', 'newspaper', 'paper towel roll', 'pen', 'pencil', 'pepper shaker', 'plate', 'potato', 'remote control', 'salt shaker', 'soap bar', 'soap bottle', 'spatula', 'spoon', 'spray bottle', 'tissue box', 'toilet paper', 'tomato', 'vase', 'watch', 'wine bottle']