from moabb.datasets import Lee2019_MI
import numpy as np
import os
save_dir = "/scratch1/amadapur/data/meng_et_al/data"
dataset = Lee2019_MI()
for subject in dataset.subject_list:
    data = dataset.get_data(subjects=[subject])
    for key in data.keys():
        X = data[key]['1train']
        np.save(os.path.join(save_dir, f"subject{subject}_class{key}"))
        print(f"saved subject {subject} class {key}")