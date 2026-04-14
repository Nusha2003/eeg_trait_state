from moabb.datasets import PhysionetMI

dataset = PhysionetMI()

dataset.download(
    path="/scratch1/amadapur/data/physionet",
    update_path=True,
    force_update=True
)