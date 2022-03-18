# Accepted by AAAI 2022, BibTex will be added soon.

# Settings in main.py
    data = 'Caltech'

# The following settings are adopted to all datasets
    Lc = 1.0
    Lr = 1.0
    lrate = 0.001
    epochs = 500
    Update_epoch = 1000
    Max_iteration = 10
    Batch = 256

# The datasets with different miss rates are generated each time
    for missrate in [0.1, 0.3, 0.5, 0.7]:

# Run the code by
    python main.py

# Requirement
    python==3.7.10
    scikit-learn==0.22.2.post1
    scipy==1.4.1
    tensorflow-gpu==2.5.0
