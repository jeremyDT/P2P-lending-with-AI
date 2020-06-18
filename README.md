# P2P-lending-with-AI
Repository with code and data associated with the paper: P2P Loan acceptance and default prediction with Artificial Intelligence

Author: Jeremy Turiel - jeremy.turiel@gmail.com

All codes in the folder are meant to be run with working directory set as per their location.

The following codes (in the code folder) load these files (in the data folder):

- code_1_acceptance.py LOADS input_file_1.csv
- code_2_default_regression_only.py LOADS input_file_2.csv
- code_3_default_neural_network_only.py LOADS input_file_2.csv
- code_3_default_neural_network_only_GRID_SEARCH.py LOADS input_file_2.csv
- code_3_default_neural_network_only_FEATURE_IMPORTANCE.py LOADS input_file_2.csv
- code_3_default_neural_network_only_XAI_PDP.py LOADS input_file_2.csv

The codes are to be run in logical order according to their numbers, but do not depend on any of the above scripts being run before. The ordering respects how results are presented in the paper.

Data is located in the "data" folder.

input_file_1.csv contains cleaned data for accepted and rejected loans

input_file_2.csv contains cleaned data for accepted loans for default analysis

The file Paper_arxiv_version.pdf contains the original paper which is on ArXiV at https://arxiv.org/abs/1907.01800
The paper is now published on Royal Society Open Science https://royalsocietypublishing.org/doi/10.1098/rsos.191649