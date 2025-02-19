"""
TRAINING:

FEATURES="token_ort"
python3 \
    ./models/metric_learning/train.py \
        --lang all \
        --save-model-path \"computed/models/rnn_metric_learning_${FEATURES}_all.pt\" \
        --number-thousands 10000 \
        --target-metric \"l2\" \
        --features ${FEATURES} \
        --epochs 20 \
    ;

scp euler:/cluster/work/sachan/vilem/pwesuite/computed/models/* ./computed/models/
"""

from models.metric_learning.model import RNNMetricLearner
from models.metric_learning.preprocessor import preprocess_dataset_foreign
from main.utils import load_multi_data
import torch
import tqdm
import math


data = load_multi_data(purpose_key="all")
data = preprocess_dataset_foreign(data[:10], features="token_ipa")

model = RNNMetricLearner(
    dimension=300,
    feature_size=data[0][0].shape[1],
)
model.load_state_dict(torch.load("computed/models/rnn_metric_learning_token_ipa_all.pt"))

# some cheap paralelization
BATCH_SIZE = 32
data_out = []
for i in tqdm.tqdm(range(math.ceil(len(data) / BATCH_SIZE))):
    batch = [f for f, _ in data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]]
    data_out += list(
        model.forward(batch).detach().cpu().numpy()
    )

assert len(data) == len(data_out)

print(len(data_out[0]))

assert all([len(x) == 300 for x in data_out])