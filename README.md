# Setup
`pip install -r requirements.txt`
# Model Creation
`python pl-csb-model train`
# Translation using the created model
`python pl-csb-model translate`
# Configuration

Pretrained model, output model name as well as training settings can be all configured inside `config.ini` file.

# Config.ini parameters
Batchsize should in general correspond to the number of memory of the used training device, ie. BatchSize=8 for a 8GB RAM GPU card.
