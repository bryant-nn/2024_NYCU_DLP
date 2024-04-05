# 2024_NYCU_DLP

## train data
### main.py:
    agent = BaseModel(100, config,'ResNet50')
    agent.train_model()

## test data
### main.py:
    agent.load('path to best_model')
    agent.evaluate_model(is_valid=False)