from ray.rllib.models import ModelCatalog

from exchange_data.agents.tf_models.deep_lob import DeepLOBModel

ModelCatalog.register_custom_model("deep_lob", DeepLOBModel)
