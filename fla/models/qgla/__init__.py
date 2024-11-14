# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.qgla.configuration_qgla import QGLAConfig
from fla.models.qgla.modeling_qgla import QGLAForCausalLM, QGLAModel

AutoConfig.register(QGLAConfig.model_type, QGLAConfig)
AutoModel.register(QGLAConfig, QGLAModel)
AutoModelForCausalLM.register(QGLAConfig, QGLAForCausalLM)
print("Register QGLAConfig, QGLAModel, QGLAForCausalLM")

__all__ = ['QGLAConfig', 'QGLAForCausalLM', 'QGLAModel']
