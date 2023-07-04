# import torch.nn as nn
# from .bireal import BiRealConv2d, BiRealConv1d, BiRealLinear
# from .xnor import XNORConv2d, XNORConv1d, XNORLinear
# from .bnn import BNNConv2d, BNNConv1d, BNNLinear
# from .react import ReActConv2d, ReActConv1d, ReActLinear
# from .dorefa import DoReFaConv2d, DoReFaConv1d, DoReFaLinear
# from .xnorplusplus import XNORPlusPlusConv2d, XNORPlusPlusConv1d, XNORPlusPlusLinear
# from .recu import ReCUConv2d, ReCUConv1d, ReCULinear
# from .fda import FDAConv2d, FDAConv1d, FDALinear

# from mmengine.registry import MODELS


# def init_layers():
#     MODELS.register_module('Hardtanh', module=nn.Hardtanh)
#     MODELS.register_module(name='Linear', module=nn.Linear)
    
#     MODELS.register_module(name='BiRealConv', module=BiRealConv2d)
#     MODELS.register_module(name='XNORConv', module=XNORConv2d)
#     MODELS.register_module(name='BNNConv', module=BNNConv2d)
#     MODELS.register_module(name='ReActConv', module=ReActConv2d)
#     MODELS.register_module(name='DoReFaConv', module=DoReFaConv2d)
#     MODELS.register_module(name='XNORPlusPlusConv', module=XNORPlusPlusConv2d)
#     MODELS.register_module(name='ReCUConv', module=ReCUConv2d)
#     MODELS.register_module(name='FDAConv', module=FDAConv2d)

#     MODELS.register_module(name='BiRealConv2d', module=BiRealConv2d)
#     MODELS.register_module(name='XNORConv2d', module=XNORConv2d)
#     MODELS.register_module(name='BNNConv2d', module=BNNConv2d)
#     MODELS.register_module(name='ReActConv2d', module=ReActConv2d)
#     MODELS.register_module(name='DoReFaConv2d', module=DoReFaConv2d)
#     MODELS.register_module(name='XNORPlusPlusConv2d', module=XNORPlusPlusConv2d)
#     MODELS.register_module(name='ReCUConv2d', module=ReCUConv2d)
#     MODELS.register_module(name='FDAConv2d', module=FDAConv2d)

#     MODELS.register_module(name='BiRealConv1d', module=BiRealConv1d)
#     MODELS.register_module(name='XNORConv1d', module=XNORConv1d)
#     MODELS.register_module(name='BNNConv1d', module=BNNConv1d)
#     MODELS.register_module(name='ReActConv1d', module=ReActConv1d)
#     MODELS.register_module(name='DoReFaConv1d', module=DoReFaConv1d)
#     MODELS.register_module(name='XNORPlusPlusConv1d', module=XNORPlusPlusConv1d)
#     MODELS.register_module(name='ReCUConv1d', module=ReCUConv1d)
#     MODELS.register_module(name='FDAConv1d', module=FDAConv1d)

#     MODELS.register_module(name='BNNLinear', module=BNNLinear)
#     MODELS.register_module(name='XNORLinear', module=XNORLinear)
#     MODELS.register_module(name='DoReFaLinear', module=DoReFaLinear)
#     MODELS.register_module(name='BiRealLinear', module=BiRealLinear)
#     MODELS.register_module(name='XNORPlusPlusLinear', module=XNORPlusPlusLinear)
#     MODELS.register_module(name='ReActLinear', module=ReActLinear)
#     MODELS.register_module(name='ReCULinear', module=ReCULinear)
#     MODELS.register_module(name='FDALinear', module=FDALinear)

