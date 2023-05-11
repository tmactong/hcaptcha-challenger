# -*- coding: utf-8 -*-
# Time       : 2022/4/30 17:00
# Author     : Bingjie Yan
# Github     : https://github.com/beiyuouo
# Description:
import os
import typing
import warnings

import cv2
import numpy as np
import yaml
from loguru import logger

from .kernel import ChallengeStyle
from .kernel import ModelHub

warnings.filterwarnings("ignore", category=UserWarning)


class ResNetFactory(ModelHub):
    def __init__(self, _onnx_prefix, _name, _dir_model: str):
        super().__init__(_onnx_prefix, _name, _dir_model)
        self.register_model()

    def classifier(
        self,
        img_stream,
        feature_filters: typing.Union[typing.Callable, typing.List[typing.Callable]] = None,
    ):
        img_arr = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(img_arr, flags=1)

        if img.shape[0] == ChallengeStyle.WATERMARK:
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        if feature_filters is not None:
            if not isinstance(feature_filters, list):
                feature_filters = [feature_filters]
            for tnt in feature_filters:
                if not tnt(img):
                    return False

        img = cv2.resize(img, (64, 64))
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (64, 64), (0, 0, 0), swapRB=True, crop=False)

        # 使用延迟反射机制确保分布式网络的两端一致性
        net = self.match_net()
        if net is None:
            _err_prompt = f"""
            The remote network does not exist or the local cache has expired.
            1. Check objects.yaml for typos | model={self.fn};
            2. Restart the program after deleting the local cache | dir={self.assets.dir_assets};
            """
            logger.warning(_err_prompt)
            self.assets.sync()
            return False
        net.setInput(blob)
        out = net.forward()
        if not np.argmax(out, axis=1)[0]:
            return True
        return False

    def solution(self, img_stream, **kwargs) -> bool:
        """Implementation process of solution"""
        return self.classifier(img_stream, feature_filters=None)


def new_tarnished(onnx_prefix: str, dir_model: str) -> ModelHub:
    """ResNet model factory, used to produce abstract model call interface."""
    return ResNetFactory(
        _onnx_prefix=onnx_prefix, _name=f"{onnx_prefix}(ResNet)_model", _dir_model=dir_model
    )
