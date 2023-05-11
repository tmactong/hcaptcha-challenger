# -*- coding: utf-8 -*-
# Time       : 2022/2/15 17:43
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import os
import shutil
import time
import typing
from urllib.parse import urlparse

from ._scaffold import Config, get_challenge_ctx, init_log
from ._solutions.kernel import ModelHub
from ._solutions.kernel import PluggableObjects
from .core import HolyChallenger

__all__ = ["HolyChallenger", "new_challenger", "get_challenge_ctx"]
__version__ = "0.4.3.3"

logger = init_log(
    error=os.path.join("datas", "logs", "error.log"),
    runtime=os.path.join("datas", "logs", "runtime.log"),
)


def new_challenger(
    dir_workspace: str = "_challenge",
    lang: typing.Optional[str] = "en",
    screenshot: typing.Optional[bool] = False,
    debug: typing.Optional[bool] = False,
    slowdown: typing.Optional[bool] = True,
) -> HolyChallenger:
    """

    :param slowdown:
    :param dir_workspace:
    :param lang:
    :param screenshot:
    :param debug:
    :return:
    """
    if not isinstance(dir_workspace, str) or not os.path.isdir(dir_workspace):
        dir_workspace = os.path.join("datas", "temp_cache", "_challenge")
        os.makedirs(dir_workspace, exist_ok=True)

    return HolyChallenger(
        dir_workspace=dir_workspace,
        dir_model=None,
        lang=lang,
        screenshot=screenshot,
        debug=debug,
        slowdown=slowdown,
    )


def set_reverse_proxy(https_cdn: str):
    parser = urlparse(https_cdn)
    if parser.netloc and parser.scheme.startswith("https"):
        ModelHub.CDN_PREFIX = https_cdn
