import ast
import datetime
import json
import os
import re
import sys
import time

from . import prompts
from .and_controller import chose_device, AndroidController, traverse_tree
from .model import parse_explore_rsp, parse_grid_rsp, chose_model
from .utils import print_with_color, draw_bbox_multi, draw_grid

def supervisor(model, gaol, per_image, cur_image) -> (bool, str):

    pass