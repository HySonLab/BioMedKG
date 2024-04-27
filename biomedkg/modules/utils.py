import re
from typing import Any

def clean_name(input_string) -> str:
    pattern = re.compile('[a-zA-Z]+')
    characters = ''.join(pattern.findall(input_string))
    return characters


def parameters_count(model:Any) -> int: 
    total_param = 0

    for param in model.parameters():
        total_param += param.numel()

    return total_param


def format_time(duration):
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))