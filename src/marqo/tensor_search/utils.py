import typing
from typing import List
import functools
import json
import torch
from marqo import errors
from marqo.tensor_search import enums
from typing import List, Optional, Union, Callable, Iterable, Sequence, Dict
import copy
import datetime
import validators
import mimetypes
from PIL.Image import Image as ImageType
from numpy import ndarray
from urllib.parse import urlparse
from marqo.tensor_search.enums import MediaType, FileType
import os
from marqo.tensor_search.enums import MediaType, FileType
from typing import Tuple
import requests, re
import wget


def dicts_to_jsonl(dicts: List[dict]) -> str:
    """Turns a list of dicts into a JSONL string"""
    return functools.reduce(
        lambda x, y: "{}\n{}".format(x, json.dumps(y)),
        dicts, ""
    ) + "\n"


def generate_vector_name(field_name: str) -> str:
    """Generates the name of the vector based on the field name"""
    return F"{enums.TensorField.vector_prefix}{field_name}"


def truncate_dict_vectors(doc: Union[dict, List], new_length: int = 5) -> Union[List, Dict]:
    """Creates a readable version of a dict by truncating identified vectors
    Looks for field names that contains the keyword "vector"
    """
    copied = copy.deepcopy(doc)

    if isinstance(doc, list):
        return [truncate_dict_vectors(d, new_length=new_length)
                if isinstance(d, list) or isinstance(d, dict)
                else copy.deepcopy(d)
                for d in doc]

    for k, v in list(copied.items()):
        if "vector" in k.lower() and isinstance(v, Sequence):
            copied[k] = v[:new_length]
        elif isinstance(v, dict) or isinstance(v, list):
            copied[k] = truncate_dict_vectors(v, new_length=new_length)

    return copied


def create_duration_string(timedelta):
    """Creates a duration string suitable that can be returned in the AP

    Args:
        timedelta (datetime.timedelta): time delta, or duration.

    Returns:

    """
    return f"PT{timedelta.total_seconds()}S"


def format_timestamp(timestamp: datetime.datetime):
    """Creates a timestring string suitable for return in the API

    Assumes timestamp is UTC offset 0
    """
    return f"{timestamp.isoformat()}Z"


def construct_authorized_url(url_base: str, username: str, password: str) -> str:
    """
    Args:
        url_base:
        username:
        password:

    Returns:

    """
    http_sep = "://"
    if http_sep not in url_base:
        raise errors.MarqoError(f"Could not parse url: {url_base}")
    url_split = url_base.split(http_sep)
    if len(url_split) != 2:
        raise errors.MarqoError(f"Could not parse url: {url_base}")
    http_part, domain_part = url_split
    return f"{http_part}{http_sep}{username}:{password}@{domain_part}"


def contextualise_filter(filter_string: str, simple_properties: typing.Iterable) -> str:
    """adds the chunk prefix to the start of properties found in simple string

    This allows for filtering within chunks.

    Args:
        filter_string:
        simple_properties: simple properties of an index (such as text or floats
            and bools)

    Returns:
        a string where the properties are referenced as children of a chunk.
    """
    contextualised_filter = filter_string
    for field in simple_properties:
        contextualised_filter = contextualised_filter.replace(f'{field}:', f'{enums.TensorField.chunks}.{field}:')
    return contextualised_filter


def check_device_is_available(device: str) -> bool:
    """Checks if a device is available on the machine

    Args:
        device: assumes device is a valid device strings (e.g: 'cpu' or
            'cuda:1')

    Returns:
        True, IFF it is available

    Raises:
        MarqoError if device is determined to be invalid
    """
    lowered = device.lower()
    if lowered == "cpu":
        return True

    split = lowered.split(":")
    if split[0] != "cuda":
        raise errors.MarqoError(f"Invalid device prefix! {device}. Valid prefixes: 'cpu' and 'cuda'")

    if not torch.cuda.is_available():
        return False

    if len(split) < 2:
        return True

    if int(split[1]) < 0:
        raise errors.MarqoError(f"Invalid cuda device number! {device}. It must not be negative")

    if int(split[1]) < torch.cuda.device_count():
        return True
    else:
        return False





def convert_to_MediaType(type_str: str) -> MediaType:
    #This can actually be a dictionary
    if type_str == "text":
        return MediaType.text
    elif type_str == "image":
        return MediaType.image
    elif type_str == "video":
        return MediaType.video
    else:
        raise TypeError(f"Unspported type for {type_str}")

def convert_to_FileType(type_str = "default") -> FileType:
    #This can actually be a dictionary :(
    if type_str == "default":
        return FileType.default
    elif type_str == "url":
        return FileType.url
    elif type_str == "www.youtube.com":
        return FileType.youtube
    elif type_str == "www.tiktok.com":
        return FileType.tiktok
    elif type_str == "local_path":
        return FileType.local
    elif type_str == "straight_text":
        return FileType.straight_text
    elif type_str == "ndarray":
        return FileType.ndarray
    elif type_str == "PILImage":
        return FileType.PILImage
    elif type_str == "ListOfPILImage":
        return FileType.ListOfPILImage
    else:
        raise TypeError(f"Unsupported filetype for {type_str}")



def get_filename_from_cd(cd):
    """
    Get filename from content-disposition
    """
    if not cd:
        return None
    fname = re.findall('filename=(.+)', cd)
    if len(fname) == 0:
        return None
    return fname[0]

def content_routering(field_content, infer_if_media = True):
    if isinstance(field_content, str):
        if infer_if_media:
            if validators.url(field_content):
                try:
                    downloaded_file = wget.download(field_content)
                    file_type, encoding = mimetypes.guess_type(downloaded_file)
                except:
                    r = requests.get(field_content, stream= True, allow_redirects=True).raw
                    downloaded_file = get_filename_from_cd(r.headers.get('content-disposition'))
                    open(downloaded_file, 'wb').write(r.content)
                    if downloaded_file and r:
                        file_type, encoding = mimetypes.guess_type(downloaded_file)
                    else:
                        raise TypeError(f"The url {field_content} is not downloadable.")


                if file_type:
                    main_type = file_type.split("/")[0]
                    return downloaded_file, convert_to_MediaType(main_type), convert_to_FileType()
                else:
                    raise TypeError(f"We can't recognize the Type of input url {field_content}")
            elif os.path.isfile(field_content):
                file_type, encoding = mimetypes.guess_type(field_content)
                main_type = file_type.split("/")[0]
                return field_content, convert_to_MediaType(main_type), FileType.local,
            else:
                return field_content, MediaType.text, FileType.straight_text

        else:
            return field_content, MediaType.text, FileType.straight_text

    elif isinstance(field_content, ImageType):
        return field_content, MediaType.image,FileType.PILImage
    else:
        raise TypeError(f"The input type {type(field_content)} is supported.")
