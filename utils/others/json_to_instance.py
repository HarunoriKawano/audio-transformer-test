import pathlib
from typing import Type
from pydantic import BaseModel

def json_to_instance(path: str, structure: Type[BaseModel]):
    json_string = pathlib.Path(path).read_text()

    instance = structure.model_validate_json(json_string)
    return instance
