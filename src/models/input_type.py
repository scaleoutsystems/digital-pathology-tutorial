from pydantic import BaseModel
from fastapi import File, UploadFile

# class PredType(BaseModel):
    # Optional, but if presents specifies the input 'inp'
    # to predict.
    # Default pred: str. Can be accessed in predict as inp.pred.
    # pred: str
    # a: str
    # b: int
PredType = UploadFile
defv = File(...)