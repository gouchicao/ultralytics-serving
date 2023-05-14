from pydantic import BaseModel


class Box(BaseModel):
    x: int
    y: int
    w: int
    h: int

    def center(self):
        return ( self.x+self.w//2, self.y+self.h//2 )
        
    def is_same_row(self, box):
        center_y = self.center()[1]
        return (center_y > box.y) and (center_y < box.y+box.h)

    def is_same_col(self, box):
        center_x = self.center()[0]
        return (center_x > box.x) and (center_x < box.x+box.w)

class DetectObject(BaseModel):
    score: float
    label: str
    box: Box

class ImageMetadata(BaseModel):
    width: int
    height: int
    type: str
    size: int

class PredictResult(BaseModel):
    objects: list[DetectObject]
    metadata: ImageMetadata
    time_ms: int

class DetectObjectWithRowColumn(DetectObject):
    row: int
    column: int
