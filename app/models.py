from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime

from .database import Base


class PlateDetection(Base):

    __tablename__ = "plate_detections"

    id = Column(Integer, primary_key=True, index=True)

    plate_number = Column(
        String,
        nullable=True
    )

    detection_confidence = Column(
        Float,
        nullable=False
    )

    x1 = Column(Integer)
    y1 = Column(Integer)
    x2 = Column(Integer)
    y2 = Column(Integer)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )