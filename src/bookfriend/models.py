from database import Base
from sqlalchemy import Column, String

class Book(Base):
    __tablename__ = "books"
    id = Column(String, primary_key=True, index=True)
    index_path = Column(String)
    # ... other columns ...