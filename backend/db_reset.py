# In a Python script or shell
from backend.database.db import engine
from database import models

models.Base.metadata.drop_all(bind=engine)
models.Base.metadata.create_all(bind=engine)