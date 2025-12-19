from adapter.utils.savable import Savable
from pydantic import BaseModel


class Topic(BaseModel):
    title: str
    description: str


class Topics(Savable):
    topics: list[Topic]


class TopicEntity(Savable):
    file_path: str
    topic: Topic


class TopicEntities(Savable):
    topics: list[TopicEntity]
