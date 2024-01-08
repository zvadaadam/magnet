import csv
from typing import List
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from magnet.data.chunking import ChunkingStrategy
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextSplitterChunking(ChunkingStrategy):

    @abstractmethod
    def chunk(self, data: str, chunk_size: int=100, chunk_overlap: int=20) -> List[str]:
        
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        
        chunks = [chunk.text for chunk in chunker(data)]
        
        return chunks