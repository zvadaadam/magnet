from typing import Callable, Optional, Union
from pathlib import Path
from magnet.data.corpus_preprocessor import CorpusProcessor
from magnet.models.colbert_model import ColBERT
from magnet.data.chunking import ChunkingStrategy, TextSplitterChunking

class NeuralSearch(object):

    model_name: Union[str, None] = None
    model: Union[ColBERT, None] = None
    corpus_processor: Optional[CorpusProcessor] = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        n_gpu: int = -1,
        verbose: int = 1,
    ):
        """
        Load a ColBERT model.
        """
        
        instance = cls()
        instance.model = ColBERT(
            pretrained_model_name_or_path, n_gpu, verbose=verbose
        )
        return instance

    @classmethod
    def from_index(
        cls, index_path: Union[str, Path], n_gpu: int = -1, verbose: int = 1
    ):
        """
        Load an Index and the associated ColBERT encoder from an existing document index.
        """
        
        instance = cls()
        index_path = Path(index_path)
        instance.model = ColBERT(
            index_path, n_gpu, verbose=verbose, load_from_index=True
        )

        return instance

    def index(
        self,
        collection: list[str],
        index_name: str = None,
        overwrite_index: bool = True,
        max_document_length: int = 256,
        split_documents: bool = True,
        chunking_strategy: Optional[ChunkingStrategy] = TextSplitterChunking(),
    ):
        """
        Build an index from a collection of documents. If allowed, run chunking on the documents before indexing.
        """
        if split_documents:
            collection = chunking_strategy.chunk(
                collection,
                chunk_size=max_document_length,
                overlap_size=max_document_length // 4,
            )
        
        overwrite = "reuse"
        if overwrite_index:
            overwrite = True
        return self.model.index(
            collection,
            index_name,
            max_document_length=max_document_length,
            overwrite=overwrite,
        )

    def add_to_index(
        self,
        new_documents: list[str],
        index_name: Optional[str] = None,
        split_documents: bool = True,
        chunking_strategy: Optional[ChunkingStrategy] = TextSplitterChunking(),
    ):
        """
        Add documents to an existing index.
        """
        if split_documents:
            new_documents = chunking_strategy.chunk(
                new_documents,
                chunk_size=self.model.config.doc_maxlen,
                overlap_size=self.model.config.doc_maxlen // 4,
            )

        self.model.add_to_index(
            new_documents,
            index_name=index_name,
        )

    def search(
        self,
        query: Union[str, list[str]],
        index_name: Optional["str"] = None,
        k: int = 10,
        force_fast: bool = False,
        zero_index_ranks: bool = False,
        **kwargs,
    ):
        """Query an index.

        Parameters:
            query (Union[str, list[str]]): The query or list of queries to search for.
            index_name (Optional[str]): Provide the name of an index to query. If None and by default, will query an already initialised one.
            k (int): The number of results to return for each query.
            force_fast (bool): Whether to force the use of a faster but less accurate search method.
            zero_index_ranks (bool): Whether to zero the index ranks of the results. By default, result rank 1 is the highest ranked result
        """
        return self.model.search(
            query=query,
            index_name=index_name,
            k=k,
            force_fast=force_fast,
            zero_index_ranks=zero_index_ranks,
            **kwargs,
        )