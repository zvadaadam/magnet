from typing import List
from magnet import NeuralSearch
from magnet.llm.openai_llm import OpenAILLM
from magnet.llm.func_instruct import instruct

# main
if __name__ == "__main__":
    
    path_to_index = ".magnet/colbert/indexes/Jobs"
    neural_search = NeuralSearch.from_index(path_to_index)
    
    query = "wen did he die?"
    
    results = neural_search.search(query=query, k=3)
    
    @instruct
    def get_answear(query: str, results: List[str]) -> str:
        """Give me an asnwear to to this question based on the provided context chunks.
        QUERY: {query}
        CHUNKS: {results}
        
        Reply only based on the porvided context.
        
        REPLY:
        """
    print(results)
    
    answear = get_answear(query=query, results=results)
    
    print(answear)
