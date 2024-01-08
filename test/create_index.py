from magnet import NeuralSearch
import requests


def get_wikipedia_page(title: str):

    URL = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }

    response = requests.get(URL, params=params)
    data = response.json()

    page = next(iter(data['query']['pages'].values()))
    return page['extract'] if 'extract' in page else None

# main
if __name__ == "__main__":
    
    page_content = get_wikipedia_page("Steve_Jobs")
    print(len(page_content))
    #print(page_content)
    
    neural_search = NeuralSearch.from_pretrained("colbert-ir/colbertv2.0")
    
    neural_search.index(collection=[page_content], index_name="Jobs", max_document_length=180, split_documents=True)
    
    results = neural_search.search(query="Which companies did Steve Jobs found?", k=3)
    
    print(results)
    
    # neural_search.index(collection=[page_content], index_name="Jeff Bezos", max_document_length=180, split_documents=True)

