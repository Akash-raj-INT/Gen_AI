import arxiv

search = arxiv.Search(
    query="large language models",
    max_results=2,
    sort_by=arxiv.SortCriterion.Relevance,
)

client = arxiv.Client()
results = client.results(search)

# print results
for i, paper in enumerate(results):
    print(f"\nResult {i+1}")
    print("Title:", paper.title)
    print("Authors:", ", ".join(author.name for author in paper.authors))
    print("Summary:", paper.summary)