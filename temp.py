from main import get_answer_from_documents, all_documents, Document

all_documents.append(
    Document(
        page_content="Dirty restaurant 2024-01-20 1 I was shocked at the state of this place. Tables were sticky, floors were dirty, and I could see into the kitchen which didn't look any better. Couldn't enjoy my pizza",
        metadata={},
        id="1",
    )
)
print(get_answer_from_documents("Is the restaurant clean?"))
