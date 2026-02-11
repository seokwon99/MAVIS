closed_book = """Your role is to revise a given question based on an accompanying image. The revised question must require the image to be understood. Remove any information from the question that is already present in the image.

Important: Your revised question must be written in the following format: Question: <revised_question>

Question: {question}
"""

rag_prompt = """Based on the documents, provide a helpful answer to the query in paragraph form. Your answer must be supported by the content in the citations. \\
You should cite the passage number (indices) in the format of [1][2][3] at the end of each sentence. \\
Do not include sentences that are not supported by the documents.

Question: {question}
Documents:
"""

cot_prompt = """Your task is to answer the question using the provided documents and cite your answer with their passage numbers.

First, before answering, show your reasoning process using the <thinking> and </thinking> tags. Follow the thinking process format below:
    1. Identify the relevant documents related to the question: "The only relevant documents to the question are documents [<relevant_doc>], [<relevant_doc>].."
    2. Extract the information from each document that is helpful in answering the question. If there is no relevant information, respond with 'None': "From document [<relevant_doc>], we know that '<relevant_info>', '<relevant_info>'.."

Second, answer the question, explicitly incorporating copied spans into your answer. Your answer must be supported by the content in the citations. You should cite the passage number (indices) in the format of [1][2][3] at the end of the sentence. Do not refer to the documents explicitly using phrases like "Document [5] states" or "According to Document [3]". Here is an output example:

--
<thinking>
1. Identify the relevant documents related to the question: "The only relevant documents to the question are documents [1], [2], [3], [5]."
2. Extract the information from each document that is helpful in answering the question: 
   - From document [1], we know that "traditional, low-sugar and no-sugar fruit preserving methods including bottling, jams and jellies, fruit pie fillings, dehydrating and cooking" can be used.
   - From document [2], we learn about "fresh fruit desserts, jams, jellies, preserves, canned peaches, pears, cherries and apricots, and fresh fruit salads," and tips for "canning, freezing fruit and keeping fruit fresh."
   - From document [3], grilling is suggested as a creative method: "Grilling stone fruits only serves to heighten their natural sweetness."
   - From document [5], apricots can be "dried, cooked into pastry, and eaten as jam" or "distilled into brandy and liqueur."
</thinking>

Thus, the final answer is: Some creative ways to use a large harvest of homegrown fruits like apricots, peaches, and cherries include preserving them through bottling, making jams and jellies, creating fruit pie fillings, dehydrating, and cooking [1]. You can also enjoy them in fresh fruit desserts, salads, or as canned goods, as well as freezing them to keep fresh [2]. Grilling these fruits enhances their sweetness and provides a unique way to enjoy them [3]. Additionally, apricots can be dried, used in pastries, or even distilled into brandy and liqueur [5].
--

Now, your turn:
Question: {question}  
Documents:
"""

aggregation_instruction = """Question: {question}
Here are some answers:
- Answer 1: {A_text}
- Answer 2: {A_image}

Your task is to produce a fluent answer to the question by integrating the given answers with sentence-level citations. Do not include any irrelevant information from either answer. Preserve all citations exactly as they appear in the original answers.

Combined Answer:"""

PRE = "\n[{}]: "

## Deprecated
routing_prompt = """Classify the following query into one of six categories: [Text, Image], based on whether it requires retrieval-augmented generation (RAG) and the most appropriate modality. Consider:
- Text: The query requires retrieving factual descriptions, straightforward explanations, or concise summaries from a single source.
- Image: The query focuses on visual aspects like appearances, structures, or spatial relationships.

Examples:
- "What is the birth date of Alan Turing?" → Text
- "Describe the appearance of a blue whale." → Image
- "Who played a key role in the development of the iPhone?" → Text
- "Describe the structure of the Eiffel Tower." → Image

Classify the following query: {question}
Provide only the category.
"""