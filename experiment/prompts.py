"""Instructions from the paper's appendix (Tables: Vanilla, Chain-of-Thought, query generation, knowledge extraction).

Answer prompts are laid out as in the paper:

    <instruction>

    Question: <image>{question}
    Document:
    [1]: ...
    ...

    Answer:            (Vanilla)  /  Let's think step-by-step:  (CoT)
"""

VANILLA_INSTRUCTION = """Based on the documents, provide a helpful answer in paragraph form. Your answer must be supported by the content in the citations.
You should cite the passage number (indices) in the format of [1][2][3] at the end of each sentence.
Do not include sentences that are not supported by the documents."""

COT_INSTRUCTION = """Your task is to answer the question using the provided documents and cite your answer with their passage numbers.

First, before answering, show your reasoning process using the <thinking> and </thinking> tags. Follow the thinking process format below:
    1. Identify the relevant documents related to the question: "The only relevant documents to the question are documents [<relevant_doc>], [<relevant_doc>].."
    2. Analyze the relevant information from the identified documents: "From document [<relevant_doc>], we know that '<relevant_info>', '<relevant_info>'.."

Second, answer the question, explicitly incorporating copied spans into your answer. Your answer must be supported by the content in the citations. You should cite the passage number (indices) in the format of [1][2][3] at the end of the sentence. Do not refer to the documents explicitly using phrases like "Document [5] states" or "According to Document [3]". Here is an output example:
--
<thinking>
1. Identify the relevant documents related to the question: "The only relevant documents to the question are documents [1], [2], [3], [5]."
2. Analyze the relevant information from the identified documents:
    - From document [1], we know that "traditional, low-sugar and no-sugar fruit preserving methods including bottling, jams and jellies, fruit pie fillings, dehydrating and cooking" can be used.
    - From document [2], we learn about "fresh fruit desserts, jams, jellies, preserves, canned peaches, pears, cherries and apricots, and fresh fruit salads," and tips for "canning, freezing fruit and keeping fruit fresh."
    - From document [3], grilling is suggested as a creative method: "Grilling stone fruits only serves to heighten their natural sweetness."
    - From document [5], apricots can be "dried, cooked into pastry, and eaten as jam" or "distilled into brandy and liqueur."
</thinking>
--"""

QUERY_GENERATION_INSTRUCTION = """<Instruction>
1. Based on the given image and question, generate {N} search queries.
2. Formulate queries to retrieve documents that provide information to generate the answer.
3. List the generated search queries separated by commas. For example: "query 1", "query 2", ..."""

KNOWLEDGE_EXTRACTION_INSTRUCTION = """Your task is to extract factual information from the provided document. Include only details that can be confidently determined, excluding imaginary, speculative, or aesthetic content. Present the information clearly and concisely in paragraph form. Do not explicitly refer to the document itself or use introductory phrases such as "the document states," "it mentions," or "according to the document." Instead, directly state the factual information.

Document:
"""

DOC_PREFIX = "\n[{}]: "
