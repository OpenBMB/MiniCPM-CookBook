# RAG (Retrieval-Augmented Generation) Introduction
## Code Location and Usage
By simply modifying the parameters at the beginning of the project [4GB VRAM for RAG](https://github.com/OpenBMB/MiniCPM/blob/main/demo/minicpm/langchain_demo.py), you can run it.

## What is RAG?

RAG (Retrieval-Augmented Generation) is a method that combines retrieval techniques with generative models. It first uses a retrieval system to find relevant background knowledge related to the query (Query), then provides this background knowledge to the generative language model (LLM) to produce more accurate and reliable responses.

## Why Do We Need RAG?

Although large language models (LLMs) have already been able to provide effective answers in many fields, due to their nature as probability models based on historical data, they may produce inaccurate or even wrong answers for new problems not covered in the training data or questions requiring the latest information. RAG enhances the capabilities of LLMs by introducing external knowledge related to the query, thereby improving the quality of the answers.

## Basic Process of RAG

1. **Train Retrieval Model**: For example, using the `bge-base-zh` model to encode queries and knowledge data into vectors.
2. **Encode Queries and Data**: Use the retrieval model to encode both the query and the candidate texts.
3. **Calculate Similarity**: Compute the similarity between the query vector and all data vectors, and select the top K most relevant pieces of data.
4. **Re-Rank** (Optional): Further optimize the ranking of the found relevant texts.
5. **Construct Prompt**: Combine the query and relevant texts according to a specific prompt template and feed them into the large language model to generate the final answer.

## Example Code

### Parameter Settings

```python
parser.add_argument(
    "--cpm_model_path",
    type=str,
    default="openbmb/MiniCPM-1B-sft-bf16",
    help="Path to the MiniCPM model or huggingface id"
)
parser.add_argument(
    "--cpm_device", type=str, default="cuda:0", choices=["auto", "cuda:0"],
    help="Device for the MiniCPM model, default is cuda:0"
)
parser.add_argument("--backend", type=str, default="torch", choices=["torch", "vllm"],
     help="Use torch or vllm backend, default is torch"
)

# Embedding Model Parameters
parser.add_argument(
    "--encode_model", type=str, default="BAAI/bge-base-zh", 
    help="Embedding model for recall, default is BAAI/bge-base-zh, can input local path"
)
parser.add_argument(
    "--encode_model_device", type=str, default="cpu", choices=["cpu", "cuda:0"],
    help="Device for the embedding model used in recall, default is cpu"
)
parser.add_argument("--query_instruction", type=str, default="", help="Prefix added during recall")
parser.add_argument(
    "--file_path", type=str, default="/root/ld/pull_request/rag/红楼梦.pdf",
    help="Path to the text file to be searched, invalid when running gradio"
)

# Generation Parameters
parser.add_argument("--top_k", type=int, default=3)
parser.add_argument("--top_p", type=float, default=0.7)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--max_new_tokens", type=int, default=4096)
parser.add_argument("--repetition_penalty", type=float, default=1.02)

# Retriever Parameters
parser.add_argument("--embed_top_k", type=int, default=5, help="Number of most similar texts to recall")
parser.add_argument("--chunk_size", type=int, default=256, help="Length of text chunks when splitting")
parser.add_argument("--chunk_overlap", type=int, default=50, help="Overlap length when splitting text")
args = parser.parse_args()
```

### Inheriting LangChain's LLM

```python
class MiniCPM_LLM(LLM):
    tokenizer: Any = Field(default=None)
    model: Any = Field(default=None)

    def __init__(self, model_path: str):
        """
        Inherits the MiniCPM model from langchain
        
        Args:
        model_path (str): Path to the MiniCPM model to load.

        Returns:
        self.model: Loaded MiniCPM model.
        self.tokenizer: Tokenizer for the loaded MiniCPM model.
        """
        super().__init__()
        if args.backend == "vllm":
            from vllm import LLM

            self.model = LLM(
                model=model_path, trust_remote_code=True, enforce_eager=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=torch.float16
            ).to(args.cpm_device)
            self.model = self.model.eval()

    def _call(self, prompt, stop: Optional[List[str]] = None):
        """
        Call of langchain.llm
        
        Args:
        prompt (str): Input prompt text

        Returns:
        responds (str): Text generated by the model in response to the prompt
        """
        if args.backend == "torch":
            inputs = self.tokenizer("<用户>{}".format(prompt), return_tensors="pt")
            inputs = inputs.to(args.cpm_device)
            # Generate
            generate_ids = self.model.generate(
                inputs.input_ids,
                max_length=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            responds = self.tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            # responds, history = self.model.chat(self.tokenizer, prompt, temperature=args.temperature, \
            # top_p=args.top_p, repetition_penalty=1.02)
        else:
            from vllm import SamplingParams

            params_dict = {
                "n": 1,
                "best_of": 1,
                "presence_penalty": args.repetition_penalty,
                "frequency_penalty": 0.0,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "use_beam_search": False,
                "length_penalty": 1,
                "early_stopping": False,
                "stop": None,
                "stop_token_ids": None,
                "ignore_eos": False,
                "max_tokens": args.max_new_tokens,
                "logprobs": None,
                "prompt_logprobs": None,
                "skip_special_tokens": True,
            }
            sampling_params = SamplingParams(**params_dict)
            prompt = "<用户>{}<AI>".format(prompt)
            responds = self.model.generate(prompt, sampling_params)
            responds = responds[0].outputs[0].text

        return responds

    @property
    def _llm_type(self) -> str:
        return "MiniCPM_LLM"
```

### Loading Embedding Model

```python
embedding_models = HuggingFaceBgeEmbeddings(
        model_name=args.encode_model,
        model_kwargs={"device": args.encode_model_device},  # or 'cuda' if you have a GPU
        encode_kwargs={
            "normalize_embeddings": True,  # Whether to normalize embeddings
            "show_progress_bar": True,  # Whether to show progress bar
            "convert_to_numpy": True,  # Whether to convert output to numpy array
            "batch_size": 8,  # Batch size
        },
        query_instruction=args.query_instruction,
    )
```

### Document Embedding

```python
def embed_documents(documents, embedding_models):
    """
    Split and embed documents
    
    Args:
    documents (list): List of read texts
    embedding_models: Embedding model

    Returns:
    vectorstore: Vector database
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    ) # Initialize a text splitter
    texts = text_splitter.split_documents(documents) # Split files into text, obtain a list of texts
    vectorstore = Chroma.from_documents(texts, embedding_models) # Convert texts to vectors
    return vectorstore
```

### Find Most Similar Texts

```python
docs = vectorstore.similarity_search(query, k=args.embed_top_k)
```

### Define Prompt Template

```python
def create_prompt_template():
    """
    Create a custom prompt template
    
    Returns:
    PROMPT: Custom prompt template
    """
    custom_prompt_template = """Please use the following content fragments to form the final response to the question. Do not guess about information not mentioned in the content,
    strictly answer according to the content, do not fabricate answers. If the answer cannot be found in the content, please respond “Not mentioned in the fragment, unable to answer”, do not fabricate answers.
    Context:
    {context}

    Question: {question}
    FINAL ANSWER:"""
    PROMPT = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )
    return PROMPT
```

### Define LangChain Process
1. First, combine the input query and highly relevant data docs into the prompt template to form combined data.
2. Then, call the llm to perform inference on the combined data. The llm inherits from langchain.llm and is the minicpm.

```python
def create_rag_chain(llm, prompt):
    qa = prompt | llm
    return qa
```

### Invoke RAG Chain

```python
final_result = rag_chain.invoke({"context": all_links, "question": query})
```

### Gradio Frontend

![alt text](../../asset/langchain.png)
### Minimum VRAM Usage
This is the highest VRAM consumption during the example usage process, totaling 2.2G VRAM.

![alt text](../../asset/langchain1.png)

# Using 4GB VRAM

To run the RAG system in an environment with limited VRAM, you can take the following steps:

1. **Quantize the Model**  
   According to the quantization guide for MiniCPM, quantize the `openbmb/MiniCPM-1B-sft-bf16` model to `int4`. It is recommended to use the `AWQ` (AutoQuantization with Weight-only) method for quantization.

2. **Modify Parameters**  
   Modify the following parameters in `MiniCPM/demo/langchain_demo.py`:
   ```python
   parser.add_argument(
       "--cpm_model_path",
       type=str,
       default="your/int4_cpm/save/path",
       help="Path to the MiniCPM model or huggingface id"
   )
   parser.add_argument(
       "--encode_model_device", type=str, default="cpu", choices=["cpu", "cuda:0"],
       help="Device for the embedding model used in recall, default is cpu"
   )
   ```

3. **Run the Script**  
   Execute `MiniCPM/demo/langchain_demo.py`.

---

# Speed Priority Method

If you prioritize running speed over VRAM usage, consider the following configuration:

1. **Use Non-Quantized Model**  
   Do not perform quantization processing on the MiniCPM model, directly use the original model.

   ```python
   parser.add_argument(
       "--cpm_model_path",
       type=str,
       default="your/cpm/save/path",  # Quantized models generally have slower speeds with the Torch backend
       help="Path to the MiniCPM model or huggingface id"
   )
   ```

2. **Use vllm as Backend**  
   The `vllm` backend typically offers better performance.

   ```python
   parser.add_argument("--backend", type=str, default="vllm", choices=["torch", "vllm"],
                        help="Use torch or vllm backend, default is torch"
   )
   ```

3. **Move Embedding Model to GPU**  
   Deploying the embedding model used for recall to the GPU can improve performance.

   ```python
   parser.add_argument(
       "--encode_model_device", type=str, default="cuda:0", choices=["cpu", "cuda:0"],
       help="Device for the embedding model used in recall, default is cpu"
   )
   ```

4. **Run the Script**  
   Execute `MiniCPM/demo/langchain_demo.py`.
