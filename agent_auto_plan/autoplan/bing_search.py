from playwright.sync_api import sync_playwright
#from searcher import *
from typing import List, Dict, Tuple, Optional

import json

import requests
from bs4 import BeautifulSoup
import openai
openai.api_key = ''  # Replace with your OpenAI API key
openai.api_base = "https://api.chatanywhere.com.cn/v1"
from langchain.document_loaders import UnstructuredURLLoader
from langchain.docstore.document import Document
from unstructured.cleaners.core import remove_punctuation,clean,clean_extra_whitespace
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain

from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain

import json
from typing import List, Dict

class SearchResult:
    def __init__(self, title, url, snip) -> None:
        self.title = title
        self.url = url
        self.snip = snip

    def dump(self):
        return {
            "title": self.title,
            "url": self.url,
            "snip": self.snip
        }

    def __str__(self) -> str:
        return json.dumps(self.dump())
    
class SearcherInterface:
    def search(self, query) -> List[SearchResult]:
        raise NotImplementedError()


def generate_document(url):
    "Given an URL, return a langchain Document to futher processing"
    loader = UnstructuredURLLoader(urls=[url],
    mode="elements",
    post_processors=[clean,remove_punctuation,clean_extra_whitespace])
    elements = loader.load()
    selected_elements = [e for e in elements if e.metadata['category']=="NarrativeText"]
    full_clean = " ".join([e.page_content for e in selected_elements])
    return Document(page_content=full_clean, metadata={"source":url})

def summarize_document(url, model_name):
    "Given an URL return the summary from OpenAI model"
    llm = OpenAI(model_name='ada',temperature=0,openai_api_key=openai.api_key)
    chain = load_summarize_chain(llm, chain_type="stuff")
    tmp_doc = generate_document(url)
    summary = chain.run([tmp_doc])
    return clean_extra_whitespace(summary)

# print(summarize_document(url, ""))
####################################################################################################


def fetch_webpage_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

def summarize_text(text,question,model,tokenizer):
    if model==None:
        messages = [{'role': 'user','content': f"base the following text:\n\n{text}\n\n use no more than 100 chinese words to answer the question:{question}"},]
        response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo-16k',
                messages=messages,
                stream=True,
            )
        completion = {'role': '', 'content': ''}
        for event in response:
            if event['choices'][0]['finish_reason'] == 'stop':
                #print(f'收到的完成数据: {completion}')
                break
            for delta_k, delta_v in event['choices'][0]['delta'].items():
                #print(f'流响应数据: {delta_k} = {delta_v}')
                completion[delta_k] += delta_v
        messages.append(completion)  # 直接在传入参数 messages 中追加消息
        return messages[1]['content']
    else:
        task_split_prompt = f"根据以下文本：:\n\n{text}\n\n 用不超过100个中文字符回答以下问题:{question}"
        response=model.chat(tokenizer,task_split_prompt,history=[])[0]
        return response

class Searcher(SearcherInterface):
    def __init__(self) -> None:
        pass

    def _parse(self, result) -> List[SearchResult]:
        if not result:
            return None
        ret = []
        for item in result:
            ret.append(SearchResult(item['title'], item['url'], None))
        return ret

    def search(self, query) -> List[SearchResult]:
        return self._parse(query_bing(query))


def get_bing_search_raw_page(question: str):
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context()
        page = context.new_page()
        try:
            page.goto(f"https://www.bing.com/search?q={question}")
        except:
            page.goto(f"https://www.bing.com")
            page.fill('input[name="q"]', question)
            page.press('input[name="q"]', 'Enter')
        try:
            page.wait_for_load_state('networkidle', timeout=6000)
        except:
            pass
        # page.wait_for_load_state('networkidle')
        search_results = page.query_selector_all('.b_algo h2')
        for result in search_results:
            title = result.inner_text()
            a_tag = result.query_selector('a')
            if not a_tag: continue
            url = a_tag.get_attribute('href')
            if not url: continue
            # print(title, url)
            results.append({
                'title': title,
                'url': url
            })
        browser.close()
    return results

def query_bing(question, max_tries=3,model=None,tokenizer=None):
    cnt = 0
    while cnt < max_tries:
        cnt += 1
        results = get_bing_search_raw_page(question)
        # print(results)
        for ret in results:
            try:
                url = ret["url"]
                webpage_text = fetch_webpage_content(url)
                webpage_text = webpage_text.rstrip("\n")
                webpage_text = webpage_text.strip()[:4097]
                summary = summarize_text(webpage_text[:1000],question,model=model,tokenizer=tokenizer)
                return summary
            except Exception as e:
                print(e)
                continue
        return '没有找到相关结果'


if __name__ == '__main__':
    print(query_bing('如何学好nlp？'))
    # with open('crawl.json', 'w', encoding='utf-8') as f:
    #     json.dump(query_bing('今天天气如何？'), f, ensure_ascii=False, indent=4)
        
    # exit(0)
