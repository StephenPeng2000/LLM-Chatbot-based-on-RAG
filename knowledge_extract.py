import pandas as pd
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from tqdm import tqdm
from local_llm_model import get_ans


def generate_qa(name,contents):
    result = []
    qa_dir = f"data/database_dir/{name}/qa"

    for content in tqdm(contents):
        # 文本太短则认为不具有信息
        if len(content) < 200:
            continue
        prompt = f"""假设你是一个新闻记者，你需要根据主题词和文章内容中帮我提取有价值和意义的问答对，有助于我进行采访。
                主题词：{name}
                文章内容：
                {content}
                请注意，你提取的问答内容必须和主题词高度符合，无需输出其他内容，提取的每个问答返回一个python字典的格式，样例如下：
                {{"问":"xxx","答":"xxx"}}
                {{"问":"xxx","答":"xxx"}}
                提取的问答内容为："""

        answers = get_ans(prompt)
        answers = answers.split("\n")

        for answer in answers:
            if len(answer) < 10:
                continue
            try:
                answer = eval(answer)
            except:
                continue
            if "问" not in answer or "答" not in answer:
                continue
            result.append({
                "question": answer["问"],
                "answer": answer["答"]
            })
    result_df = pd.DataFrame(result)
    result_df.to_excel(qa_dir + "/qa_result.xlsx", index=False)
    return result_df


if __name__ == "__main__":
    name = "小米汽车"
    dir = f"data/database_dir/{name}/txt"
    qa_dir = f"data/database_dir/{name}/qa"
    # 加载文本
    loader = DirectoryLoader(dir)
    documents = loader.load()
    # 对文本进行切割
    text_spliter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)  # 400   0:0-200  1:150-350 2:300-400
    split_docs = text_spliter.split_documents(documents)
    contents = [i.page_content for i in split_docs]
    generate_qa(contents)

