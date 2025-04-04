import os
import gradio as gr
import shutil
import jieba.analyse as aly
from collections import Counter
from database import MyDataBase
import pandas as pd
import random
from database import load_database
from local_llm_model import get_ans
from knowledge_extract import generate_qa


def get_type_name(files):
    content = []
    for file in files:
        try:
            with open(file.name, encoding="utf-8") as f:
                data = f.readlines(1)
                content.extend(aly.tfidf(data[0]))
        except:
            continue
    count = Counter(content)
    kw = count.most_common(2)

    return "".join([i[0] for i in kw])


def upload(files):
    global database_list, database_namelist, input_qa

    check_txt = False

    for file in files:
        if check_txt:
            break
        if file.name.endswith(".txt"):
            check_txt = True
    else:
        if check_txt == False:
            raise Exception("请上传包含txt文档的文件夹")

    type_name = get_type_name(files)
    save_path = os.path.join("data/database_dir", type_name)

    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, "txt"))
    for file in files:
        if file.name.endswith(".txt"):
            shutil.copy(file.name, os.path.join(save_path, "txt"))

    database = MyDataBase(save_path, type_name)
    database_list.append(database)
    database_namelist.append(type_name)
    knowledge_names.choices.append((type_name, type_name))
    input_qa.choices.append((type_name, type_name))
    context = pd.DataFrame(database.document.contents, columns=["context"])
    return type_name, context


def database_change(name):
    global database_list, database_namelist
    context = pd.DataFrame(database_list[database_namelist.index(name)].document.contents, columns=["question"])
    return context


def generate_rag_result(knowledge_name, question):
    # Call your data generation function here
    database = database_list[database_namelist.index(knowledge_name)]
    search_result = database.search(question, 3)
    abstract = "\n".join(search_result["answer"])
    prompt = f"请根据已知内容简洁明了的回复用户的问题，已知内容如下：```{abstract}```,用户的问题是：{question}，如何已知内容无法回答用户的问题，请直接回复：不知道，无需输出其他内容"
    result = get_ans(prompt)
    return result, search_result


def process_qa_data(input_qa):
    global database_list, database_namelist
    select_database = database_list[database_namelist.index(input_qa)]
    contexts = select_database.document.contents
    qa_df = generate_qa(input_qa,contexts)
    select_database.create_emb_database(qa_df)
    return qa_df


if __name__ == "__main__":
    database_list, database_namelist = load_database()
    print("data................")

    with gr.Blocks() as demo:
        with gr.Tab("知识库管理"):
            knowledge_names = gr.Dropdown(choices=database_namelist, label="知识库选择", value=database_namelist[0])
            context = gr.DataFrame(pd.DataFrame(database_list[0].document.contents, columns=["context"]), height=800)
            input3 = gr.UploadButton(label="上传知识库", file_count="directory")
            input3.upload(upload, input3, [knowledge_names, context])
            knowledge_names.change(database_change, knowledge_names, context)
        with gr.Tab("生成问答数据"):
            input_qa = gr.Dropdown(choices=database_namelist, label="知识库选择", value=database_namelist[0])
            output_data = gr.DataFrame(database_list[0].document.qa_df, height=400)
            generate_data_button = gr.Button("生成问答QA数据")
            generate_data_button.click(process_qa_data, [input_qa], [output_data])

        with gr.Tab("RAG知识库问答"):
            input_qa = gr.Dropdown(choices=database_namelist, label="知识库问答", value=database_namelist[0])
            text_input = gr.Textbox(label="请输出问题")
            text_rag = gr.DataFrame(pd.DataFrame(), height=400,label="RAG 结果")
            text_output = gr.Textbox(label="大模型生成答案")
            qa_button = gr.Button("生成答案")
            qa_button.click(generate_rag_result, [input_qa, text_input], [text_output,text_rag])

    demo.launch(server_name="0.0.0.0", server_port=9999, show_api=False, auth=("username", "password"))
