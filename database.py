from pypinyin import lazy_pinyin
import pickle
import os
import faiss
from emb_model import MyEmbModel
from load_documents import MyDocument


def cover_name(name):
    return "_".join(lazy_pinyin(name))


class MyEmbDatabase():
    def __init__(self, emb_dir, qa_df, name):
        global client
        self.name = cover_name(name)
        self.emb_model = MyEmbModel(emb_dir)
        if not os.path.exists(os.path.join(".cache", f"{name}_faiss_index.pkl")):
            index = faiss.IndexFlatL2(self.emb_model.model.get_sentence_embedding_dimension())
            embs = self.emb_model.to_emb(qa_df["question"])
            index.add(embs)

            with open(os.path.join(".cache", f"{name}_faiss_index.pkl"), "wb") as f:
                pickle.dump(index, f)
            with open(os.path.join(".cache", f"{name}_faiss_index_df.pkl"), "wb") as f:
                pickle.dump(qa_df, f)
        else:
            with open(os.path.join(".cache", f"{name}_faiss_index.pkl"), "rb") as f:
                index = pickle.load(f)
            with open(os.path.join(".cache", f"{name}_faiss_index_df.pkl"), "rb") as f:
                qa_df = pickle.load(f)
        self.index = index
        self.qa_df = qa_df

    def search(self, content, topn=3):
        global client

        if isinstance(content, str):
            content = self.emb_model.to_emb(content)
        distances , idxs = self.index.search(content,topn) # mlivus
        results = self.qa_df.iloc[ idxs[0]]
        return results


class MyDataBase:
    def __init__(self, path, name=None):
        if name is None:
            name = path
        print(os.path.join(path, "txt"))
        if not os.path.exists(os.path.join(path, "txt")):
            print("对应的知识库txt文件夹不存在")
            exit(-999)

        if not os.path.exists(os.path.join(path, "qa")):
            print("对应的知识库没有进行QA问答对抽取")
            exit(-999)

        if not os.path.exists(".cache"):
            os.mkdir(".cache")
        self.name = name
        self.document = MyDocument(os.path.join(path, "txt"), name)
        self.emb_database = None

    def create_emb_database(self, qa_df):
        self.emb_database = MyEmbDatabase("moka-ai_m3e-base", qa_df, self.name)
        self.qa_df = qa_df

    def search(self, text, topn=3):
        if not self.emb_database:
            self.emb_database = MyEmbDatabase("moka-ai_m3e-base", None, self.name)
        return self.emb_database.search(text, topn)


def load_database(dir_path="data/database_dir"):
    dirs = [name for name in os.listdir(dir_path) if os.path.isdir(f"{dir_path}/{name}")]

    database_list = []
    database_namelist = []

    for dir in dirs:
        database = MyDataBase(f"{dir_path}/{dir}",dir)

        database_list.append(database)
        database_namelist.append(dir)

    return database_list,database_namelist


if __name__ == "__main__":
    # my_documents = MyDocument("data/小米汽车", "小米汽车")
    # my_documents_database = MyEmbDatabase("moka-ai_m3e-base", my_documents.contents, "小米汽车")
    # result = my_documents_database.search("小米 SU7 内饰")
    # print(result)
    database = MyDataBase("data/database_dir/小米汽车")
    result = database.search("小米 SU7 内饰")
    print(result)
