from sentence_transformers import SentenceTransformer


class MyEmbModel():
    def __init__(self, model_dir):
        self.model = SentenceTransformer(model_dir)

    def to_emb(self, sentence):
        if isinstance(sentence, str):
            sentence = [sentence]
        return self.model.encode(sentence)


if __name__ == "__main__":
    my_documents = MyEmbModel("moka-ai_m3e-base")
    sent_vec = my_documents.to_emb("大模型是什么")
    print("vec dim:", len(sent_vec[0]))