# from langchain_core.documents import Document
# from langchain_qdrant import QdrantVectorStore
# from langchain_ollama import OllamaEmbeddings
# from qdrant_client import QdrantClient, models


def create_database(logs):
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # collection_name = "robot_logs"
    # # Указываем путь к папке, где будет лежать база
    # storage_path = "./qdrant_data"
    #
    # # 1. Инициализируем клиент с привязкой к диску
    # client = QdrantClient(path=storage_path)
    #
    # # 2. Проверяем, существует ли уже коллекция
    # if not client.collection_exists(collection_name):
    #     print(f"Коллекция '{collection_name}' не найдена. Создаем новую...")
    #
    #     # Вычисляем размер вектора (делаем 1 пробный запрос к Ollama)
    #     sample_embedding = embeddings.embed_query("test_dimension_check")
    #     vector_size = len(sample_embedding)
    #
    #     # Создаем коллекцию
    #     client.create_collection(
    #         collection_name=collection_name,
    #         vectors_config=models.VectorParams(
    #             size=vector_size,
    #             distance=models.Distance.COSINE
    #         )
    #     )
    #
    #     # Создаем объект хранилища LangChain
    #     db = QdrantVectorStore(
    #         client=client,
    #         embedding=embeddings,
    #         collection_name=collection_name
    #     )
    #
    #     # Если есть стартовые логи, добавляем их (только при первом создании!)
    #     if logs:
    #         print(f"Добавляем {len(logs)} стартовых логов...")
    #         db.add_documents(logs)
    #
    #     return db
    #
    # else:
    #     print(f"Коллекция '{collection_name}' найдена на диске. Загружаем...")
    #     # Если коллекция уже есть, просто подключаемся к ней
    #     # Примечание: logs здесь игнорируются, чтобы не дублировать их при каждом запуске
    #     return QdrantVectorStore(
    #         client=client,
    #         embedding=embeddings,
    #         collection_name=collection_name
    #     )
    return None

def update_query(query: str, db, k=1, threshold=0.5) -> str:
    # # Ищем похожие записи
    # found_query = db.similarity_search_with_score(query, k=k)
    #
    # found_any = False
    # for q, score in found_query:
    #     # Для Cosine similarity score обычно от 0 до 1 (в Qdrant/LangChain)
    #     if score > threshold:
    #         found_any = True
    #         answer = q.metadata.get('answer', 'N/A')
    #         query = query + f"\nSimilar example: {q.page_content}\nAnomalies: {answer}.\n"
    #
    # if not found_any:
    #     # Можно раскомментировать, если нужно явное сообщение
    #     # return query + '\nNo similar examples found.'
    #     pass

    return query


def add_query(query, output, db):
    # doc = Document(
    #     page_content=query,
    #     metadata={
    #         "answer": output
    #     }
    # )
    # db.add_documents([doc])
    # print("Новая запись сохранена в базу.")
    return None