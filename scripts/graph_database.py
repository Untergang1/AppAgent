import os
import hashlib

from sentence_transformers import SentenceTransformer
import neo4j


def calculate_hash(xml_path, algorithm="sha256"):
    hash_func = getattr(hashlib, algorithm)()
    with open(xml_path, "rb") as f:
        hash_func.update(f.read())
    return hash_func.hexdigest()

class GraphDatabase:
    def __init__(self):
        self.URI = 'bolt://localhost:7687'
        self.AUTH = (os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
        self.DB_NAME = 'AppAgent'

        self.driver = neo4j.GraphDatabase.driver(self.URI, auth=self.AUTH)
        self.driver.verify_connectivity()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # vector size 384
        self.create_index()

    def create_index(self):
        self.driver.execute_query('''
        CREATE VECTOR INDEX fun_desc IF NOT EXISTS
        FOR (n:XMLNode)
        ON n.embedding
        OPTIONS {indexConfig: {
            `vector.dimensions`: 384,
            `vector.similarity_function`: 'cosine'
        }}
        ''')

    def creat_node(self, hashid, fun_desc, xml_path):
        embedding = self.embedding_model.encode(fun_desc)
        summary = self.driver.execute_query('''
        MERGE (n:XMLNode {id: $hashid})
        SET n += {fun_desc : $fun_desc, embedding: $embedding, xml_path: $xml_path}
        ''', hashid=hashid, fun_desc=fun_desc, embedding=embedding, xml_path=xml_path, database_=self.DB_NAME).summary

        count = summary.counters.nodes_created
        if count != 0:
            print(f"{count} node created")
        else:
            print("1 node updated")

        return hashid

    def create_relationship(self, pre_id, post_id, action):
        summary = self.driver.execute_query('''
        MATCH (pre:XMLNode {id: $pre_id}), (post:XMLNode {id: $post_id})
        MERGE (pre)-[r:act {action: $action}]->(post)
        ''', pre_id=pre_id, post_id=post_id, action=action, database_=self.DB_NAME).summary

        count = summary.counters.relationships_created
        if count != 0:
            print(f"{count} relationship created")
        else:
            print("1 relationship updated")

    def query(self, cur_hashid, tar):
        query_embedding = self.embedding_model.encode(tar)
        related_paths, _, _ = self.driver.execute_query('''
            CALL db.index.vector.queryNodes('fun_desc', 3, $query_embedding)
            YIELD node, score
            WHERE score > 0.7
            WITH node.id AS tar_id
            CALL (tar_id) {
                MATCH p = SHORTEST 1 (start:XMLNode {id: $cur_hashid})--+(end:XMLNode {id: tar_id})
                RETURN p
            }
            RETURN reduce(result = "", i IN range(0, length(p)-1) | 
                result + "(function description of screen node:" + nodes(p)[i].fun_desc + 
                ")-[action:" + relationships(p)[i].action + "]->"
            ) + "(function description of the end screen node:" + nodes(p)[-1].fun_desc + ")" AS path
            ''',cur_hashid=cur_hashid, query_embedding=query_embedding, database_=self.DB_NAME)

        count = len(related_paths)
        if count == 0:
            print("No related paths found")
        else:
            print(f"{count} related paths found")
            for record in related_paths:
                print(record.data()['path'])

        return [record.data()['path'] for record in related_paths]

def main():
    db = GraphDatabase()
    xml_path1 = "./tasks/task_system_2024-05-14_21-52-51/task_system_2024-05-14_21-52-51_1.xml"
    hashid1 = calculate_hash(xml_path1)
    db.creat_node(hashid1, 'homepage')

    xml_path2 = "./tasks/task_system_2024-05-20_20-18-50/task_system_2024-05-20_20-18-50_3.xml"
    hashid2 = calculate_hash(xml_path2)
    db.creat_node(hashid2, 'display setting')

    db.create_relationship(hashid1, hashid2, "tap 2")
    # db.query(hashid1, 'display settings')
    db.query(hashid1, 'display settings')


if __name__ == "__main__":
    main()


