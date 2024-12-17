import os
import hashlib

from sentence_transformers import SentenceTransformer
import neo4j

from .utils import print_with_color


log_color = 'blue'


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
        CREATE VECTOR INDEX fun_desc_embedding IF NOT EXISTS
        FOR (n:XMLNode)
        ON n.embedding
        OPTIONS {indexConfig: {
            `vector.dimensions`: 384,
            `vector.similarity_function`: 'cosine'
        }}
        ''')


    def get_most_similar_node(self, query_embedding):
        records, _, _ = self.driver.execute_query('''
        CALL db.index.vector.queryNodes('fun_desc_embedding', 1, $query_embedding)
        YIELD node, score
        WHERE score > 0.95
        RETURN node.id AS node_id, score, node.embedding AS embedding
        ORDER BY score DESC
        LIMIT 1
        ''', query_embedding=query_embedding, database_=self.DB_NAME)

        return records[0] if len(records) > 0 else None

    def create_or_update_node(self, hashid, fun_desc, xml_path):
        embedding = self.embedding_model.encode(fun_desc)
        # First, check if there's an existing similar node based on the embedding
        similar_node = self.get_most_similar_node(embedding)

        if similar_node:
            # If the most similar node exists and similarity score is high enough, update it
            new_embedding = embedding
            self.driver.execute_query('''
            MATCH (n:XMLNode {id: $node_id})
            SET n.embedding = $new_embedding, n.id = $new_hashid, n.fun_desc = $fun_desc, n.xml_path = $xml_path
            ''', node_id=similar_node['node_id'], new_embedding=new_embedding, new_hashid=hashid, fun_desc=fun_desc, xml_path=xml_path, database_=self.DB_NAME)

            print_with_color(f"Node updated with score: {similar_node['score']:.3f}", color=log_color)
            return similar_node['node_id']
        else:
            # If no similar node or similarity is too low, create a new node
            self.driver.execute_query('''
            MERGE (n:XMLNode {id: $hashid})
            SET n += {fun_desc: $fun_desc, embedding: $embedding, xml_path: $xml_path}
            ''', hashid=hashid, fun_desc=fun_desc, embedding=embedding, xml_path=xml_path, database_=self.DB_NAME)

            print_with_color(f"New node created.", color=log_color)
            return hashid

    def create_or_update_relationship(self, pre_id, post_id, action):
        # Calculate the embedding for the action
        action_embedding = self.embedding_model.encode(action)

        # Check for existing similar relationships
        similar_relationship, _, _ = self.driver.execute_query('''
        MATCH (pre:XMLNode {id: $pre_id})-[r:act]->(post:XMLNode {id: $post_id})
        RETURN r, r.action_embedding AS existing_embedding, r.action AS existing_action,
               vector.similarity.cosine(r.action_embedding, $action_embedding) AS similarity
        ORDER BY similarity DESC
        LIMIT 1
        ''', pre_id=pre_id, post_id=post_id, action_embedding=action_embedding, database_=self.DB_NAME)

        # If a similar relationship exists
        if len(similar_relationship) > 0 and similar_relationship[0]['similarity'] >= 0.95:
            # Update the existing relationship
            existing_embedding = similar_relationship[0]['existing_embedding']
            updated_embedding = action_embedding
            self.driver.execute_query('''
            MATCH (pre:XMLNode {id: $pre_id})-[r:act {action: $existing_action}]->(post:XMLNode {id: $post_id})
            SET r.action = $new_action, r.action_embedding = $updated_embedding
            ''', pre_id=pre_id, post_id=post_id, existing_action=similar_relationship[0]['existing_action'],
                                      new_action=action, updated_embedding=updated_embedding, database_=self.DB_NAME)
            print_with_color(f"Relationship updated with score: {similar_relationship[0]['similarity']:.3f}.", color=log_color)
        else:
            # Create a new relationship if no similar one exists
            self.driver.execute_query('''
            MATCH (pre:XMLNode {id: $pre_id}), (post:XMLNode {id: $post_id})
            CREATE (pre)-[r:act {action: $action, action_embedding: $action_embedding}]->(post)
            ''', pre_id=pre_id, post_id=post_id, action=action, action_embedding=action_embedding,
                                      database_=self.DB_NAME)
            print_with_color(f"New relationship created.", color=log_color)

    def query_realted_paths(self, cur_hashid, tar):
        query_embedding = self.embedding_model.encode(tar)
        related_paths, _, _ = self.driver.execute_query('''
            CALL db.index.vector.queryNodes('fun_desc_embedding', 3, $query_embedding)
            YIELD node, score
            WHERE score > 0.6
            WITH node.id AS tar_id
            CALL (tar_id) {
                MATCH p = SHORTEST 1 (start:XMLNode {id: $cur_hashid})--+(end:XMLNode {id: tar_id})
                RETURN p
            }
            RETURN reduce(result = "", i IN range(0, length(p)-1) | 
                result + "(description of screen node:" + nodes(p)[i].fun_desc + 
                ")-[action:" + relationships(p)[i].action + "]->"
            ) + "(description of the end screen node:" + nodes(p)[-1].fun_desc + ")" AS path
            ''',cur_hashid=cur_hashid, query_embedding=query_embedding, database_=self.DB_NAME)

        count = len(related_paths)
        if count == 0:
            print_with_color("No related paths found", color=log_color)
        else:
            print_with_color(f"{count} related paths found", color=log_color)
            for record in related_paths:
                print_with_color(record.data()['path'], color=log_color)

        return [record.data()['path'] for record in related_paths]

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")


def main():
    db = GraphDatabase()
    xml_path1 = "./tasks/task_system_2024-05-14_21-52-51/task_system_2024-05-14_21-52-51_1.xml"
    hashid1 = calculate_hash(xml_path1)
    db.create_or_update_node(hashid1, 'homepage', xml_path1)

    xml_path2 = "./tasks/task_system_2024-05-20_20-18-50/task_system_2024-05-20_20-18-50_3.xml"
    hashid2 = calculate_hash(xml_path2)
    db.create_or_update_node(hashid2, 'display setting', xml_path2)

    db.create_or_update_relationship(hashid1, hashid2, "tap 2")
    db.query_realted_paths(hashid1, 'display settings')


if __name__ == "__main__":
    main()

