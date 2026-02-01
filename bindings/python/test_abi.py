import unittest
import os
import sys
from abi import ABI, AbiError

class TestABI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"Loading ABI from {os.getcwd()}...")
        cls.abi = ABI()
        print(f"ABI Version: {cls.abi.version()}")

    def test_version(self):
        self.assertEqual(self.abi.version(), "0.4.0")

    def test_database_lifecycle(self):
        db = self.abi.create_db(128)
        self.assertIsNotNone(db.handle)
        # cleanup happens in __del__

    def test_vector_ops(self):
        dim = 4
        db = self.abi.create_db(dim)
        
        # Insert
        vec1 = [1.0, 0.0, 0.0, 0.0]
        db.insert(1, vec1)
        
        vec2 = [0.0, 1.0, 0.0, 0.0]
        db.insert(2, vec2)
        
        # Search
        query = [1.0, 0.0, 0.0, 0.0]
        results = db.search(query, k=2)
        
        # We expect ID 1 to be the best match
        self.assertEqual(results[0]['id'], 1)
        # Cosine distance or similarity? WDBX defaults vary, but identical vectors should be top ranked. 
        
        print("\nSearch Results:")
        for res in results:
            print(f"  ID: {res['id']}, Score: {res['score']}")

if __name__ == '__main__':
    unittest.main()
