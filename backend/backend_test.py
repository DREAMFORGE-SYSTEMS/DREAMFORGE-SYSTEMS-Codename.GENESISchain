import unittest
import requests
import json
import time
import os

BACKEND_URL = "https://f645c1b7-3502-4044-aa09-a8aca94d492b.preview.emergentagent.com"

class GenesisChainTest(unittest.TestCase):
    def setUp(self):
        self.base_url = f"{BACKEND_URL}/api"
        
    def test_1_blockchain_endpoints(self):
        """Test blockchain related endpoints"""
        print("\nTesting blockchain endpoints...")
        
        # Test root endpoint
        response = requests.get(f"{self.base_url}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "GenesisChain API")
        print("✓ Root endpoint working")

        # Test chain endpoint
        response = requests.get(f"{self.base_url}/chain")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("chain", data)
        self.assertIn("length", data)
        print("✓ Chain endpoint working")

    def test_2_mining_functionality(self):
        """Test mining with quantum-enhanced features"""
        print("\nTesting mining functionality...")
        
        # Get initial chain state
        initial_chain = requests.get(f"{self.base_url}/chain").json()
        initial_length = len(initial_chain["chain"])

        # Mine a new block
        response = requests.get(f"{self.base_url}/mine")
        self.assertEqual(response.status_code, 200)
        mine_data = response.json()
        self.assertIn("message", mine_data)
        self.assertIn("New Block Forged", mine_data["message"])
        print("✓ Mining endpoint working")

        # Verify chain length increased
        new_chain = requests.get(f"{self.base_url}/chain").json()
        self.assertEqual(len(new_chain["chain"]), initial_length + 1)
        print("✓ Chain length increased after mining")

    def test_3_transaction_functionality(self):
        """Test transaction creation and retrieval"""
        print("\nTesting transaction functionality...")
        
        # Create a test transaction
        test_tx = {
            "sender": "test_sender",
            "recipient": "test_recipient",
            "amount": 10.0
        }
        
        response = requests.post(
            f"{self.base_url}/transactions/new",
            json=test_tx
        )
        self.assertEqual(response.status_code, 201)
        print("✓ Transaction creation working")

        # Verify transaction appears in list
        response = requests.get(f"{self.base_url}/transactions")
        self.assertEqual(response.status_code, 200)
        transactions = response.json()["transactions"]
        self.assertTrue(any(
            tx["sender"] == "test_sender" and 
            tx["recipient"] == "test_recipient" 
            for tx in transactions
        ))
        print("✓ Transaction retrieval working")

    def test_4_data_input_and_self_replication(self):
        """Test data input processing and self-replication"""
        print("\nTesting data input and self-replication...")
        
        # Submit test data
        test_data = {
            "content": "Testing QNodeOS-inspired self-replication",
            "timestamp": time.time()
        }
        
        response = requests.post(
            f"{self.base_url}/data-input",
            json=test_data
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        
        # Verify response structure
        self.assertIn("message", result)
        self.assertIn("hash", result)
        self.assertIn("data_id", result)
        self.assertIn("generated_content", result)
        self.assertIn("replications", result)
        print("✓ Data input processing working")

        # Verify self-replication occurred
        self.assertGreater(result["replications"], 0)
        print("✓ Self-replication mechanism working")

        # Check data inputs endpoint
        response = requests.get(f"{self.base_url}/data-inputs")
        self.assertEqual(response.status_code, 200)
        data_inputs = response.json()["data_inputs"]
        
        # Verify original data and replications are present
        original_data = next(
            (d for d in data_inputs if d["source"] == "user_input"), 
            None
        )
        self.assertIsNotNone(original_data)
        
        replications = [d for d in data_inputs if d["parent_data_id"] == original_data["data_id"]]
        self.assertGreater(len(replications), 0)
        print("✓ Data inputs retrieval working")
        print("✓ Self-replicated data verified")

if __name__ == '__main__':
    unittest.main(verbosity=2)