import unittest
import requests
import json
import time
from typing import Dict, Any

class GenesisChainTest(unittest.TestCase):
    def setUp(self):
        self.base_url = "https://f645c1b7-3502-4044-aa09-a8aca94d492b.preview.emergentagent.com"
        self.test_transaction = {
            "sender": "test_sender",
            "recipient": "test_recipient",
            "amount": 1.0
        }
        self.test_data = {
            "content": "Test data for quantum processing",
            "timestamp": time.time()
        }

    def test_1_api_root(self):
        """Test the root API endpoint"""
        response = requests.get(f"{self.base_url}/api")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "GenesisChain API")
        print("â Root API test passed")

    def test_2_blockchain_operations(self):
        """Test blockchain operations including mining"""
        # Get initial chain
        response = requests.get(f"{self.base_url}/api/chain")
        self.assertEqual(response.status_code, 200)
        initial_length = len(response.json()["chain"])
        print(f"Initial blockchain length: {initial_length}")

        # Mine a new block
        mine_start = time.time()
        response = requests.get(f"{self.base_url}/api/mine")
        mine_time = time.time() - mine_start
        self.assertEqual(response.status_code, 200)
        print(f"â¡ Mining completed in {mine_time:.2f} seconds")

        # Verify chain length increased
        response = requests.get(f"{self.base_url}/api/chain")
        new_length = len(response.json()["chain"])
        self.assertEqual(new_length, initial_length + 1)
        print("â Blockchain mining test passed")

    def test_3_transaction_flow(self):
        """Test creating and confirming a transaction"""
        # Create transaction
        response = requests.post(
            f"{self.base_url}/api/transactions/new",
            json=self.test_transaction
        )
        self.assertEqual(response.status_code, 201)
        print("Transaction created successfully")

        # Mine a block to confirm transaction
        response = requests.get(f"{self.base_url}/api/mine")
        self.assertEqual(response.status_code, 200)
        print("Block mined to confirm transaction")

        # Verify transaction is in the list
        response = requests.get(f"{self.base_url}/api/transactions")
        transactions = response.json()["transactions"]
        found = False
        for tx in transactions:
            if (tx["sender"] == self.test_transaction["sender"] and 
                tx["recipient"] == self.test_transaction["recipient"]):
                found = True
                break
        self.assertTrue(found)
        print("â Transaction flow test passed")

    def test_4_data_input_and_replication(self):
        """Test data input processing and self-replication"""
        # Submit data
        response = requests.post(
            f"{self.base_url}/api/data-input",
            json=self.test_data
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        
        # Verify generated content
        self.assertIn("generated_content", result)
        self.assertGreater(result["generated_content"]["images"], 0)
        self.assertGreater(result["generated_content"]["audio"], 0)
        print(f"Generated {result['generated_content']['images']} images and {result['generated_content']['audio']} audio tracks")

        # Verify replications
        self.assertGreater(result["replications"], 0)
        print(f"Created {result['replications']} self-replications")

        # Check data inputs list
        response = requests.get(f"{self.base_url}/api/data-inputs")
        self.assertEqual(response.status_code, 200)
        data_inputs = response.json()["data_inputs"]
        self.assertGreater(len(data_inputs), 0)
        print("â Data input and replication test passed")

if __name__ == '__main__':
    unittest.main(verbosity=2)
