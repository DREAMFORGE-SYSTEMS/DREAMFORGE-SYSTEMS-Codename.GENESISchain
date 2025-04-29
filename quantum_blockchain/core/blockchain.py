"""
GenesisChain Core Blockchain Implementation

This module implements the core blockchain functionality for GenesisChain,
with a focus on quantum-resistant security and self-replication.
"""

import time
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple, Union, Set
import threading
import logging
import copy

# Import our quantum-resistant cryptography module
from quantum_blockchain.cryptography.quantum_resistant import QuantumResistantCrypto

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GenesisChain")


class MerkleTree:
    """
    Implementation of a Merkle Tree for efficient and secure verification of transaction sets.
    Uses quantum-resistant SHA3-256 for hashing.
    """
    
    @staticmethod
    def build_merkle_root(transactions: List[Dict[str, Any]]) -> str:
        """
        Build a Merkle Root from a list of transactions
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            Merkle root hash as a hex string
        """
        if not transactions:
            # Empty tree has a zero hash
            return hashlib.sha3_256(b'').hexdigest()
        
        # Generate leaf hashes from transactions
        leaves = [QuantumResistantCrypto.hash_hex(tx) for tx in transactions]
        
        # If odd number of leaves, duplicate the last one
        if len(leaves) % 2 == 1:
            leaves.append(leaves[-1])
        
        # Build the tree bottom-up
        while len(leaves) > 1:
            # Process pairs of leaves
            new_level = []
            for i in range(0, len(leaves), 2):
                # Combine two leaves and hash them
                combined = leaves[i] + leaves[i+1]
                new_hash = hashlib.sha3_256(combined.encode()).hexdigest()
                new_level.append(new_hash)
            
            # Move up to the next level
            leaves = new_level
            
            # If odd number of nodes at this level, duplicate the last one
            if len(leaves) % 2 == 1 and len(leaves) > 1:
                leaves.append(leaves[-1])
        
        # Return the root hash
        return leaves[0]
    
    @staticmethod
    def verify_transaction(transaction: Dict[str, Any], merkle_root: str, all_transactions: List[Dict[str, Any]]) -> bool:
        """
        Verify that a transaction is part of a block with the given Merkle root
        
        Args:
            transaction: Transaction to verify
            merkle_root: Expected Merkle root hash
            all_transactions: All transactions in the block
            
        Returns:
            True if the transaction is part of the block, False otherwise
        """
        # Hash all transactions
        tx_hashes = [QuantumResistantCrypto.hash_hex(tx) for tx in all_transactions]
        
        # Find the transaction in the list
        tx_hash = QuantumResistantCrypto.hash_hex(transaction)
        try:
            tx_index = tx_hashes.index(tx_hash)
        except ValueError:
            # Transaction not in the block
            return False
        
        # Build the authentication path
        auth_path = []
        n = len(tx_hashes)
        
        # If odd number of transactions, duplicate the last one
        if n % 2 == 1:
            tx_hashes.append(tx_hashes[-1])
            n += 1
        
        # Generate the authentication path
        index = tx_index
        level_size = n
        
        while level_size > 1:
            # Determine if we're on the left or right
            is_right = index % 2 == 1
            
            if is_right:
                # We're on the right, so we need the left sibling
                auth_path.append(("left", tx_hashes[index - 1]))
            else:
                # We're on the left, so we need the right sibling
                if index + 1 < level_size:
                    auth_path.append(("right", tx_hashes[index + 1]))
                else:
                    # No right sibling, use ourselves (this happens with odd counts)
                    auth_path.append(("right", tx_hashes[index]))
            
            # Move to the next level up
            index = index // 2
            tx_hashes = [
                hashlib.sha3_256((tx_hashes[i] + tx_hashes[i+1]).encode()).hexdigest()
                for i in range(0, level_size, 2)
            ]
            level_size = len(tx_hashes)
            
            # If odd number of nodes at this level, duplicate the last one
            if level_size % 2 == 1 and level_size > 1:
                tx_hashes.append(tx_hashes[-1])
                level_size += 1
        
        # Verify the authentication path
        current_hash = QuantumResistantCrypto.hash_hex(transaction)
        
        for direction, sibling in auth_path:
            if direction == "left":
                # Sibling is on the left
                current_hash = hashlib.sha3_256((sibling + current_hash).encode()).hexdigest()
            else:
                # Sibling is on the right
                current_hash = hashlib.sha3_256((current_hash + sibling).encode()).hexdigest()
        
        # The final hash should match the Merkle root
        return current_hash == merkle_root


class Block:
    """
    Quantum-secure blockchain block
    """
    
    def __init__(self, index: int, previous_hash: str, timestamp: Optional[int] = None, 
                 transactions: Optional[List[Dict[str, Any]]] = None, nonce: int = 0):
        """
        Initialize a new block
        
        Args:
            index: Block index (height in the blockchain)
            previous_hash: Hash of the previous block
            timestamp: Block creation timestamp (default: current time)
            transactions: List of transactions in the block
            nonce: Nonce value for proof-of-work
        """
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp or int(time.time())
        self.transactions = transactions or []
        self.nonce = nonce
        
        # Calculate the Merkle root
        self.merkle_root = MerkleTree.build_merkle_root(self.transactions)
        
        # The hash will be calculated when needed
        self._hash = None
    
    @property
    def hash(self) -> str:
        """Get the hash of the block (calculated on demand)"""
        if self._hash is None:
            self._hash = self.calculate_hash()
        return self._hash
    
    def calculate_hash(self) -> str:
        """
        Calculate the hash of the block using quantum-resistant SHA3-256
        
        Returns:
            Hash of the block as a hex string
        """
        # Create a dictionary with block data
        block_data = {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "merkle_root": self.merkle_root,
            "nonce": self.nonce
        }
        
        # Use our quantum-resistant hash function
        return QuantumResistantCrypto.hash_hex(block_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the block to a dictionary
        
        Returns:
            Dictionary representation of the block
        """
        return {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "transactions": self.transactions,
            "merkle_root": self.merkle_root,
            "nonce": self.nonce,
            "hash": self.hash
        }
    
    @classmethod
    def from_dict(cls, block_dict: Dict[str, Any]) -> 'Block':
        """
        Create a block from a dictionary
        
        Args:
            block_dict: Dictionary representation of the block
            
        Returns:
            Block instance
        """
        block = cls(
            index=block_dict["index"],
            previous_hash=block_dict["previous_hash"],
            timestamp=block_dict["timestamp"],
            transactions=block_dict["transactions"],
            nonce=block_dict["nonce"]
        )
        
        # Verify that the hash matches
        if "hash" in block_dict and block.hash != block_dict["hash"]:
            logger.warning(f"Block hash mismatch: {block.hash} != {block_dict['hash']}")
        
        return block


class QuantumBlockchain:
    """
    Main blockchain implementation with quantum-resistant security
    """
    
    def __init__(self, difficulty: int = 4):
        """
        Initialize a new blockchain
        
        Args:
            difficulty: Mining difficulty (number of leading zeros in block hash)
        """
        self.chain = []  # List of blocks
        self.pending_transactions = []  # Transactions waiting to be mined
        self.difficulty = difficulty  # Mining difficulty
        self.mining_reward = 1.0  # Reward for mining a block
        
        # Set for tracking spent transaction outputs (for double-spend prevention)
        self.spent_outputs = set()
        
        # Thread lock for concurrent access
        self.lock = threading.RLock()
        
        # Create the genesis block
        self._create_genesis_block()
    
    def _create_genesis_block(self):
        """Create the genesis block (first block in the chain)"""
        genesis_block = Block(
            index=0,
            previous_hash="0" * 64,
            timestamp=int(time.time()),
            transactions=[
                {
                    "sender": "0",
                    "recipient": "genesis-address",
                    "amount": 100.0,
                    "timestamp": int(time.time()),
                    "signature": "genesis-signature",
                    "transaction_id": "genesis"
                }
            ]
        )
        
        self.chain.append(genesis_block)
        logger.info(f"Genesis block created with hash {genesis_block.hash}")
    
    @property
    def last_block(self) -> Block:
        """Get the last block in the chain"""
        return self.chain[-1]
    
    def add_transaction(self, transaction: Dict[str, Any], verify: bool = True) -> bool:
        """
        Add a transaction to the pending transactions
        
        Args:
            transaction: Transaction dictionary
            verify: Whether to verify the transaction before adding
            
        Returns:
            True if the transaction was added, False otherwise
        """
        # Create a deep copy to avoid modifications
        tx = copy.deepcopy(transaction)
        
        # Generate transaction ID if not present
        if "transaction_id" not in tx:
            tx["transaction_id"] = QuantumResistantCrypto.hash_hex({
                "sender": tx["sender"],
                "recipient": tx["recipient"],
                "amount": tx["amount"],
                "timestamp": tx["timestamp"],
                "nonce": tx.get("nonce", 0)
            })
        
        with self.lock:
            # Verify the transaction if requested
            if verify and not self.verify_transaction(tx):
                logger.warning(f"Transaction verification failed: {tx['transaction_id']}")
                return False
            
            # Check if this transaction is already pending
            for pending_tx in self.pending_transactions:
                if pending_tx["transaction_id"] == tx["transaction_id"]:
                    logger.info(f"Transaction {tx['transaction_id']} is already pending")
                    return False
            
            # Add to pending transactions
            self.pending_transactions.append(tx)
            logger.info(f"Transaction {tx['transaction_id']} added to pending transactions")
            
            return True
    
    def verify_transaction(self, transaction: Dict[str, Any]) -> bool:
        """
        Verify a transaction's signature and validity
        
        Args:
            transaction: Transaction to verify
            
        Returns:
            True if the transaction is valid, False otherwise
        """
        # Skip verification for mining reward transactions
        if transaction["sender"] == "0":
            return True
        
        # Check if the transaction has a valid signature
        try:
            # Prepare the transaction data for signature verification
            tx_data = transaction.copy()
            signature_hex = tx_data.pop("signature", None)
            
            if not signature_hex:
                logger.warning("Transaction has no signature")
                return False
            
            # Convert hex signature to bytes
            signature = bytes.fromhex(signature_hex)
            
            # For multi-signature transactions
            if tx_data.get("type") == "multi_signature":
                # Verify that we have enough signatures
                signatures = tx_data.get("signatures", {})
                required = tx_data.get("required_signatures", 0)
                
                if len(signatures) < required:
                    logger.warning(f"Multi-sig transaction has {len(signatures)} signatures, {required} required")
                    return False
                
                # Verify each signature (simplified - a real implementation would be more complex)
                # This is a placeholder for actual multi-sig verification
                return True
            
            # For standard transactions, extract algorithm from transaction
            algorithm = tx_data.get("algorithm", "falcon")
            
            # Create a simulated public key structure for verification
            # In a real implementation, we would retrieve the actual public key from a registry
            # This is a simplified placeholder
            sender = tx_data["sender"]
            public_key = {
                "algorithm": algorithm,
                "key_component": hashlib.sha3_256(sender.encode()).hexdigest()
            }
            
            # Verify the signature (simplified - this is a placeholder)
            # In a real implementation, we would use actual signature verification
            # return QuantumResistantCrypto.verify(public_key, tx_data, signature)
            
            # For this prototype, just return True
            return True
            
        except Exception as e:
            logger.error(f"Error verifying transaction: {str(e)}")
            return False
    
    def mine_block(self, miner_address: str) -> Optional[Block]:
        """
        Mine a new block with the pending transactions
        
        Args:
            miner_address: Address to receive the mining reward
            
        Returns:
            The newly mined block, or None if mining failed
        """
        with self.lock:
            # Nothing to mine if there are no pending transactions
            if not self.pending_transactions and miner_address != "genesis-address":
                logger.info("No pending transactions to mine")
                return None
            
            # Create a copy of pending transactions to avoid modifications during mining
            transactions = copy.deepcopy(self.pending_transactions)
            
            # Add mining reward transaction
            mining_reward_tx = {
                "sender": "0",  # "0" signifies a mining reward
                "recipient": miner_address,
                "amount": self.mining_reward,
                "timestamp": int(time.time()),
                "transaction_id": f"mining-reward-{int(time.time())}",
                "signature": "reward-signature"  # Placeholder
            }
            transactions.append(mining_reward_tx)
            
            # Create a new block
            new_block = Block(
                index=len(self.chain),
                previous_hash=self.last_block.hash,
                timestamp=int(time.time()),
                transactions=transactions
            )
            
            # Mine the block (find a valid nonce)
            self._proof_of_work(new_block)
            
            # Add the block to the chain
            self.chain.append(new_block)
            
            # Clear pending transactions
            self.pending_transactions = []
            
            # Update spent outputs
            for tx in transactions:
                if "transaction_id" in tx:
                    self.spent_outputs.add(f"{tx['sender']}:{tx['transaction_id']}")
            
            logger.info(f"New block mined with hash {new_block.hash}")
            return new_block
    
    def _proof_of_work(self, block: Block) -> None:
        """
        Find a nonce that gives the block a hash with the required difficulty
        
        Args:
            block: Block to mine
            
        Modifies the block's nonce in place
        """
        target_prefix = "0" * self.difficulty
        
        while not block.hash.startswith(target_prefix):
            block.nonce += 1
            block._hash = None  # Reset the hash to force recalculation
    
    def is_chain_valid(self) -> bool:
        """
        Verify the integrity of the blockchain
        
        Returns:
            True if the blockchain is valid, False otherwise
        """
        with self.lock:
            # Check each block in the chain
            for i in range(1, len(self.chain)):
                current_block = self.chain[i]
                previous_block = self.chain[i - 1]
                
                # Check block hash
                if current_block.hash != current_block.calculate_hash():
                    logger.error(f"Block {current_block.index} has invalid hash")
                    return False
                
                # Check link to previous block
                if current_block.previous_hash != previous_block.hash:
                    logger.error(f"Block {current_block.index} has invalid previous hash")
                    return False
                
                # Check Merkle root
                if current_block.merkle_root != MerkleTree.build_merkle_root(current_block.transactions):
                    logger.error(f"Block {current_block.index} has invalid Merkle root")
                    return False
            
            return True
    
    def get_balance(self, address: str) -> float:
        """
        Calculate the balance of an address
        
        Args:
            address: Address to calculate balance for
            
        Returns:
            Balance as a float
        """
        with self.lock:
            balance = 0.0
            
            # Check all transactions in all blocks
            for block in self.chain:
                for tx in block.transactions:
                    if tx["recipient"] == address:
                        balance += tx["amount"]
                    if tx["sender"] == address:
                        balance -= tx["amount"]
                        # Subtract fee if present
                        if "fee" in tx:
                            balance -= tx["fee"]
            
            # Check pending transactions
            for tx in self.pending_transactions:
                if tx["recipient"] == address:
                    balance += tx["amount"]
                if tx["sender"] == address:
                    balance -= tx["amount"]
                    # Subtract fee if present
                    if "fee" in tx:
                        balance -= tx["fee"]
            
            return max(0.0, balance)  # Ensure balance is not negative
    
    def replace_chain(self, new_chain: List[Block]) -> bool:
        """
        Replace the chain with a new one if it's longer and valid
        
        Args:
            new_chain: New chain to replace the current one
            
        Returns:
            True if the chain was replaced, False otherwise
        """
        with self.lock:
            # Check if the new chain is longer
            if len(new_chain) <= len(self.chain):
                logger.info("Received chain is not longer than the current chain")
                return False
            
            # Check if the new chain is valid
            for i in range(1, len(new_chain)):
                current_block = new_chain[i]
                previous_block = new_chain[i - 1]
                
                # Check block hash
                if current_block.hash != current_block.calculate_hash():
                    logger.error(f"Received chain has invalid hash at block {current_block.index}")
                    return False
                
                # Check link to previous block
                if current_block.previous_hash != previous_block.hash:
                    logger.error(f"Received chain has invalid previous hash at block {current_block.index}")
                    return False
                
                # Check Merkle root
                if current_block.merkle_root != MerkleTree.build_merkle_root(current_block.transactions):
                    logger.error(f"Received chain has invalid Merkle root at block {current_block.index}")
                    return False
            
            # Replace the chain
            self.chain = new_chain
            
            # Update spent outputs
            self.spent_outputs = set()
            for block in self.chain:
                for tx in block.transactions:
                    if "transaction_id" in tx:
                        self.spent_outputs.add(f"{tx['sender']}:{tx['transaction_id']}")
            
            logger.info(f"Chain replaced with a new chain of length {len(new_chain)}")
            return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the blockchain to a dictionary
        
        Returns:
            Dictionary representation of the blockchain
        """
        return {
            "chain": [block.to_dict() for block in self.chain],
            "pending_transactions": self.pending_transactions,
            "difficulty": self.difficulty,
            "mining_reward": self.mining_reward
        }
    
    @classmethod
    def from_dict(cls, blockchain_dict: Dict[str, Any]) -> 'QuantumBlockchain':
        """
        Create a blockchain from a dictionary
        
        Args:
            blockchain_dict: Dictionary representation of the blockchain
            
        Returns:
            QuantumBlockchain instance
        """
        blockchain = cls(difficulty=blockchain_dict["difficulty"])
        
        # Clear the genesis block
        blockchain.chain = []
        
        # Add all blocks
        for block_dict in blockchain_dict["chain"]:
            blockchain.chain.append(Block.from_dict(block_dict))
        
        # Add pending transactions
        blockchain.pending_transactions = blockchain_dict["pending_transactions"]
        
        # Set mining reward
        blockchain.mining_reward = blockchain_dict["mining_reward"]
        
        # Update spent outputs
        blockchain.spent_outputs = set()
        for block in blockchain.chain:
            for tx in block.transactions:
                if "transaction_id" in tx:
                    blockchain.spent_outputs.add(f"{tx['sender']}:{tx['transaction_id']}")
        
        return blockchain


# Example usage
if __name__ == "__main__":
    # Create a new blockchain
    blockchain = QuantumBlockchain(difficulty=4)
    
    # Create a test transaction
    transaction = {
        "sender": "alice-address",
        "recipient": "bob-address",
        "amount": 5.0,
        "timestamp": int(time.time()),
        "transaction_id": f"test-{int(time.time())}",
        "signature": "simulated-signature",  # Simulated signature
    }
    
    # Add the transaction
    blockchain.add_transaction(transaction, verify=False)
    
    # Mine a block
    block = blockchain.mine_block("miner-address")
    
    print(f"Mined block with hash: {block.hash}")
    print(f"Chain length: {len(blockchain.chain)}")
    print(f"Alice's balance: {blockchain.get_balance('alice-address')}")
    print(f"Bob's balance: {blockchain.get_balance('bob-address')}")
    print(f"Miner's balance: {blockchain.get_balance('miner-address')}")