"""
GenesisChain Main Module

This is the main entry point for the GenesisChain blockchain system.
It initializes and connects all components of the system.
"""

import os
import logging
import threading
import time
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
import uuid

# Import GenesisChain components
from quantum_blockchain.core.blockchain import QuantumBlockchain, Block
from quantum_blockchain.cryptography.quantum_resistant import QuantumResistantCrypto
from quantum_blockchain.cryptography.wallet import Wallet
from quantum_blockchain.consensus.quantum_proof_of_work import QuantumProofOfWork, QuantumMiningOracle, QNPUSimulator
from quantum_blockchain.core.self_replication import SelfReplicationEngine, ContentGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GenesisChain")


class GenesisChain:
    """
    Main GenesisChain system
    
    This class coordinates all components of the GenesisChain blockchain.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the GenesisChain system
        
        Args:
            config: Configuration dictionary (uses default config if None)
        """
        self.config = config or self._default_config()
        logger.info("Initializing GenesisChain")
        
        # Initialize components
        self._init_blockchain()
        self._init_pow()
        self._init_self_replication()
        
        # Wallet storage
        self.wallets = {}
        
        # System metrics
        self.metrics = {
            "start_time": time.time(),
            "blocks_mined": 0,
            "transactions_processed": 0,
            "data_inputs_processed": 0,
            "assets_generated": 0
        }
        
        # Thread lock for concurrent access
        self.lock = threading.RLock()
        
        # Status flag
        self.running = False
        
        logger.info("GenesisChain initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """
        Create default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            "blockchain": {
                "difficulty": 4,
                "mining_reward": 1.0
            },
            "pow": {
                "target_block_time": 60.0,  # seconds
                "adjustment_period": 10  # blocks
            },
            "self_replication": {
                "max_level": 3,
                "replication_probability": 0.5
            },
            "quantum": {
                "qubits_available": 8,
                "simulation_level": "medium"  # low, medium, high
            }
        }
    
    def _init_blockchain(self) -> None:
        """Initialize the blockchain component"""
        config = self.config["blockchain"]
        self.blockchain = QuantumBlockchain(difficulty=config["difficulty"])
        self.blockchain.mining_reward = config["mining_reward"]
        logger.info(f"Blockchain initialized with difficulty {config['difficulty']}")
    
    def _init_pow(self) -> None:
        """Initialize the proof of work component"""
        config = self.config["pow"]
        
        # Initialize QNPU simulator
        qnpu_config = self.config["quantum"]
        qnpu = QNPUSimulator(qubits_available=qnpu_config["qubits_available"])
        
        # Initialize mining oracle with QNPU
        oracle = QuantumMiningOracle(qnpu=qnpu)
        
        # Initialize quantum proof of work
        self.pow = QuantumProofOfWork(
            difficulty=self.config["blockchain"]["difficulty"],
            oracle=oracle
        )
        logger.info("Quantum Proof of Work initialized")
    
    def _init_self_replication(self) -> None:
        """Initialize the self-replication component"""
        config = self.config["self_replication"]
        self.replication_engine = SelfReplicationEngine()
        logger.info("Self-Replication Engine initialized")
    
    def start(self) -> None:
        """Start the GenesisChain system"""
        with self.lock:
            if self.running:
                logger.warning("GenesisChain is already running")
                return
            
            self.running = True
            logger.info("Starting GenesisChain")
            
            # Start any background processes here if needed
    
    def stop(self) -> None:
        """Stop the GenesisChain system"""
        with self.lock:
            if not self.running:
                logger.warning("GenesisChain is not running")
                return
            
            self.running = False
            logger.info("Stopping GenesisChain")
            
            # Stop any background processes here if needed
    
    def create_wallet(self, name: str, seed: Optional[bytes] = None) -> Wallet:
        """
        Create a new wallet
        
        Args:
            name: Name of the wallet
            seed: Optional seed for key generation
            
        Returns:
            New wallet instance
        """
        with self.lock:
            wallet = Wallet(seed=seed, name=name)
            self.wallets[name] = wallet
            logger.info(f"Created wallet '{name}' with address {wallet.get_address()}")
            return wallet
    
    def get_wallet(self, name: str) -> Optional[Wallet]:
        """
        Get a wallet by name
        
        Args:
            name: Name of the wallet
            
        Returns:
            Wallet instance or None if not found
        """
        with self.lock:
            return self.wallets.get(name)
    
    def create_transaction(self, wallet_name: str, recipient: str, amount: float, 
                          fee: float = 0.001, data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Create and add a transaction
        
        Args:
            wallet_name: Name of the sending wallet
            recipient: Recipient address
            amount: Amount to send
            fee: Transaction fee
            data: Optional additional data
            
        Returns:
            Transaction dictionary or None if failed
        """
        with self.lock:
            # Get the wallet
            wallet = self.get_wallet(wallet_name)
            if wallet is None:
                logger.error(f"Wallet '{wallet_name}' not found")
                return None
            
            # Check balance
            sender_address = wallet.get_address()
            balance = self.blockchain.get_balance(sender_address)
            
            if balance < amount + fee:
                logger.error(f"Insufficient balance: {balance} < {amount + fee}")
                return None
            
            # Create the transaction
            transaction = wallet.create_transaction(recipient, amount, fee, data)
            
            # Add to blockchain
            if self.blockchain.add_transaction(transaction):
                self.metrics["transactions_processed"] += 1
                logger.info(f"Transaction created: {amount} from {sender_address} to {recipient}")
                return transaction
            else:
                logger.error("Failed to add transaction to blockchain")
                return None
    
    def mine_block(self, miner_wallet_name: str) -> Optional[Block]:
        """
        Mine a new block
        
        Args:
            miner_wallet_name: Name of the wallet to receive the mining reward
            
        Returns:
            Newly mined block or None if failed
        """
        with self.lock:
            # Get the miner's wallet
            wallet = self.get_wallet(miner_wallet_name)
            if wallet is None:
                logger.error(f"Miner wallet '{miner_wallet_name}' not found")
                return None
            
            # Get the miner's address
            miner_address = wallet.get_address()
            
            # Define hash function for proof of work
            def calculate_hash(block_header, nonce):
                header_copy = block_header.copy()
                header_copy["nonce"] = nonce
                return QuantumResistantCrypto.hash_hex(header_copy)
            
            # Prepare block header
            last_block = self.blockchain.last_block
            block_header = {
                "index": last_block.index + 1,
                "previous_hash": last_block.hash,
                "timestamp": int(time.time()),
                "transactions": [tx for tx in self.blockchain.pending_transactions],
                "previous_nonce": last_block.nonce
            }
            
            # Find a valid nonce using quantum-enhanced proof of work
            start_time = time.time()
            nonce, _ = self.pow.find_nonce(block_header, calculate_hash)
            end_time = time.time()
            mining_time = end_time - start_time
            
            if nonce == 0:
                logger.error("Failed to find a valid nonce")
                return None
            
            # Mine the block
            block = self.blockchain.mine_block(miner_address)
            
            if block is not None:
                self.metrics["blocks_mined"] += 1
                logger.info(f"Block {block.index} mined in {mining_time:.2f} seconds with nonce {nonce}")
                
                # Adjust difficulty if needed
                self._adjust_difficulty(mining_time)
                
                # Process mining operational data for self-replication
                asyncio.create_task(self._process_mining_data(mining_time, nonce, block.index))
                
                return block
            else:
                logger.error("Failed to mine block")
                return None
    
    def _adjust_difficulty(self, mining_time: float) -> None:
        """
        Adjust mining difficulty based on mining time
        
        Args:
            mining_time: Time taken to mine the last block
        """
        # Only adjust difficulty after a certain number of blocks
        if self.metrics["blocks_mined"] % self.config["pow"]["adjustment_period"] == 0:
            target_time = self.config["pow"]["target_block_time"]
            self.pow.adjust_difficulty(mining_time, target_time)
            
            # Update blockchain difficulty
            self.blockchain.difficulty = self.pow.difficulty
            logger.info(f"Adjusted difficulty to {self.pow.difficulty}")
    
    async def _process_mining_data(self, mining_time: float, nonce: int, block_index: int) -> None:
        """
        Process mining data for self-replication
        
        Args:
            mining_time: Time taken to mine the block
            nonce: Nonce used for mining
            block_index: Index of the mined block
        """
        # Get operational data from mining oracle
        operational_hash = self.pow.get_operational_data()
        
        if operational_hash:
            # Create operational data
            mining_data = {
                "block_index": block_index,
                "mining_time": mining_time,
                "nonce": nonce,
                "difficulty": self.blockchain.difficulty,
                "operational_hash": operational_hash,
                "timestamp": time.time()
            }
            
            # Process with self-replication engine
            result = await self.replication_engine.process_operational_data("mining", mining_data)
            logger.info(f"Processed mining data: {result['assets_generated']} assets, {result['replications_created']} replications")
    
    async def process_data_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data input for self-replication
        
        Args:
            data: Input data dictionary
            
        Returns:
            Processing result dictionary
        """
        with self.lock:
            result = await self.replication_engine.process_data_input(data)
            self.metrics["data_inputs_processed"] += 1
            self.metrics["assets_generated"] += result["assets_generated"]
            logger.info(f"Processed data input: {result['assets_generated']} assets, {result['replications_created']} replications")
            
            # Store the hash in a transaction
            self._create_data_hash_transaction(result["hash"])
            
            return result
    
    def _create_data_hash_transaction(self, data_hash: str) -> None:
        """
        Create a transaction to store a data hash on the blockchain
        
        Args:
            data_hash: Hash to store
        """
        # This would typically use a system wallet
        # For simplicity, we'll just create a transaction if we have a wallet
        if len(self.wallets) > 0:
            wallet_name = list(self.wallets.keys())[0]
            data = {"type": "data_hash", "hash": data_hash}
            self.create_transaction(wallet_name, wallet_name, 0.0, 0.001, data)
    
    async def process_system_metrics(self) -> Dict[str, Any]:
        """
        Process system metrics for self-replication
        
        Returns:
            Processing result dictionary
        """
        with self.lock:
            # Collect system metrics
            uptime = time.time() - self.metrics["start_time"]
            system_metrics = {
                "uptime": uptime,
                "blocks_mined": self.metrics["blocks_mined"],
                "transactions_processed": self.metrics["transactions_processed"],
                "data_inputs_processed": self.metrics["data_inputs_processed"],
                "assets_generated": self.metrics["assets_generated"],
                "memory_usage": self._get_memory_usage(),
                "blockchain_length": len(self.blockchain.chain),
                "pending_transactions": len(self.blockchain.pending_transactions)
            }
            
            # Process with self-replication engine
            result = await self.replication_engine.process_system_metrics(system_metrics)
            logger.info(f"Processed system metrics: {result['assets_generated']} assets")
            
            return result
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage metrics
        
        Returns:
            Memory usage dictionary
        """
        # In a real implementation, this would get actual memory usage
        # Here we'll just return a placeholder
        return {
            "blockchain_size": len(str(self.blockchain.to_dict())),
            "transaction_pool_size": len(str(self.blockchain.pending_transactions)),
            "replication_data_size": 0  # Placeholder
        }
    
    def get_assets(self, asset_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get generated assets
        
        Args:
            asset_type: Type of assets to get (None for all)
            
        Returns:
            List of assets
        """
        with self.lock:
            if asset_type:
                return self.replication_engine.get_assets_by_type(asset_type)
            else:
                # Combine all asset types
                assets = []
                for asset_type in ["image", "audio", "text"]:
                    assets.extend(self.replication_engine.get_assets_by_type(asset_type))
                return assets
    
    def get_blockchain_info(self) -> Dict[str, Any]:
        """
        Get blockchain information
        
        Returns:
            Blockchain info dictionary
        """
        with self.lock:
            last_block = self.blockchain.last_block
            
            return {
                "length": len(self.blockchain.chain),
                "difficulty": self.blockchain.difficulty,
                "mining_reward": self.blockchain.mining_reward,
                "last_block": {
                    "index": last_block.index,
                    "hash": last_block.hash,
                    "timestamp": last_block.timestamp,
                    "transactions": len(last_block.transactions)
                },
                "pending_transactions": len(self.blockchain.pending_transactions)
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get system metrics
        
        Returns:
            System metrics dictionary
        """
        with self.lock:
            metrics = self.metrics.copy()
            
            # Add component metrics
            metrics["replication"] = self.replication_engine.get_metrics()
            metrics["mining"] = self.pow.oracle.get_efficiency_stats()
            
            # Add uptime
            metrics["uptime"] = time.time() - metrics["start_time"]
            
            return metrics


# Example usage
async def example_usage():
    # Create GenesisChain instance
    genesis_chain = GenesisChain()
    
    # Start the system
    genesis_chain.start()
    
    # Create wallets
    alice = genesis_chain.create_wallet("alice")
    bob = genesis_chain.create_wallet("bob")
    miner = genesis_chain.create_wallet("miner")
    
    print(f"Alice's address: {alice.get_address()}")
    print(f"Bob's address: {bob.get_address()}")
    print(f"Miner's address: {miner.get_address()}")
    
    # Mine a block to get some coins
    block = genesis_chain.mine_block("miner")
    print(f"Mined block: {block.index}")
    
    # Create a transaction
    miner_address = miner.get_address()
    alice_address = alice.get_address()
    tx = genesis_chain.create_transaction("miner", alice_address, 0.5)
    print(f"Transaction created: {tx['transaction_id']}")
    
    # Mine another block to confirm the transaction
    block = genesis_chain.mine_block("miner")
    print(f"Mined block: {block.index}")
    
    # Check balances
    miner_balance = genesis_chain.blockchain.get_balance(miner_address)
    alice_balance = genesis_chain.blockchain.get_balance(alice_address)
    print(f"Miner balance: {miner_balance}")
    print(f"Alice balance: {alice_balance}")
    
    # Process a data input
    data = {
        "content": "Test data for self-replication",
        "timestamp": time.time(),
        "metadata": {
            "source": "test",
            "priority": "high"
        }
    }
    
    result = await genesis_chain.process_data_input(data)
    print(f"Processed data input: {result}")
    
    # Process system metrics
    metrics_result = await genesis_chain.process_system_metrics()
    print(f"Processed system metrics: {metrics_result}")
    
    # Get system metrics
    metrics = genesis_chain.get_metrics()
    print(f"System metrics: {metrics}")
    
    # Stop the system
    genesis_chain.stop()


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage())