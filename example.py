"""
GenesisChain Example Script

This script demonstrates the key features of the GenesisChain blockchain.
"""

import asyncio
import time
import logging
import json
from typing import Dict, Any

# Import the GenesisChain main module
from quantum_blockchain.main import GenesisChain


async def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("GenesisChainExample")
    
    logger.info("Starting GenesisChain example...")
    
    # Create configuration
    config = {
        "blockchain": {
            "difficulty": 3,  # Lower difficulty for faster mining in the example
            "mining_reward": 1.0
        },
        "pow": {
            "target_block_time": 10.0,  # Faster blocks for the example
            "adjustment_period": 5
        },
        "self_replication": {
            "max_level": 2,  # Limit replication depth for the example
            "replication_probability": 0.5
        },
        "quantum": {
            "qubits_available": 8,
            "simulation_level": "medium"
        }
    }
    
    # Create and start GenesisChain
    genesis_chain = GenesisChain(config)
    genesis_chain.start()
    
    try:
        # Step 1: Create wallets
        logger.info("Creating wallets...")
        alice = genesis_chain.create_wallet("alice")
        bob = genesis_chain.create_wallet("bob")
        miner = genesis_chain.create_wallet("miner")
        
        alice_address = alice.get_address()
        bob_address = bob.get_address()
        miner_address = miner.get_address()
        
        logger.info(f"Alice's address: {alice_address}")
        logger.info(f"Bob's address: {bob_address}")
        logger.info(f"Miner's address: {miner_address}")
        
        # Step 2: Mine initial blocks to get some coins
        logger.info("Mining initial blocks...")
        for i in range(3):
            block = genesis_chain.mine_block("miner")
            if block:
                logger.info(f"Mined block {block.index} with {len(block.transactions)} transactions")
            else:
                logger.info("No block was mined (possibly no pending transactions)")
        
        # Check miner balance
        miner_balance = genesis_chain.blockchain.get_balance(miner_address)
        logger.info(f"Miner balance after initial mining: {miner_balance}")
        
        # Step 3: Create transactions
        logger.info("Creating transactions...")
        # Transfer from miner to Alice
        tx1 = genesis_chain.create_transaction("miner", alice_address, 1.0)
        if tx1:
            logger.info(f"Created transaction: {tx1['transaction_id']}")
        
        # Transfer from miner to Bob
        tx2 = genesis_chain.create_transaction("miner", bob_address, 0.5)
        if tx2:
            logger.info(f"Created transaction: {tx2['transaction_id']}")
        
        # Mine a block to confirm transactions
        logger.info("Mining block to confirm transactions...")
        block = genesis_chain.mine_block("miner")
        if block:
            logger.info(f"Mined block {block.index} with {len(block.transactions)} transactions")
        else:
            logger.info("No block was mined (possibly no pending transactions)")
        
        # Check balances
        alice_balance = genesis_chain.blockchain.get_balance(alice_address)
        bob_balance = genesis_chain.blockchain.get_balance(bob_address)
        miner_balance = genesis_chain.blockchain.get_balance(miner_address)
        
        logger.info(f"Alice balance: {alice_balance}")
        logger.info(f"Bob balance: {bob_balance}")
        logger.info(f"Miner balance: {miner_balance}")
        
        # Step 4: Process data inputs for self-replication
        logger.info("Processing data inputs for self-replication...")
        data_inputs = [
            {
                "content": "The first data input for testing GenesisChain",
                "timestamp": time.time(),
                "tags": ["test", "genesis", "first"]
            },
            {
                "content": "Quantum-resistant security is a critical feature",
                "timestamp": time.time(),
                "tags": ["quantum", "security", "cryptography"]
            },
            {
                "content": "Self-replication creates exponential value growth",
                "timestamp": time.time(),
                "tags": ["self-replication", "growth", "value"]
            }
        ]
        
        for i, data in enumerate(data_inputs):
            logger.info(f"Processing data input {i+1}...")
            result = await genesis_chain.process_data_input(data)
            logger.info(f"Data input {i+1} processed: generated {result['assets_generated']} assets, {result['replications_created']} replications")
        
        # Step 5: Process system metrics
        logger.info("Processing system metrics...")
        metrics_result = await genesis_chain.process_system_metrics()
        logger.info(f"System metrics processed: generated {metrics_result['assets_generated']} assets")
        
        # Step 6: Mine more blocks to include data hash transactions
        logger.info("Mining additional blocks...")
        for i in range(2):
            block = genesis_chain.mine_block("miner")
            logger.info(f"Mined block {block.index} with {len(block.transactions)} transactions")
        
        # Step 7: Get generated assets
        logger.info("Getting generated assets...")
        images = genesis_chain.get_assets("image")
        audio = genesis_chain.get_assets("audio")
        text = genesis_chain.get_assets("text")
        
        logger.info(f"Generated {len(images)} images, {len(audio)} audio tracks, and {len(text)} text assets")
        
        if images:
            logger.info(f"Sample image: {json.dumps(images[0], indent=2)}")
        
        if audio:
            logger.info(f"Sample audio: {json.dumps(audio[0], indent=2)}")
        
        # Step 8: Get blockchain info
        blockchain_info = genesis_chain.get_blockchain_info()
        logger.info(f"Blockchain info: {json.dumps(blockchain_info, indent=2)}")
        
        # Step 9: Get system metrics
        metrics = genesis_chain.get_metrics()
        logger.info(f"System metrics:")
        logger.info(f"  Blocks mined: {metrics['blocks_mined']}")
        logger.info(f"  Transactions processed: {metrics['transactions_processed']}")
        logger.info(f"  Data inputs processed: {metrics['data_inputs_processed']}")
        logger.info(f"  Assets generated: {metrics['assets_generated']}")
        logger.info(f"  Mining efficiency: {metrics['mining']['energy_efficiency']:.2%}")
        
        # Print self-replication metrics
        replication_metrics = metrics['replication']
        logger.info(f"Self-replication metrics:")
        logger.info(f"  Data inputs processed: {replication_metrics['data_inputs_processed']}")
        logger.info(f"  Assets generated: {replication_metrics['assets_generated']}")
        logger.info(f"  Replications created: {replication_metrics['replications_created']}")
        logger.info(f"  Max chain depth: {replication_metrics['max_chain_depth']}")
        
    finally:
        # Shutdown GenesisChain
        genesis_chain.stop()
        logger.info("GenesisChain example completed")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())