import asyncio
from quantum_blockchain.core.blockchain import QuantumBlockchain
from quantum_blockchain.cryptography.wallet import Wallet

async def test():
    # Create a blockchain with lower difficulty for testing
    blockchain = QuantumBlockchain(difficulty=2)
    print(f"Created blockchain with difficulty {blockchain.difficulty}")
    
    # Create a wallet
    wallet = Wallet(name='miner')
    miner_address = wallet.get_address()
    print(f"Created wallet with address: {miner_address}")
    
    # Mine a block (create genesis block)
    print("Mining a block...")
    block = blockchain.mine_block(miner_address)
    print(f"Mined block with index {block.index} and hash {block.hash}")
    
    # Check the balance
    balance = blockchain.get_balance(miner_address)
    print(f"Miner balance: {balance}")
    
    # Create a transaction
    transaction = {
        "sender": miner_address,
        "recipient": "recipient-address",
        "amount": 0.5,
        "timestamp": 1234567890,
        "signature": "test-signature",
        "transaction_id": "test-transaction"
    }
    
    # Add the transaction to the blockchain
    blockchain.add_transaction(transaction, verify=False)
    print("Added transaction to blockchain")
    
    # Mine another block to include the transaction
    print("Mining another block...")
    block = blockchain.mine_block(miner_address)
    print(f"Mined block with index {block.index} and hash {block.hash}")
    
    # Check the balance again
    balance = blockchain.get_balance(miner_address)
    print(f"Miner balance after transaction: {balance}")
    
    # Verify that the blockchain is valid
    is_valid = blockchain.is_chain_valid()
    print(f"Blockchain is valid: {is_valid}")

if __name__ == "__main__":
    asyncio.run(test())