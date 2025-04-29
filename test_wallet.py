import asyncio
from quantum_blockchain.cryptography.wallet import Wallet

async def test():
    wallet = Wallet(name='test')
    print(f'Created wallet with address: {wallet.get_address()}')
    
    # Create a transaction
    transaction = wallet.create_transaction("recipient-address", 10.0)
    print(f'Created transaction: {transaction}')

if __name__ == "__main__":
    asyncio.run(test())