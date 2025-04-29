import { useState, useEffect } from "react";
import "./App.css";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

function App() {
  const [blocks, setBlocks] = useState([]);
  const [transactions, setTransactions] = useState([]);
  const [dataInputs, setDataInputs] = useState([]);
  const [newTransaction, setNewTransaction] = useState({
    sender: "",
    recipient: "",
    amount: 0
  });
  const [newData, setNewData] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("blockchain");
  const [notification, setNotification] = useState(null);

  useEffect(() => {
    fetchBlockchain();
    fetchTransactions();
    fetchDataInputs();
  }, []);

  const showNotification = (message, type = "success") => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 3000);
  };

  const fetchBlockchain = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/chain`);
      const data = await response.json();
      setBlocks(data.chain);
    } catch (error) {
      console.error("Error fetching blockchain:", error);
      showNotification("Failed to fetch blockchain data", "error");
    }
  };

  const fetchTransactions = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/transactions`);
      const data = await response.json();
      setTransactions(data.transactions || []);
    } catch (error) {
      console.error("Error fetching transactions:", error);
      showNotification("Failed to fetch transaction data", "error");
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setNewTransaction({
      ...newTransaction,
      [name]: name === "amount" ? parseFloat(value) : value
    });
  };

  const handleDataInputChange = (e) => {
    setNewData(e.target.value);
  };

  const handleTransactionSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      const response = await fetch(`${BACKEND_URL}/api/transactions/new`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(newTransaction)
      });

      if (response.ok) {
        setNewTransaction({ sender: "", recipient: "", amount: 0 });
        await fetchTransactions();
        showNotification("Transaction submitted successfully");
      } else {
        const errorData = await response.json();
        showNotification(`Error: ${errorData.detail || "Failed to submit transaction"}`, "error");
      }
    } catch (error) {
      console.error("Error submitting transaction:", error);
      showNotification("Failed to submit transaction", "error");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDataSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      // Create a structured data object from the input
      const dataObject = {
        content: newData,
        timestamp: new Date().toISOString()
      };

      const response = await fetch(`${BACKEND_URL}/api/data-input`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(dataObject)
      });

      if (response.ok) {
        const result = await response.json();
        setNewData("");
        // Display more detailed notification with generated content info
        showNotification(
          `Data processed successfully! Generated ${result.generated_content.images} images, ${result.generated_content.audio} audio tracks, and created ${result.replications} self-replications.`
        );
        
        // Refresh data after a short delay to show new entries
        setTimeout(() => {
          fetchDataInputs();
        }, 1000);
      } else {
        const errorData = await response.json();
        showNotification(`Error: ${errorData.detail || "Failed to process data"}`, "error");
      }
    } catch (error) {
      console.error("Error submitting data:", error);
      showNotification("Failed to process data", "error");
    } finally {
      setIsLoading(false);
    }
  };
  
  // Fetch data inputs for the data tab
  const fetchDataInputs = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/data-inputs`);
      if (response.ok) {
        const data = await response.json();
        setDataInputs(data.data_inputs || []);
      }
    } catch (error) {
      console.error("Error fetching data inputs:", error);
    }
  };

  const handleMining = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/mine`);
      const data = await response.json();
      await Promise.all([fetchBlockchain(), fetchTransactions()]);
      showNotification(`New block mined! Block #${data.index}`);
    } catch (error) {
      console.error("Error mining block:", error);
      showNotification("Failed to mine block", "error");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      {notification && (
        <div className={`notification ${notification.type}`}>
          {notification.message}
        </div>
      )}
      
      <header className="header">
        <div className="logo-container">
          <h1>GenesisChain</h1>
          <p className="tagline">A Self-Replicating, Value-Generating Blockchain</p>
        </div>
      </header>

      <nav className="tabs">
        <button 
          className={activeTab === "blockchain" ? "active" : ""} 
          onClick={() => setActiveTab("blockchain")}
        >
          Blockchain
        </button>
        <button 
          className={activeTab === "transactions" ? "active" : ""} 
          onClick={() => setActiveTab("transactions")}
        >
          Transactions
        </button>
        <button 
          className={activeTab === "data-input" ? "active" : ""} 
          onClick={() => setActiveTab("data-input")}
        >
          Data Input
        </button>
        <button 
          className={activeTab === "mining" ? "active" : ""} 
          onClick={() => setActiveTab("mining")}
        >
          Mining
        </button>
      </nav>

      <main className="content">
        {activeTab === "blockchain" && (
          <div className="blockchain-container">
            <h2>Blockchain Explorer</h2>
            <div className="blocks-list">
              {blocks.map((block) => (
                <div key={block.block_id} className="block-card">
                  <div className="block-header">
                    <h3>Block #{block.index}</h3>
                    <span className="timestamp">
                      {new Date(block.timestamp * 1000).toLocaleString()}
                    </span>
                  </div>
                  <div className="block-details">
                    <p>
                      <strong>Hash:</strong>{" "}
                      <span className="hash">{block.previous_hash}</span>
                    </p>
                    <p>
                      <strong>Proof:</strong> {block.proof}
                    </p>
                    <p>
                      <strong>Transactions:</strong>{" "}
                      {block.transactions.length || 0}
                    </p>
                  </div>
                </div>
              ))}
              {blocks.length === 0 && (
                <div className="empty-state">
                  <p>No blocks found. Start by mining the first block!</p>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === "transactions" && (
          <div className="transactions-container">
            <h2>Transactions</h2>
            <form onSubmit={handleTransactionSubmit} className="transaction-form">
              <div className="form-group">
                <label>Sender</label>
                <input
                  type="text"
                  name="sender"
                  value={newTransaction.sender}
                  onChange={handleInputChange}
                  required
                  placeholder="Sender address"
                />
              </div>
              <div className="form-group">
                <label>Recipient</label>
                <input
                  type="text"
                  name="recipient"
                  value={newTransaction.recipient}
                  onChange={handleInputChange}
                  required
                  placeholder="Recipient address"
                />
              </div>
              <div className="form-group">
                <label>Amount</label>
                <input
                  type="number"
                  name="amount"
                  step="0.01"
                  value={newTransaction.amount}
                  onChange={handleInputChange}
                  required
                  placeholder="Amount"
                />
              </div>
              <button type="submit" disabled={isLoading} className="submit-button">
                {isLoading ? "Processing..." : "Submit Transaction"}
              </button>
            </form>

            <div className="transactions-list">
              <h3>Recent Transactions</h3>
              {transactions.length > 0 ? (
                transactions.map((tx) => (
                  <div key={tx.transaction_id} className="transaction-card">
                    <div className="transaction-details">
                      <p>
                        <strong>From:</strong> <span className="address">{tx.sender}</span>
                      </p>
                      <p>
                        <strong>To:</strong> <span className="address">{tx.recipient}</span>
                      </p>
                      <p>
                        <strong>Amount:</strong> <span className="amount">{tx.amount}</span>
                      </p>
                      <p className="transaction-timestamp">
                        {new Date(tx.timestamp * 1000).toLocaleString()}
                      </p>
                    </div>
                    <div className="transaction-status">
                      {tx.confirmed ? (
                        <span className="confirmed">Confirmed</span>
                      ) : (
                        <span className="pending">Pending</span>
                      )}
                    </div>
                  </div>
                ))
              ) : (
                <div className="empty-state">
                  <p>No transactions found. Create a new transaction!</p>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === "data-input" && (
          <div className="data-input-container">
            <h2>Data Input</h2>
            <p className="description">
              Submit data to GenesisChain to be processed, hashed, and trigger the self-replication mechanism.
            </p>
            
            <form onSubmit={handleDataSubmit} className="data-form">
              <div className="form-group">
                <label>Data Content</label>
                <textarea
                  name="data"
                  value={newData}
                  onChange={handleDataInputChange}
                  required
                  placeholder="Enter any data (text, JSON, etc.)"
                  rows={5}
                />
              </div>
              <button type="submit" disabled={isLoading} className="submit-button">
                {isLoading ? "Processing..." : "Process Data"}
              </button>
            </form>
            
            <div className="info-box">
              <h3>How It Works</h3>
              <p>
                <strong>1.</strong> Your input data is hashed using SHA-256 algorithm
              </p>
              <p>
                <strong>2.</strong> The hash is used to generate unique digital assets
              </p>
              <p>
                <strong>3.</strong> These assets become part of the GenesisChain ecosystem
              </p>
              <p>
                <strong>4.</strong> The process self-replicates, creating new hashes and assets
              </p>
            </div>
            
            <div className="data-inputs-section">
              <h3>Recent Data Inputs & Generated Content</h3>
              <div className="data-inputs-list">
                {dataInputs.length > 0 ? (
                  dataInputs.map((data) => (
                    <div key={data.data_id} className="data-input-card">
                      <div className="data-input-header">
                        <h4>{data.parent_data_id ? "Self-Replicated Data" : "User Input Data"}</h4>
                        <span className="timestamp">
                          {new Date(data.timestamp * 1000).toLocaleString()}
                        </span>
                      </div>
                      
                      <div className="data-input-details">
                        <p>
                          <strong>Hash:</strong>{" "}
                          <span className="hash">{data.hash.substring(0, 16)}...</span>
                        </p>
                        {data.original_data && (
                          <p>
                            <strong>Content:</strong>{" "}
                            <span className="data-content">
                              {data.original_data.content ? 
                                data.original_data.content.substring(0, 40) + (data.original_data.content.length > 40 ? "..." : "") :
                                JSON.stringify(data.original_data).substring(0, 40) + "..."}
                            </span>
                          </p>
                        )}
                        
                        {data.generated_content && (
                          <div className="generated-content">
                            <p><strong>Generated Content:</strong></p>
                            <div className="content-stats">
                              <div className="stat">
                                <span className="stat-value">{data.generated_content.images.length}</span>
                                <span className="stat-label">Images</span>
                              </div>
                              <div className="stat">
                                <span className="stat-value">{data.generated_content.audio.length}</span>
                                <span className="stat-label">Audio</span>
                              </div>
                            </div>
                          </div>
                        )}
                        
                        {data.parent_data_id && (
                          <p className="replication-info">
                            <strong>Parent:</strong>{" "}
                            <span className="parent-id">{data.parent_data_id.substring(0, 8)}...</span>
                          </p>
                        )}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="empty-state">
                    <p>No data inputs found. Submit data to start the self-replication process!</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === "mining" && (
          <div className="mining-container">
            <h2>Mining</h2>
            <p className="description">
              Mining adds a new block to the chain, validates pending transactions, and generates rewards.
            </p>
            
            <div className="mining-controls">
              <button 
                onClick={handleMining} 
                disabled={isLoading}
                className="mining-button"
              >
                {isLoading ? "Mining in progress..." : "Mine New Block"}
              </button>
            </div>
            
            <div className="info-box quantum-info">
              <h3>Quantum-Enhanced Mining</h3>
              <p>
                GenesisChain uses a quantum-inspired AI Oracle to optimize mining efficiency:
              </p>
              <div className="quantum-features">
                <div className="feature">
                  <div className="feature-icon">ð§ </div>
                  <div className="feature-text">
                    <h4>AI Oracle</h4>
                    <p>Pre-screens potential proofs before hashing</p>
                  </div>
                </div>
                <div className="feature">
                  <div className="feature-icon">â¡</div>
                  <div className="feature-text">
                    <h4>Energy Efficient</h4>
                    <p>Reduces wasted hash operations by ~40-60%</p>
                  </div>
                </div>
                <div className="feature">
                  <div className="feature-icon">ð</div>
                  <div className="feature-text">
                    <h4>Self-Learning</h4>
                    <p>Continuously improves through mining data</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="info-box">
              <h3>About Mining</h3>
              <p>
                <strong>Process:</strong> Mining solves a cryptographic puzzle (Proof of Work)
              </p>
              <p>
                <strong>Purpose:</strong> Adds transaction records to GenesisChain's public ledger
              </p>
              <p>
                <strong>Reward:</strong> Miners receive newly created coins and transaction fees
              </p>
              <p>
                <strong>Difficulty:</strong> Automatically adjusts to maintain consistent block times
              </p>
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>GenesisChain - A Self-Replicating Blockchain Prototype &copy; 2025</p>
      </footer>
    </div>
  );
}

export default App;
