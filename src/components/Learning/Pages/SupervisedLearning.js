import React, { useState, useEffect } from 'react';
import { Container, Accordion, Row, Col, Tabs, Tab, Button } from 'react-bootstrap';
import Particle from '../../Particle';
import '../../../style.css';
import { Line, Scatter, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const SupervisedLearning = () => {
    const [knnK, setKnnK] = useState(3);
    const [svmKernel, setSvmKernel] = useState('linear');
    const [treeDepth, setTreeDepth] = useState(3);
    const [forestSize, setForestSize] = useState(5);
    const [pyodide, setPyodide] = useState(null);
    const [pythonOutput, setPythonOutput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [regressionResults, setRegressionResults] = useState(null);
    const [logisticResults, setLogisticResults] = useState(null);
    const [svmResults, setSvmResults] = useState(null);
    const [knnResults, setKnnResults] = useState(null);
    const [treeResults, setTreeResults] = useState(null);
    const [forestResults, setForestResults] = useState(null);
    const [naiveBayesResults, setNaiveBayesResults] = useState(null);
    const [activeTab, setActiveTab] = useState('linear');
    const [pyodideLoaded, setPyodideLoaded] = useState(false);
    const [regressionPlot, setRegressionPlot] = useState(null);
    const [activeAccordion, setActiveAccordion] = useState(null);
    const [svmPlot, setSvmPlot] = useState(null);
    const [knnPlot, setKnnPlot] = useState(null);
    const [treePlot, setTreePlot] = useState(null);
    const [forestPlot, setForestPlot] = useState(null);
    const [naiveBayesPlot, setNaiveBayesPlot] = useState(null);
    const [logisticPlot, setLogisticPlot] = useState(null);
    const [nnPlot, setNnPlot] = useState(null);
    const [nnResults, setNnResults] = useState(null);

    // Function to handle tab changes
    const handleTabChange = (k) => {
        setActiveTab(k);
        setActiveAccordion(null); // Collapse all accordion items
    };

    // Initialize Pyodide
    useEffect(() => {
        const loadPyodide = async () => {
            try {
                console.log("Starting Pyodide initialization...");
                const pyodideInstance = await window.loadPyodide({
                    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/"
                });
                console.log("Pyodide core loaded, loading packages...");
                await pyodideInstance.loadPackage(['numpy', 'scikit-learn', 'matplotlib']);
                console.log("Packages loaded successfully");
                setPyodide(pyodideInstance);
                setPyodideLoaded(true);
            } catch (error) {
                console.error("Error loading Pyodide:", error);
                setPythonOutput(`Error loading Pyodide: ${error.message}`);
            }
        };
        loadPyodide();
    }, []);

    // Function to run Python code
    const runPythonCode = async (code, setResults, setPlot) => {
        if (!pyodideLoaded) {
            setPythonOutput("Error: Pyodide is still loading. Please wait a moment and try again.");
            return;
        }
        setIsLoading(true);
        setPythonOutput('Running code...');
        try {
            // Clear previous output
            setPythonOutput('');
            
            // Run the Python code
            const result = await pyodide.runPythonAsync(code);
            
            // Parse the result if it's a string containing JSON
            try {
                const parsedResult = JSON.parse(result);
                setResults(parsedResult);
                if (parsedResult.plot) {
                    setPlot(`data:image/png;base64,${parsedResult.plot}`);
                }
                setPythonOutput('Code executed successfully!');
            } catch {
                setPythonOutput(result);
            }
        } catch (error) {
            setPythonOutput(`Error: ${error.message}`);
        }
        setIsLoading(false);
    };

    // Sample Python code for linear regression
    const linearRegressionCode = `
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io
import base64

# Set the backend to Agg for non-interactive mode
plt.switch_backend('Agg')

# Set custom style
plt.style.use('dark_background')

# Generate sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#2e261a')
ax.set_facecolor('#2e261a')

# Plot data points and regression line (all #759e6d)
ax.scatter(X, y, color='#759e6d', label='Actual Data', s=100, alpha=0.8)
ax.plot(X, y_pred, color='#759e6d', label='Linear Regression', linewidth=2)

# Customize the plot
ax.set_xlabel('X', fontsize=12, color='#759e6d')
ax.set_ylabel('y', fontsize=12, color='#759e6d')
ax.set_title('Linear Regression Results', fontsize=14, pad=20, color='#759e6d')

# Customize grid
ax.grid(True, alpha=0.3, color='#759e6d')
ax.set_axisbelow(True)

# Customize spines
for spine in ax.spines.values():
    spine.set_color('#759e6d')

# Customize ticks
ax.tick_params(colors='#759e6d')

# Customize legend
legend = ax.legend(frameon=True, facecolor='#2e261a', edgecolor='#759e6d')
plt.setp(legend.get_texts(), color='#759e6d')

# Adjust layout
plt.tight_layout()

# Save plot to a bytes buffer
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#2e261a')
buf.seek(0)
plt.close()

# Convert to base64
img_str = base64.b64encode(buf.read()).decode('utf-8')

# Return results as JSON string
import json
results = {
    "coefficients": model.coef_[0],
    "intercept": model.intercept_,
    "mse": float(mse),
    "r2": float(r2),
    "plot": img_str
}
json.dumps(results)
`;

    // Sample Python code for logistic regression
    const logisticRegressionCode = `
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import json

# Generate sample data
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])

# Create and train the model
model = LogisticRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y, y_pred)

# Prepare results
results = {
    "coefficients": model.coef_[0][0],
    "intercept": model.intercept_[0],
    "accuracy": float(accuracy),
    "predictions": y_pred.tolist(),
    "probabilities": y_prob.tolist(),
    "actual": y.tolist()
}

# Return results as JSON string
json.dumps(results)
`;

    // Sample Python code for SVM
    const svmCode = `
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import io
import base64

# Set the backend to Agg for non-interactive mode
plt.switch_backend('Agg')

# Set custom style
plt.style.use('dark_background')

# Generate sample data
X = np.array([[1, 2], [2, 3], [3, 2], [4, 5], [5, 4], [6, 5]])
y = np.array([0, 0, 0, 1, 1, 1])

# Create and train the model
model = SVC(kernel='${svmKernel}', probability=True)
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y, y_pred)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#2e261a')
ax.set_facecolor('#2e261a')

# Plot data points
for i in range(len(X)):
    color = '#759e6d' if y[i] == 0 else '#c5a861'
    ax.scatter(X[i, 0], X[i, 1], color=color, s=100, alpha=0.8, 
              label=f'Class {y[i]}' if i < 2 else "")

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contour(xx, yy, Z, colors='#c5a861', alpha=0.3)

# Customize the plot
ax.set_xlabel('Feature 1', fontsize=12, color='#c5a861')
ax.set_ylabel('Feature 2', fontsize=12, color='#c5a861')
ax.set_title('SVM Classification Results', fontsize=14, pad=20, color='#759e6d')

# Customize grid
ax.grid(True, alpha=0.3, color='#c5a861')
ax.set_axisbelow(True)

# Customize spines
for spine in ax.spines.values():
    spine.set_color('#c5a861')

# Customize ticks
ax.tick_params(colors='#c5a861')

# Customize legend
legend = ax.legend(frameon=True, facecolor='#2e261a', edgecolor='#c5a861')
plt.setp(legend.get_texts(), color='#c5a861')

# Adjust layout
plt.tight_layout()

# Save plot to a bytes buffer
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#2e261a')
buf.seek(0)
plt.close()

# Convert to base64
img_str = base64.b64encode(buf.read()).decode('utf-8')

# Return results as JSON string
import json
results = {
    "accuracy": float(accuracy),
    "predictions": y_pred.tolist(),
    "probabilities": y_prob.tolist(),
    "actual": y.tolist(),
    "support_vectors": model.support_vectors_.tolist(),
    "plot": img_str
}
json.dumps(results)
`;

    // Sample Python code for KNN
    const knnCode = `
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import io
import base64

# Set the backend to Agg for non-interactive mode
plt.switch_backend('Agg')

# Set custom style
plt.style.use('dark_background')

# Generate sample data with 50 points
np.random.seed(42)
X = np.random.randn(50, 2) * 2
# Create two distinct clusters
X[:25, 0] += 3
X[:25, 1] += 3
y = np.zeros(50)
y[25:] = 1

# Create and train the model
model = KNeighborsClassifier(n_neighbors=${knnK})
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y, y_pred)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#2e261a')
ax.set_facecolor('#2e261a')

# Plot data points
for i in range(len(X)):
    color = '#759e6d' if y[i] == 0 else '#c5a861'
    ax.scatter(X[i, 0], X[i, 1], color=color, s=100, alpha=0.8,
              label=f'Class {y[i]}' if i < 2 else "")

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contour(xx, yy, Z, colors='#c5a861', alpha=0.3)

# Customize the plot
ax.set_xlabel('Feature 1', fontsize=12, color='#c5a861')
ax.set_ylabel('Feature 2', fontsize=12, color='#c5a861')
ax.set_title(f'KNN Classification (k=${knnK})', fontsize=14, pad=20, color='#759e6d')

# Customize grid
ax.grid(True, alpha=0.3, color='#c5a861')
ax.set_axisbelow(True)

# Customize spines
for spine in ax.spines.values():
    spine.set_color('#c5a861')

# Customize ticks
ax.tick_params(colors='#c5a861')

# Customize legend
legend = ax.legend(frameon=True, facecolor='#2e261a', edgecolor='#c5a861')
plt.setp(legend.get_texts(), color='#c5a861')

# Adjust layout
plt.tight_layout()

# Save plot to a bytes buffer
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#2e261a')
buf.seek(0)
plt.close()

# Convert to base64
img_str = base64.b64encode(buf.read()).decode('utf-8')

# Return results as JSON string
import json
results = {
    "accuracy": float(accuracy),
    "predictions": y_pred.tolist(),
    "probabilities": y_prob.tolist(),
    "actual": y.tolist(),
    "X": X.tolist(),
    "plot": img_str
}
json.dumps(results)
`;

    // Sample Python code for Decision Tree
    const treeCode = `
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import io
import base64

# Set the backend to Agg for non-interactive mode
plt.switch_backend('Agg')

# Set custom style
plt.style.use('dark_background')

# Generate sample data
X = np.array([[1, 2, 3, 4, 5],
              [2, 3, 4, 5, 6],
              [3, 4, 5, 6, 7],
              [4, 5, 6, 7, 8],
              [5, 6, 7, 8, 9],
              [6, 7, 8, 9, 10]])
y = np.array([0, 0, 0, 1, 1, 1])

# Create and train the model
model = DecisionTreeClassifier(max_depth=${treeDepth})
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y, y_pred)

# Get feature importance
feature_importance = model.feature_importances_

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#2e261a')
ax.set_facecolor('#2e261a')

# Plot feature importance
features = [f'Feature {i+1}' for i in range(len(feature_importance))]
bars = ax.bar(features, feature_importance, color='#759e6d', alpha=0.8)

# Customize the plot
ax.set_xlabel('Features', fontsize=12, color='#c5a861')
ax.set_ylabel('Importance', fontsize=12, color='#c5a861')
ax.set_title(f'Decision Tree Feature Importance (Depth=${treeDepth})', fontsize=14, pad=20, color='#759e6d')

# Customize grid
ax.grid(True, alpha=0.3, color='#c5a861')
ax.set_axisbelow(True)

# Customize spines
for spine in ax.spines.values():
    spine.set_color('#c5a861')

# Customize ticks
ax.tick_params(colors='#c5a861')
plt.xticks(rotation=45)

# Adjust layout
plt.tight_layout()

# Save plot to a bytes buffer
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#2e261a')
buf.seek(0)
plt.close()

# Convert to base64
img_str = base64.b64encode(buf.read()).decode('utf-8')

# Return results as JSON string
import json
results = {
    "accuracy": float(accuracy),
    "predictions": y_pred.tolist(),
    "probabilities": y_prob.tolist(),
    "actual": y.tolist(),
    "feature_importance": feature_importance.tolist(),
    "plot": img_str
}
json.dumps(results)
`;

    // Sample Python code for Random Forest
    const forestCode = `
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import io
import base64

# Set the backend to Agg for non-interactive mode
plt.switch_backend('Agg')

# Set custom style
plt.style.use('dark_background')

# Generate sample data
X = np.array([[1, 2, 3, 4, 5],
              [2, 3, 4, 5, 6],
              [3, 4, 5, 6, 7],
              [4, 5, 6, 7, 8],
              [5, 6, 7, 8, 9],
              [6, 7, 8, 9, 10]])
y = np.array([0, 0, 0, 1, 1, 1])

# Create and train the model
model = RandomForestClassifier(n_estimators=${forestSize}, max_depth=${treeDepth})
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y, y_pred)

# Get feature importance
feature_importance = model.feature_importances_

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#2e261a')
ax.set_facecolor('#2e261a')

# Plot feature importance
features = [f'Feature {i+1}' for i in range(len(feature_importance))]
bars = ax.bar(features, feature_importance, color='#759e6d', alpha=0.8)

# Customize the plot
ax.set_xlabel('Features', fontsize=12, color='#c5a861')
ax.set_ylabel('Importance', fontsize=12, color='#c5a861')
ax.set_title(f'Random Forest Feature Importance (Trees=${forestSize})', fontsize=14, pad=20, color='#759e6d')

# Customize grid
ax.grid(True, alpha=0.3, color='#c5a861')
ax.set_axisbelow(True)

# Customize spines
for spine in ax.spines.values():
    spine.set_color('#c5a861')

# Customize ticks
ax.tick_params(colors='#c5a861')
plt.xticks(rotation=45)

# Adjust layout
plt.tight_layout()

# Save plot to a bytes buffer
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#2e261a')
buf.seek(0)
plt.close()

# Convert to base64
img_str = base64.b64encode(buf.read()).decode('utf-8')

# Return results as JSON string
import json
results = {
    "accuracy": float(accuracy),
    "predictions": y_pred.tolist(),
    "probabilities": y_prob.tolist(),
    "actual": y.tolist(),
    "feature_importance": feature_importance.tolist(),
    "plot": img_str
}
json.dumps(results)
`;

    // Sample Python code for Naive Bayes
    const naiveBayesCode = `
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import io
import base64

# Set the backend to Agg for non-interactive mode
plt.switch_backend('Agg')

# Set custom style
plt.style.use('dark_background')

# Generate sample data
X = np.array([[1, 2], [2, 3], [3, 2], [4, 5], [5, 4], [6, 5]])
y = np.array([0, 0, 0, 1, 1, 1])

# Create and train the model
model = GaussianNB()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y, y_pred)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#2e261a')
ax.set_facecolor('#2e261a')

# Plot data points
for i in range(len(X)):
    color = '#759e6d' if y[i] == 0 else '#c5a861'
    ax.scatter(X[i, 0], X[i, 1], color=color, s=100, alpha=0.8,
              label=f'Class {y[i]}' if i < 2 else "")

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contour(xx, yy, Z, colors='#c5a861', alpha=0.3)

# Customize the plot
ax.set_xlabel('Feature 1', fontsize=12, color='#c5a861')
ax.set_ylabel('Feature 2', fontsize=12, color='#c5a861')
ax.set_title('Naive Bayes Classification', fontsize=14, pad=20, color='#759e6d')

# Customize grid
ax.grid(True, alpha=0.3, color='#c5a861')
ax.set_axisbelow(True)

# Customize spines
for spine in ax.spines.values():
    spine.set_color('#c5a861')

# Customize ticks
ax.tick_params(colors='#c5a861')

# Customize legend
legend = ax.legend(frameon=True, facecolor='#2e261a', edgecolor='#c5a861')
plt.setp(legend.get_texts(), color='#c5a861')

# Adjust layout
plt.tight_layout()

# Save plot to a bytes buffer
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#2e261a')
buf.seek(0)
plt.close()

# Convert to base64
img_str = base64.b64encode(buf.read()).decode('utf-8')

# Return results as JSON string
import json
results = {
    "accuracy": float(accuracy),
    "predictions": y_pred.tolist(),
    "probabilities": y_prob.tolist(),
    "actual": y.tolist(),
    "plot": img_str
}
json.dumps(results)
`;

    // Sample Python code for Neural Network
    const neuralNetworkCode = `
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# Set the backend to Agg for non-interactive mode
plt.switch_backend('Agg')

# Set custom style
plt.style.use('dark_background')

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Neural Network implementation
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output, learning_rate):
        m = X.shape[0]
        y = y.reshape(-1, 1)
        
        # Output layer error
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)
        
        # Hidden layer error
        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)
        
        # Update weights and biases
        self.W2 += self.a1.T.dot(output_delta) * learning_rate
        self.b2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.W1 += X.T.dot(hidden_delta) * learning_rate
        self.b1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

# Create and train the model
model = SimpleNN(input_size=2, hidden_size=4, output_size=1)
epochs = 1000
learning_rate = 0.01
losses = []

for epoch in range(epochs):
    # Forward pass
    output = model.forward(X)
    
    # Calculate loss
    loss = -np.mean(y.reshape(-1, 1) * np.log(output + 1e-8) + 
                   (1 - y.reshape(-1, 1)) * np.log(1 - output + 1e-8))
    losses.append(loss)
    
    # Backward pass
    model.backward(X, y, output, learning_rate)

# Make predictions
predictions = (model.forward(X) > 0.5).astype(int)
accuracy = np.mean(predictions == y.reshape(-1, 1))

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor='#2e261a')
fig.suptitle('Neural Network Results', fontsize=14, color='#759e6d', y=1.05)

# Plot training loss
ax1.set_facecolor('#2e261a')
ax1.plot(losses, color='#759e6d', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12, color='#c5a861')
ax1.set_ylabel('Loss', fontsize=12, color='#c5a861')
ax1.set_title('Training Loss', fontsize=12, color='#759e6d')
ax1.grid(True, alpha=0.3, color='#c5a861')
ax1.tick_params(colors='#c5a861')

# Plot decision boundary
ax2.set_facecolor('#2e261a')
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_predictions = (model.forward(grid_points) > 0.5).astype(int)
grid_predictions = grid_predictions.reshape(xx.shape);

ax2.contourf(xx, yy, grid_predictions, alpha=0.3, colors=['#759e6d', '#c5a861'])
ax2.scatter(X[y==0, 0], X[y==0, 1], color='#759e6d', label='Class 0', alpha=0.8)
ax2.scatter(X[y==1, 0], X[y==1, 1], color='#c5a861', label='Class 1', alpha=0.8)
ax2.set_xlabel('Feature 1', fontsize=12, color='#c5a861')
ax2.set_ylabel('Feature 2', fontsize=12, color='#c5a861')
ax2.set_title('Decision Boundary', fontsize=12, color='#759e6d')
ax2.grid(True, alpha=0.3, color='#c5a861')
ax2.tick_params(colors='#c5a861')
ax2.legend(frameon=True, facecolor='#2e261a', edgecolor='#c5a861')

# Customize spines
for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_color('#c5a861')

# Adjust layout
plt.tight_layout()

# Save plot to a bytes buffer
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#2e261a')
buf.seek(0)
plt.close()

# Convert to base64
img_str = base64.b64encode(buf.read()).decode('utf-8')

# Return results as JSON string
import json
results = {
    "accuracy": float(accuracy),
    "final_loss": float(losses[-1]),
    "plot": img_str
}
json.dumps(results)
`;

    // Update regression data based on Python results
    const regressionData = {
        labels: ['1', '2', '3', '4', '5'],
        datasets: [
            {
                label: 'Actual Values',
                data: regressionResults?.actual || [2, 4, 5, 4, 5],
                fill: false,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                tension: 0.1,
                pointRadius: 6,
                pointHoverRadius: 8
            },
            {
                label: 'Predicted Values',
                data: regressionResults?.predictions || [2.2, 3.4, 4.6, 5.8, 7.0],
                fill: false,
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                tension: 0.1,
                pointRadius: 6,
                pointHoverRadius: 8
            }
        ]
    };

    const regressionOptions = {
        responsive: true,
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    color: '#fff',
                    font: {
                        size: 14
                    }
                }
            },
            title: {
                display: true,
                text: 'Linear Regression Results',
                color: '#fff',
                font: {
                    size: 16
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#fff'
                }
            },
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#fff'
                }
            }
        }
    };

    // Update logistic regression data based on Python results
    const logisticData = {
        labels: ['1', '2', '3', '4', '5', '6'],
        datasets: [
            {
                label: 'Actual Class',
                data: logisticResults?.actual || [0, 0, 0, 1, 1, 1],
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                borderColor: 'rgb(75, 192, 192)',
                borderWidth: 1,
                type: 'bar'
            },
            {
                label: 'Predicted Probability',
                data: logisticResults?.probabilities || [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                borderColor: 'rgb(255, 99, 132)',
                borderWidth: 1,
                type: 'line',
                fill: false,
                tension: 0.4,
                pointRadius: 6,
                pointHoverRadius: 8
            }
        ]
    };

    const logisticOptions = {
        responsive: true,
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    color: '#fff',
                    font: {
                        size: 14
                    }
                }
            },
            title: {
                display: true,
                text: 'Logistic Regression Results',
                color: '#fff',
                font: {
                    size: 16
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                max: 1,
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#fff'
                }
            },
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#fff'
                }
            }
        }
    };

    // Decision boundary data
    const decisionBoundaryData = {
        datasets: [
            {
                label: 'Class 0',
                data: [
                    { x: 1, y: 0.1 },
                    { x: 2, y: 0.2 },
                    { x: 3, y: 0.3 }
                ],
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                borderColor: 'rgb(75, 192, 192)',
                pointRadius: 8,
                pointHoverRadius: 10
            },
            {
                label: 'Class 1',
                data: [
                    { x: 4, y: 0.7 },
                    { x: 5, y: 0.8 },
                    { x: 6, y: 0.9 }
                ],
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                borderColor: 'rgb(255, 99, 132)',
                pointRadius: 8,
                pointHoverRadius: 10
            },
            {
                label: 'Decision Boundary',
                data: [
                    { x: 3.5, y: 0 },
                    { x: 3.5, y: 1 }
                ],
                borderColor: 'rgba(255, 255, 255, 0.8)',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false,
                tension: 0
            }
        ]
    };

    const decisionBoundaryOptions = {
        responsive: true,
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    color: '#fff',
                    font: {
                        size: 14
                    }
                }
            },
            title: {
                display: true,
                text: 'Decision Boundary',
                color: '#fff',
                font: {
                    size: 16
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                max: 1,
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#fff'
                }
            },
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#fff'
                }
            }
        }
    };

    // Confusion matrix data
    const confusionMatrixData = {
        labels: ['Predicted 0', 'Predicted 1'],
        datasets: [
            {
                label: 'Actual 0',
                data: [2, 1],
                backgroundColor: [
                    'rgba(75, 192, 192, 0.5)',
                    'rgba(255, 99, 132, 0.5)'
                ],
                borderColor: [
                    'rgb(75, 192, 192)',
                    'rgb(255, 99, 132)'
                ],
                borderWidth: 1
            },
            {
                label: 'Actual 1',
                data: [0, 3],
                backgroundColor: [
                    'rgba(255, 99, 132, 0.5)',
                    'rgba(75, 192, 192, 0.5)'
                ],
                borderColor: [
                    'rgb(255, 99, 132)',
                    'rgb(75, 192, 192)'
                ],
                borderWidth: 1
            }
        ]
    };

    const confusionMatrixOptions = {
        responsive: true,
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    color: '#fff',
                    font: {
                        size: 14
                    }
                }
            },
            title: {
                display: true,
                text: 'Confusion Matrix',
                color: '#fff',
                font: {
                    size: 16
                }
            }
        },
        scales: {
            y: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#fff'
                }
            },
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#fff'
                }
            }
        }
    };

    // Sample data for classification
    const classificationData = {
        datasets: [
            {
                label: 'Class 1',
                data: [
                    { x: 1, y: 2 },
                    { x: 2, y: 3 },
                    { x: 3, y: 2 },
                ],
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
            },
            {
                label: 'Class 2',
                data: [
                    { x: 4, y: 5 },
                    { x: 5, y: 4 },
                    { x: 6, y: 5 },
                ],
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
            },
        ],
    };

    // Sample data for feature importance
    const featureImportanceData = {
        labels: ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'],
        datasets: [
            {
                label: 'Importance Score',
                data: [0.3, 0.25, 0.2, 0.15, 0.1],
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
            }
        ]
    };

    const chartOptions = {
        scales: {
            x: {
                type: 'linear',
                position: 'bottom',
            },
        },
        plugins: {
            title: {
                display: true,
                text: 'Classification Results',
            },
        },
    };

    // SVM Data and Options
    const svmData = {
        datasets: [
            {
                label: 'Class 1',
                data: [
                    { x: 1, y: 2 },
                    { x: 2, y: 3 },
                    { x: 3, y: 2 },
                ],
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                borderColor: 'rgb(255, 99, 132)',
                pointRadius: 8,
                pointHoverRadius: 10
            },
            {
                label: 'Class 2',
                data: [
                    { x: 4, y: 5 },
                    { x: 5, y: 4 },
                    { x: 6, y: 5 },
                ],
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                borderColor: 'rgb(75, 192, 192)',
                pointRadius: 8,
                pointHoverRadius: 10
            },
            {
                label: 'Decision Boundary',
                data: [
                    { x: 3.5, y: 1 },
                    { x: 3.5, y: 6 }
                ],
                borderColor: 'rgba(255, 255, 255, 0.8)',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false,
                tension: 0
            }
        ]
    };

    const svmOptions = {
        responsive: true,
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    color: '#fff',
                    font: {
                        size: 14
                    }
                }
            },
            title: {
                display: true,
                text: 'SVM Classification',
                color: '#fff',
                font: {
                    size: 16
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#fff'
                }
            },
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#fff'
                }
            }
        }
    };

    // KNN Data and Options
    const knnData = {
        datasets: [
            {
                label: 'Class 1',
                data: knnResults?.X?.filter((_, i) => knnResults.actual[i] === 0) || [
                    { x: 1, y: 2 },
                    { x: 2, y: 3 },
                    { x: 3, y: 2 },
                ],
                backgroundColor: 'rgba(255, 99, 132, 0.7)',
                borderColor: 'rgb(255, 99, 132)',
                pointRadius: 6,
                pointHoverRadius: 8,
                pointStyle: 'circle'
            },
            {
                label: 'Class 2',
                data: knnResults?.X?.filter((_, i) => knnResults.actual[i] === 1) || [
                    { x: 4, y: 5 },
                    { x: 5, y: 4 },
                    { x: 6, y: 5 },
                ],
                backgroundColor: 'rgba(75, 192, 192, 0.7)',
                borderColor: 'rgb(75, 192, 192)',
                pointRadius: 6,
                pointHoverRadius: 8,
                pointStyle: 'circle'
            },
            {
                label: 'K-Nearest Neighbors',
                data: knnResults?.X?.map((point, i) => ({
                    x: point[0],
                    y: point[1],
                    r: knnResults.predictions[i] === 0 ? 8 : 8,
                    backgroundColor: knnResults.predictions[i] === 0 ? 
                        'rgba(255, 99, 132, 0.3)' : 'rgba(75, 192, 192, 0.3)'
                })) || [],
                backgroundColor: 'rgba(255, 255, 255, 0.8)',
                borderColor: 'rgb(255, 255, 255)',
                pointRadius: 8,
                pointHoverRadius: 10,
                pointStyle: 'star'
            }
        ]
    };

    const knnOptions = {
        responsive: true,
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    color: '#fff',
                    font: {
                        size: 14
                    }
                }
            },
            title: {
                display: true,
                text: `KNN Classification (k=${knnK})`,
                color: '#fff',
                font: {
                    size: 16
                }
            },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        const point = context.raw;
                        if (point.r) {
                            return `Predicted Class: ${point.r === 8 ? 'Class 1' : 'Class 2'}`;
                        }
                        return context.dataset.label;
                    }
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#fff'
                }
            },
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#fff'
                }
            }
        }
    };

    // Decision Tree Data and Options
    const treeData = {
        labels: ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'],
        datasets: [
            {
                label: 'Feature Importance',
                data: [0.3, 0.25, 0.2, 0.15, 0.1],
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                borderColor: 'rgb(75, 192, 192)',
                borderWidth: 1
            }
        ]
    };

    const treeOptions = {
        responsive: true,
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    color: '#fff',
                    font: {
                        size: 14
                    }
                }
            },
            title: {
                display: true,
                text: `Decision Tree (Depth=${treeDepth})`,
                color: '#fff',
                font: {
                    size: 16
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                max: 1,
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#fff'
                }
            },
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#fff'
                }
            }
        }
    };

    // Random Forest Data and Options
    const forestData = {
        labels: ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'],
        datasets: [
            {
                label: 'Feature Importance',
                data: [0.35, 0.25, 0.2, 0.15, 0.05],
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                borderColor: 'rgb(255, 99, 132)',
                borderWidth: 1
            }
        ]
    };

    const forestOptions = {
        responsive: true,
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    color: '#fff',
                    font: {
                        size: 14
                    }
                }
            },
            title: {
                display: true,
                text: `Random Forest (Trees=${forestSize})`,
                color: '#fff',
                font: {
                    size: 16
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                max: 1,
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#fff'
                }
            },
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#fff'
                }
            }
        }
    };

    // Naive Bayes Data and Options
    const naiveBayesData = {
        labels: ['Class 1', 'Class 2'],
        datasets: [
            {
                label: 'Predicted Probabilities',
                data: [0.7, 0.3],
                backgroundColor: [
                    'rgba(75, 192, 192, 0.5)',
                    'rgba(255, 99, 132, 0.5)'
                ],
                borderColor: [
                    'rgb(75, 192, 192)',
                    'rgb(255, 99, 132)'
                ],
                borderWidth: 1
            }
        ]
    };

    const naiveBayesOptions = {
        responsive: true,
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    color: '#fff',
                    font: {
                        size: 14
                    }
                }
            },
            title: {
                display: true,
                text: 'Naive Bayes Classification',
                color: '#fff',
                font: {
                    size: 16
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                max: 1,
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#fff'
                }
            },
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#fff'
                }
            }
        }
    };

    // SVM Decision Boundary Visualization
    const svmDecisionBoundaryData = {
        datasets: [
            {
                label: 'Class 1',
                data: svmResults?.support_vectors?.filter((_, i) => svmResults.actual[i] === 0) || [
                    { x: 1, y: 2 },
                    { x: 2, y: 3 },
                    { x: 3, y: 2 },
                ],
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                borderColor: 'rgb(255, 99, 132)',
                pointRadius: 8,
                pointHoverRadius: 10
            },
            {
                label: 'Class 2',
                data: svmResults?.support_vectors?.filter((_, i) => svmResults.actual[i] === 1) || [
                    { x: 4, y: 5 },
                    { x: 5, y: 4 },
                    { x: 6, y: 5 },
                ],
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                borderColor: 'rgb(75, 192, 192)',
                pointRadius: 8,
                pointHoverRadius: 10
            },
            {
                label: 'Decision Boundary',
                data: [
                    { x: 3.5, y: 0 },
                    { x: 3.5, y: 6 }
                ],
                borderColor: 'rgba(255, 255, 255, 0.8)',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false,
                tension: 0
            }
        ]
    };

    // KNN Neighbor Visualization
    const knnNeighborsData = {
        datasets: [
            {
                label: 'Class 1',
                data: knnResults?.actual?.map((label, i) => ({
                    x: i + 1,
                    y: label === 0 ? 1 : 0
                })) || [
                    { x: 1, y: 1 },
                    { x: 2, y: 1 },
                    { x: 3, y: 1 }
                ],
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                borderColor: 'rgb(255, 99, 132)',
                pointRadius: 8,
                pointHoverRadius: 10
            },
            {
                label: 'Class 2',
                data: knnResults?.actual?.map((label, i) => ({
                    x: i + 1,
                    y: label === 1 ? 1 : 0
                })) || [
                    { x: 4, y: 1 },
                    { x: 5, y: 1 },
                    { x: 6, y: 1 }
                ],
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                borderColor: 'rgb(75, 192, 192)',
                pointRadius: 8,
                pointHoverRadius: 10
            },
            {
                label: 'K-Nearest Neighbors',
                data: knnResults?.predictions?.map((pred, i) => ({
                    x: i + 1,
                    y: pred
                })) || [
                    { x: 3.5, y: 0.5 }
                ],
                backgroundColor: 'rgba(255, 255, 255, 0.8)',
                borderColor: 'rgb(255, 255, 255)',
                pointRadius: 10,
                pointHoverRadius: 12,
                pointStyle: 'star'
            }
        ]
    };

    // Decision Tree Feature Importance
    const treeFeatureImportanceData = {
        labels: ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'],
        datasets: [
            {
                label: 'Feature Importance',
                data: treeResults?.feature_importance || [0.3, 0.25, 0.2, 0.15, 0.1],
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                borderColor: 'rgb(75, 192, 192)',
                borderWidth: 1
            }
        ]
    };

    // Random Forest Feature Importance
    const forestFeatureImportanceData = {
        labels: ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'],
        datasets: [
            {
                label: 'Feature Importance',
                data: forestResults?.feature_importance || [0.35, 0.25, 0.2, 0.15, 0.05],
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                borderColor: 'rgb(255, 99, 132)',
                borderWidth: 1
            }
        ]
    };

    // Naive Bayes Class Probabilities
    const naiveBayesProbabilitiesData = {
        labels: ['Class 1', 'Class 2'],
        datasets: [
            {
                label: 'Class Probabilities',
                data: naiveBayesResults?.probabilities?.map(p => [1 - p, p]).flat() || [0.7, 0.3],
                backgroundColor: [
                    'rgba(75, 192, 192, 0.5)',
                    'rgba(255, 99, 132, 0.5)'
                ],
                borderColor: [
                    'rgb(75, 192, 192)',
                    'rgb(255, 99, 132)'
                ],
                borderWidth: 1
            }
        ]
    };

    return (
        <Container fluid className="learning-section">
            <Particle />
            <Container style={{ position: 'relative', zIndex: 1 }}>
                <h1 className="heading">Supervised Learning</h1>
                <p className="description">
                    Here you can explore different supervised learning techniques and run Python code directly in your browser.
                </p>
                {!pyodideLoaded && (
                    <div className="alert alert-python-loading">
                        Loading Python environment... Please wait.
                    </div>
                )}
                <Accordion 
                    activeKey={activeAccordion} 
                    onSelect={(k) => setActiveAccordion(k)}
                    style={{ position: 'relative', zIndex: 2 }}
                >
                    <Accordion.Item eventKey="0" className="accordion-item-custom">
                        <Accordion.Header className="accordion-header-custom">Linear Regression</Accordion.Header>
                        <Accordion.Body className="accordion-body-custom">
                            <Row>
                                <Col md={8}>
                                    <div className="code-section mt-3">
                                        <h5>Python Implementation</h5>
                                        <pre className="code-block">
                                            {linearRegressionCode}
                                        </pre>
                                        <Button 
                                            variant="primary" 
                                            onClick={(e) => {
                                                e.preventDefault();
                                                e.stopPropagation();
                                                runPythonCode(linearRegressionCode, setRegressionResults, setRegressionPlot);
                                            }}
                                            disabled={isLoading || !pyodideLoaded}
                                            className="run-button"
                                            style={{ position: 'relative', zIndex: 4 }}
                                        >
                                            {isLoading ? 'Running...' : 'Run Code'}
                                        </Button>
                                        {pythonOutput && (
                                            <div className="output-section mt-3">
                                                <h6>Output:</h6>
                                                <pre>{pythonOutput}</pre>
                                            </div>
                                        )}
                                        {regressionResults && (
                                            <div className="results-section mt-3">
                                                <h6>Results:</h6>
                                                <p>Model Coefficients: {regressionResults.coefficients.toFixed(2)}</p>
                                                <p>Intercept: {regressionResults.intercept.toFixed(2)}</p>
                                                <p>Mean Squared Error: {regressionResults.mse.toFixed(2)}</p>
                                                <p>R² Score: {regressionResults.r2.toFixed(2)}</p>
                                            </div>
                                        )}
                                        {regressionPlot && (
                                            <div className="chart-container mt-4">
                                                <img src={regressionPlot} alt="Linear Regression Plot" style={{ width: '100%', height: 'auto', borderRadius: '8px' }} />
                                            </div>
                                        )}
                                    </div>
                                </Col>
                                <Col md={4}>
                                    <div className="learning-description">
                                        <h4>Linear Regression</h4>
                                        <p>
                                            Linear regression is a statistical method that models the relationship between a dependent variable and one or more independent variables.
                                            The model assumes a linear relationship between the variables.
                                        </p>
                                        <ul>
                                            <li>Simple to understand and implement</li>
                                            <li>Works well with linear relationships</li>
                                            <li>Can be extended to multiple variables</li>
                                        </ul>
                                    </div>
                                </Col>
                            </Row>
                        </Accordion.Body>
                    </Accordion.Item>

                    <Accordion.Item eventKey="1" className="accordion-item-custom">
                        <Accordion.Header className="accordion-header-custom">Logistic Regression</Accordion.Header>
                        <Accordion.Body className="accordion-body-custom">
                            <Row>
                                <Col md={8}>
                                    <div className="code-section mt-3">
                                        <h5>Python Implementation</h5>
                                        <pre className="code-block">
                                            {logisticRegressionCode}
                                        </pre>
                                        <Button 
                                            variant="primary" 
                                            onClick={(e) => {
                                                e.preventDefault();
                                                e.stopPropagation();
                                                runPythonCode(logisticRegressionCode, setLogisticResults, setLogisticPlot);
                                            }}
                                            disabled={isLoading || !pyodideLoaded}
                                            className="run-button"
                                            style={{ position: 'relative', zIndex: 4 }}
                                        >
                                            {isLoading ? 'Running...' : 'Run Code'}
                                        </Button>
                                        {pythonOutput && (
                                            <div className="output-section mt-3">
                                                <h6>Output:</h6>
                                                <pre>{pythonOutput}</pre>
                                            </div>
                                        )}
                                        {logisticResults && (
                                            <div className="results-section mt-3">
                                                <h6>Results:</h6>
                                                <p>Model Coefficients: {logisticResults.coefficients.toFixed(2)}</p>
                                                <p>Intercept: {logisticResults.intercept.toFixed(2)}</p>
                                                <p>Accuracy: {logisticResults.accuracy.toFixed(2)}</p>
                                            </div>
                                        )}
                                        {logisticPlot && (
                                            <div className="chart-container mt-4">
                                                <img src={logisticPlot} alt="Logistic Regression Plot" style={{ width: '100%', height: 'auto', borderRadius: '8px' }} />
                                            </div>
                                        )}
                                    </div>
                                </Col>
                                <Col md={4}>
                                    <div className="learning-description">
                                        <h4>Logistic Regression</h4>
                                        <p>
                                            Logistic regression is used for binary classification problems. It predicts the probability of an event occurring by fitting data to a logistic function.
                                        </p>
                                        <ul>
                                            <li>Outputs probabilities between 0 and 1</li>
                                            <li>Works well for binary classification</li>
                                            <li>Easy to interpret results</li>
                                        </ul>
                                    </div>
                                </Col>
                            </Row>
                        </Accordion.Body>
                    </Accordion.Item>

                    <Accordion.Item eventKey="2" className="accordion-item-custom">
                        <Accordion.Header className="accordion-header-custom">Support Vector Machine (SVM)</Accordion.Header>
                        <Accordion.Body className="accordion-body-custom">
                            <Row>
                                <Col md={8}>
                                    <div className="interactive-controls">
                                        <select 
                                            value={svmKernel} 
                                            onChange={(e) => setSvmKernel(e.target.value)}
                                            className="form-select mb-3"
                                        >
                                            <option value="linear">Linear Kernel</option>
                                            <option value="rbf">RBF Kernel</option>
                                            <option value="poly">Polynomial Kernel</option>
                                        </select>
                                        <div className="code-section mt-3">
                                            <h5>Python Implementation</h5>
                                            <pre className="code-block">
                                                {svmCode}
                                            </pre>
                                            <Button 
                                                variant="primary" 
                                                onClick={(e) => {
                                                    e.preventDefault();
                                                    e.stopPropagation();
                                                    runPythonCode(svmCode, setSvmResults, setSvmPlot);
                                                }}
                                                disabled={isLoading || !pyodideLoaded}
                                                className="run-button"
                                                style={{ position: 'relative', zIndex: 4 }}
                                            >
                                                {isLoading ? 'Running...' : 'Run Code'}
                                            </Button>
                                            {pythonOutput && (
                                                <div className="output-section mt-3">
                                                    <h6>Output:</h6>
                                                    <pre>{pythonOutput}</pre>
                                                </div>
                                            )}
                                            {svmResults && (
                                                <div className="results-section mt-3">
                                                    <h6>Results:</h6>
                                                    <p>Accuracy: {svmResults.accuracy.toFixed(2)}</p>
                                                </div>
                                            )}
                                            {svmPlot && (
                                                <div className="chart-container mt-4">
                                                    <img src={svmPlot} alt="SVM Plot" style={{ width: '100%', height: 'auto', borderRadius: '8px' }} />
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </Col>
                                <Col md={4}>
                                    <div className="learning-description">
                                        <h4>Support Vector Machine</h4>
                                        <p>
                                            SVM is a powerful classification algorithm that finds the optimal hyperplane to separate classes.
                                            It works by maximizing the margin between classes.
                                        </p>
                                        <ul>
                                            <li>Effective in high-dimensional spaces</li>
                                            <li>Different kernel functions for non-linear classification</li>
                                            <li>Robust against overfitting</li>
                                        </ul>
                                    </div>
                                </Col>
                            </Row>
                        </Accordion.Body>
                    </Accordion.Item>

                    <Accordion.Item eventKey="3" className="accordion-item-custom">
                        <Accordion.Header className="accordion-header-custom">K-Nearest Neighbors (KNN)</Accordion.Header>
                        <Accordion.Body className="accordion-body-custom">
                            <Row>
                                <Col md={8}>
                                    <div className="interactive-controls">
                                        <input
                                            type="range"
                                            min="1"
                                            max="7"
                                            value={knnK}
                                            onChange={(e) => setKnnK(parseInt(e.target.value))}
                                            className="form-range mb-3"
                                        />
                                        <div className="code-section mt-3">
                                            <h5>Python Implementation</h5>
                                            <pre className="code-block">
                                                {knnCode}
                                            </pre>
                                            <Button 
                                                variant="primary" 
                                                onClick={(e) => {
                                                    e.preventDefault();
                                                    e.stopPropagation();
                                                    runPythonCode(knnCode, setKnnResults, setKnnPlot);
                                                }}
                                                disabled={isLoading || !pyodideLoaded}
                                                className="run-button"
                                                style={{ position: 'relative', zIndex: 4 }}
                                            >
                                                {isLoading ? 'Running...' : 'Run Code'}
                                            </Button>
                                            {pythonOutput && (
                                                <div className="output-section mt-3">
                                                    <h6>Output:</h6>
                                                    <pre>{pythonOutput}</pre>
                                                </div>
                                            )}
                                            {knnResults && (
                                                <div className="results-section mt-3">
                                                    <h6>Results:</h6>
                                                    <p>Accuracy: {knnResults.accuracy.toFixed(2)}</p>
                                                </div>
                                            )}
                                            {knnPlot && (
                                                <div className="chart-container mt-4">
                                                    <img src={knnPlot} alt="KNN Plot" style={{ width: '100%', height: 'auto', borderRadius: '8px' }} />
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </Col>
                                <Col md={4}>
                                    <div className="learning-description">
                                        <h4>K-Nearest Neighbors</h4>
                                        <p>
                                            KNN is a simple, instance-based learning algorithm that classifies new instances based on the majority class of their k nearest neighbors.
                                        </p>
                                        <ul>
                                            <li>No training phase required</li>
                                            <li>Easy to understand and implement</li>
                                            <li>Works well with non-linear data</li>
                                        </ul>
                                    </div>
                                </Col>
                            </Row>
                        </Accordion.Body>
                    </Accordion.Item>

                    <Accordion.Item eventKey="4" className="accordion-item-custom">
                        <Accordion.Header className="accordion-header-custom">Decision Trees</Accordion.Header>
                        <Accordion.Body className="accordion-body-custom">
                            <Row>
                                <Col md={8}>
                                    <div className="interactive-controls">
                                        <input
                                            type="range"
                                            min="1"
                                            max="5"
                                            value={treeDepth}
                                            onChange={(e) => setTreeDepth(parseInt(e.target.value))}
                                            className="form-range mb-3"
                                        />
                                        <div className="code-section mt-3">
                                            <h5>Python Implementation</h5>
                                            <pre className="code-block">
                                                {treeCode}
                                            </pre>
                                            <Button 
                                                variant="primary" 
                                                onClick={(e) => {
                                                    e.preventDefault();
                                                    e.stopPropagation();
                                                    runPythonCode(treeCode, setTreeResults, setTreePlot);
                                                }}
                                                disabled={isLoading || !pyodideLoaded}
                                                className="run-button"
                                                style={{ position: 'relative', zIndex: 4 }}
                                            >
                                                {isLoading ? 'Running...' : 'Run Code'}
                                            </Button>
                                            {pythonOutput && (
                                                <div className="output-section mt-3">
                                                    <h6>Output:</h6>
                                                    <pre>{pythonOutput}</pre>
                                                </div>
                                            )}
                                            {treeResults && (
                                                <div className="results-section mt-3">
                                                    <h6>Results:</h6>
                                                    <p>Accuracy: {treeResults.accuracy.toFixed(2)}</p>
                                                </div>
                                            )}
                                            {treePlot && (
                                                <div className="chart-container mt-4">
                                                    <img src={treePlot} alt="Decision Tree Plot" style={{ width: '100%', height: 'auto', borderRadius: '8px' }} />
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </Col>
                                <Col md={4}>
                                    <div className="learning-description">
                                        <h4>Decision Trees</h4>
                                        <p>
                                            Decision trees are hierarchical structures that make decisions by splitting data into subsets based on feature values.
                                        </p>
                                        <ul>
                                            <li>Easy to interpret and visualize</li>
                                            <li>Can handle both numerical and categorical data</li>
                                            <li>Non-parametric method</li>
                                        </ul>
                                    </div>
                                </Col>
                            </Row>
                        </Accordion.Body>
                    </Accordion.Item>

                    <Accordion.Item eventKey="5" className="accordion-item-custom">
                        <Accordion.Header className="accordion-header-custom">Random Forest</Accordion.Header>
                        <Accordion.Body className="accordion-body-custom">
                            <Row>
                                <Col md={8}>
                                    <div className="interactive-controls">
                                        <input
                                            type="range"
                                            min="1"
                                            max="10"
                                            value={forestSize}
                                            onChange={(e) => setForestSize(parseInt(e.target.value))}
                                            className="form-range mb-3"
                                        />
                                        <div className="code-section mt-3">
                                            <h5>Python Implementation</h5>
                                            <pre className="code-block">
                                                {forestCode}
                                            </pre>
                                            <Button 
                                                variant="primary" 
                                                onClick={(e) => {
                                                    e.preventDefault();
                                                    e.stopPropagation();
                                                    runPythonCode(forestCode, setForestResults, setForestPlot);
                                                }}
                                                disabled={isLoading || !pyodideLoaded}
                                                className="run-button"
                                                style={{ position: 'relative', zIndex: 4 }}
                                            >
                                                {isLoading ? 'Running...' : 'Run Code'}
                                            </Button>
                                            {pythonOutput && (
                                                <div className="output-section mt-3">
                                                    <h6>Output:</h6>
                                                    <pre>{pythonOutput}</pre>
                                                </div>
                                            )}
                                            {forestResults && (
                                                <div className="results-section mt-3">
                                                    <h6>Results:</h6>
                                                    <p>Accuracy: {forestResults.accuracy.toFixed(2)}</p>
                                                </div>
                                            )}
                                            {forestPlot && (
                                                <div className="chart-container mt-4">
                                                    <img src={forestPlot} alt="Random Forest Plot" style={{ width: '100%', height: 'auto', borderRadius: '8px' }} />
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </Col>
                                <Col md={4}>
                                    <div className="learning-description">
                                        <h4>Random Forest</h4>
                                        <p>
                                            Random Forest is an ensemble learning method that operates by constructing multiple decision trees and outputting the class that is the mode of the classes.
                                        </p>
                                        <ul>
                                            <li>Reduces overfitting through ensemble learning</li>
                                            <li>Handles high-dimensional data well</li>
                                            <li>Provides feature importance measures</li>
                                        </ul>
                                    </div>
                                </Col>
                            </Row>
                        </Accordion.Body>
                    </Accordion.Item>

                    <Accordion.Item eventKey="6" className="accordion-item-custom">
                        <Accordion.Header className="accordion-header-custom">Naive Bayes</Accordion.Header>
                        <Accordion.Body className="accordion-body-custom">
                            <Row>
                                <Col md={8}>
                                    <div className="interactive-controls">
                                        <div className="code-section mt-3">
                                            <h5>Python Implementation</h5>
                                            <pre className="code-block">
                                                {naiveBayesCode}
                                            </pre>
                                            <Button 
                                                variant="primary" 
                                                onClick={(e) => {
                                                    e.preventDefault();
                                                    e.stopPropagation();
                                                    runPythonCode(naiveBayesCode, setNaiveBayesResults, setNaiveBayesPlot);
                                                }}
                                                disabled={isLoading || !pyodideLoaded}
                                                className="run-button"
                                                style={{ position: 'relative', zIndex: 4 }}
                                            >
                                                {isLoading ? 'Running...' : 'Run Code'}
                                            </Button>
                                            {pythonOutput && (
                                                <div className="output-section mt-3">
                                                    <h6>Output:</h6>
                                                    <pre>{pythonOutput}</pre>
                                                </div>
                                            )}
                                            {naiveBayesResults && (
                                                <div className="results-section mt-3">
                                                    <h6>Results:</h6>
                                                    <p>Accuracy: {naiveBayesResults.accuracy.toFixed(2)}</p>
                                                </div>
                                            )}
                                            {naiveBayesPlot && (
                                                <div className="chart-container mt-4">
                                                    <img src={naiveBayesPlot} alt="Naive Bayes Plot" style={{ width: '100%', height: 'auto', borderRadius: '8px' }} />
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </Col>
                                <Col md={4}>
                                    <div className="learning-description">
                                        <h4>Naive Bayes</h4>
                                        <p>
                                            Naive Bayes is a probabilistic classifier based on Bayes' theorem with strong independence assumptions between features.
                                        </p>
                                        <ul>
                                            <li>Fast and efficient</li>
                                            <li>Works well with high-dimensional data</li>
                                            <li>Requires less training data</li>
                                        </ul>
                                    </div>
                                </Col>
                            </Row>
                        </Accordion.Body>
                    </Accordion.Item>

                    <Accordion.Item eventKey="7" className="accordion-item-custom">
                        <Accordion.Header className="accordion-header-custom">Neural Network</Accordion.Header>
                        <Accordion.Body className="accordion-body-custom">
                            <Row>
                                <Col md={8}>
                                    <div className="interactive-controls">
                                        <div className="code-section mt-3">
                                            <h5>Python Implementation</h5>
                                            <pre className="code-block">
                                                {neuralNetworkCode}
                                            </pre>
                                            <Button 
                                                variant="primary" 
                                                onClick={(e) => {
                                                    e.preventDefault();
                                                    e.stopPropagation();
                                                    runPythonCode(neuralNetworkCode, setNnResults, setNnPlot);
                                                }}
                                                disabled={isLoading || !pyodideLoaded}
                                                className="run-button"
                                                style={{ position: 'relative', zIndex: 4 }}
                                            >
                                                {isLoading ? 'Running...' : 'Run Code'}
                                            </Button>
                                            {pythonOutput && (
                                                <div className="output-section mt-3">
                                                    <h6>Output:</h6>
                                                    <pre>{pythonOutput}</pre>
                                                </div>
                                            )}
                                            {nnResults && (
                                                <div className="results-section mt-3">
                                                    <h6>Results:</h6>
                                                    <p>Accuracy: {nnResults.accuracy.toFixed(2)}</p>
                                                </div>
                                            )}
                                            {nnPlot && (
                                                <div className="chart-container mt-4">
                                                    <img src={nnPlot} alt="Neural Network Plot" style={{ width: '100%', height: 'auto', borderRadius: '8px' }} />
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </Col>
                                <Col md={4}>
                                    <div className="learning-description">
                                        <h4>Neural Network</h4>
                                        <p>
                                            Neural Networks are powerful machine learning models inspired by the human brain. They can learn complex patterns through multiple layers of interconnected neurons.
                                        </p>
                                        <ul>
                                            <li>Can learn complex non-linear relationships</li>
                                            <li>Powerful feature extraction capabilities</li>
                                            <li>Versatile for various types of problems</li>
                                        </ul>
                                    </div>
                                </Col>
                            </Row>
                        </Accordion.Body>
                    </Accordion.Item>
                </Accordion>
            </Container>
        </Container>
    );
};

export default SupervisedLearning;
