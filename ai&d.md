# Data Science & AI Cheatsheet

## Table of Contents
1. [Python for Data Science](#python-for-data-science)
2. [Data Manipulation](#data-manipulation)
3. [Data Visualization](#data-visualization)
4. [Statistics & Probability](#statistics--probability)
5. [Machine Learning Basics](#machine-learning-basics)
6. [Supervised Learning](#supervised-learning)
7. [Unsupervised Learning](#unsupervised-learning)
8. [Deep Learning](#deep-learning)
9. [Natural Language Processing](#natural-language-processing)
10. [Model Evaluation](#model-evaluation)
11. [Feature Engineering](#feature-engineering)
12. [MLOps & Deployment](#mlops--deployment)

---

## Python for Data Science

### Essential Libraries
```python
import numpy as np           # Numerical computing
import pandas as pd          # Data manipulation
import matplotlib.pyplot as plt  # Visualization
import seaborn as sns        # Statistical visualization
import sklearn               # Machine learning
```

### NumPy Basics
NumPy is the foundation for numerical computing in Python, providing efficient array operations.

```python
# Create arrays
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2], [3, 4]])
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
identity = np.eye(3)
random = np.random.rand(3, 3)  # Random values between 0 and 1

# Array operations
arr + 10                # Element-wise addition
arr * 2                 # Element-wise multiplication
arr ** 2                # Element-wise power
np.sqrt(arr)            # Square root
np.exp(arr)             # Exponential
np.log(arr)             # Natural logarithm

# Statistical operations
arr.mean()              # Average
arr.std()               # Standard deviation
arr.sum()               # Sum of all elements
arr.min()               # Minimum value
arr.max()               # Maximum value
arr.argmin()            # Index of minimum
arr.argmax()            # Index of maximum

# Array manipulation
arr.reshape(5, 1)       # Change shape
arr.T                   # Transpose
np.concatenate([arr1, arr2])  # Join arrays
np.vstack([arr1, arr2]) # Stack vertically
np.hstack([arr1, arr2]) # Stack horizontally

# Indexing and slicing
arr[0]                  # First element
arr[-1]                 # Last element
arr[1:4]                # Elements from index 1 to 3
arr[arr > 3]            # Boolean indexing
```

**Context:** NumPy arrays are much faster than Python lists for numerical operations because they're stored in contiguous memory blocks and support vectorized operations (operations on entire arrays without loops).

---

## Data Manipulation

### Pandas DataFrames
Pandas is the go-to library for working with structured data (like spreadsheets or SQL tables).

```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})

# Read/Write data
df = pd.read_csv('data.csv')
df = pd.read_excel('data.xlsx')
df = pd.read_json('data.json')
df.to_csv('output.csv', index=False)

# Basic info
df.head()               # First 5 rows
df.tail()               # Last 5 rows
df.info()               # Data types and non-null counts
df.describe()           # Statistical summary
df.shape                # (rows, columns)
df.columns              # Column names
df.dtypes               # Data types of each column

# Selecting data
df['name']              # Single column (Series)
df[['name', 'age']]     # Multiple columns (DataFrame)
df.loc[0]               # Select by label/index
df.iloc[0]              # Select by position
df.loc[0, 'name']       # Specific cell by label
df.iloc[0, 0]           # Specific cell by position

# Filtering
df[df['age'] > 25]      # Boolean filtering
df[df['name'].str.contains('Alice')]  # String matching
df[(df['age'] > 25) & (df['salary'] > 55000)]  # Multiple conditions

# Sorting
df.sort_values('age')                    # Sort by column
df.sort_values(['age', 'salary'], ascending=[True, False])

# Adding/Removing columns
df['bonus'] = df['salary'] * 0.1         # Add column
df = df.drop('bonus', axis=1)            # Remove column
df = df.drop(0, axis=0)                  # Remove row

# Handling missing data
df.isnull()             # Check for missing values
df.dropna()             # Remove rows with missing values
df.fillna(0)            # Fill missing values with 0
df.fillna(df.mean())    # Fill with mean

# Grouping and aggregation
df.groupby('age').mean()                 # Group by age, calculate mean
df.groupby('age')['salary'].sum()        # Sum salaries by age group
df.groupby('age').agg({'salary': ['mean', 'sum', 'count']})

# Merging DataFrames
pd.merge(df1, df2, on='id')              # SQL-like join
pd.concat([df1, df2], axis=0)            # Concatenate vertically
pd.concat([df1, df2], axis=1)            # Concatenate horizontally

# Apply functions
df['age'].apply(lambda x: x + 1)         # Apply function to column
df.apply(lambda row: row['age'] + row['salary'], axis=1)  # Row-wise
```

**Context:** DataFrames are similar to spreadsheets or SQL tables. They're ideal for data cleaning, exploration, and preparation before modeling.

---

## Data Visualization

### Matplotlib
The foundational plotting library in Python.

```python
import matplotlib.pyplot as plt

# Line plot
plt.plot(x, y)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Line Plot')
plt.legend(['Line 1'])
plt.grid(True)
plt.show()

# Scatter plot
plt.scatter(x, y, c='red', alpha=0.5)
plt.show()

# Bar plot
plt.bar(categories, values)
plt.show()

# Histogram
plt.hist(data, bins=30, edgecolor='black')
plt.show()

# Subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x, y)
axes[0, 1].scatter(x, y)
axes[1, 0].bar(x, y)
axes[1, 1].hist(data)
plt.tight_layout()
plt.show()
```

### Seaborn
Built on top of Matplotlib, provides more attractive and informative statistical graphics.

```python
import seaborn as sns

# Set style
sns.set_style('whitegrid')

# Distribution plots
sns.histplot(data, kde=True)            # Histogram with density curve
sns.kdeplot(data)                       # Kernel density estimation
sns.boxplot(x='category', y='value', data=df)  # Box plot
sns.violinplot(x='category', y='value', data=df)  # Violin plot

# Relationship plots
sns.scatterplot(x='feature1', y='feature2', hue='category', data=df)
sns.lineplot(x='time', y='value', data=df)
sns.regplot(x='x', y='y', data=df)      # Scatter with regression line

# Categorical plots
sns.barplot(x='category', y='value', data=df)
sns.countplot(x='category', data=df)

# Matrix plots
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
sns.clustermap(data, cmap='viridis')

# Pairwise relationships
sns.pairplot(df, hue='category')        # All pairwise scatter plots
```

**Context:** Visualization is crucial for understanding data distributions, relationships between variables, and communicating insights. Always visualize your data before building models.

---

## Statistics & Probability

### Descriptive Statistics
Understanding the basic properties of your data.

```python
# Central tendency
data.mean()             # Average value
data.median()           # Middle value
data.mode()             # Most frequent value

# Dispersion
data.std()              # Standard deviation (spread)
data.var()              # Variance (std squared)
data.min()              # Minimum
data.max()              # Maximum
data.quantile(0.25)     # 25th percentile (Q1)
data.quantile(0.75)     # 75th percentile (Q3)

# Distribution shape
from scipy.stats import skew, kurtosis
skew(data)              # Asymmetry of distribution
kurtosis(data)          # Tailedness of distribution
```

**Context:** Descriptive statistics summarize data. Mean tells you the center, standard deviation tells you the spread. Skewness tells you if data leans left or right, kurtosis tells you about extreme values (outliers).

### Probability Distributions
```python
from scipy import stats

# Normal (Gaussian) distribution
# Bell curve, most data near mean
norm = stats.norm(loc=0, scale=1)  # mean=0, std=1
norm.pdf(x)             # Probability density at x
norm.cdf(x)             # Cumulative probability up to x
norm.rvs(size=1000)     # Generate random samples

# Binomial distribution
# Number of successes in n trials (coin flips)
binom = stats.binom(n=10, p=0.5)
binom.pmf(k)            # Probability of exactly k successes

# Poisson distribution
# Number of events in fixed interval (emails per hour)
poisson = stats.poisson(mu=3)
poisson.pmf(k)          # Probability of exactly k events

# Uniform distribution
# All values equally likely
uniform = stats.uniform(loc=0, scale=10)
```

**Context:** Understanding distributions helps you know what to expect from data and choose appropriate statistical tests and models.

### Hypothesis Testing
Testing whether patterns in data are statistically significant or just random chance.

```python
from scipy import stats

# T-test: Compare means of two groups
# Example: Does treatment affect test scores?
t_stat, p_value = stats.ttest_ind(group1, group2)
# If p_value < 0.05, groups are significantly different

# Chi-square test: Test independence of categorical variables
# Example: Is smoking related to lung cancer?
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

# ANOVA: Compare means of multiple groups
# Example: Do three teaching methods produce different results?
f_stat, p_value = stats.f_oneway(group1, group2, group3)

# Correlation
# Pearson: Linear relationship between continuous variables
corr, p_value = stats.pearsonr(x, y)
# Spearman: Monotonic relationship (works with ranks)
corr, p_value = stats.spearmanr(x, y)
```

**Context:** P-value < 0.05 means "there's less than 5% chance this result is random." It's the standard threshold for claiming something is "statistically significant."

---

## Machine Learning Basics

### The ML Workflow
1. **Problem Definition:** What are you trying to predict?
2. **Data Collection:** Gather relevant data
3. **Data Exploration:** Understand patterns and issues
4. **Data Preprocessing:** Clean and prepare data
5. **Feature Engineering:** Create useful features
6. **Model Selection:** Choose appropriate algorithm
7. **Training:** Fit model to data
8. **Evaluation:** Test model performance
9. **Tuning:** Optimize hyperparameters
10. **Deployment:** Put model into production

### Types of Machine Learning

**Supervised Learning:** Learn from labeled examples (input → output)
- Classification: Predict categories (spam/not spam)
- Regression: Predict numbers (house prices)

**Unsupervised Learning:** Find patterns in unlabeled data
- Clustering: Group similar items
- Dimensionality Reduction: Simplify data

**Reinforcement Learning:** Learn by trial and error with rewards
- Game playing, robotics

### Scikit-learn Template
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Prepare data
X = df.drop('target', axis=1)  # Features
y = df['target']                # Target variable

# 2. Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Scale features (important for many algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 5. Make predictions
y_pred = model.predict(X_test_scaled)

# 6. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
```

**Context:** This is the standard workflow for most ML projects. Always split data before any preprocessing to avoid data leakage (accidentally using test information during training).

---

## Supervised Learning

### Linear Regression
Predicts continuous values using a straight line (or hyperplane).

**When to use:** Predicting prices, sales, temperatures, etc.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R-squared (0 to 1, higher is better)

# Get coefficients
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')
```

**Context:** RMSE tells you the average prediction error in the same units as your target. R² tells you what proportion of variance your model explains (0.8 = model explains 80% of variance).

### Logistic Regression
Despite the name, it's for classification (binary outcomes).

**When to use:** Email spam detection, disease diagnosis, customer churn.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)  # Probability scores

# Metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

**Context:** Logistic regression outputs probabilities (0 to 1). Threshold of 0.5 is default for classification, but you can adjust based on cost of false positives vs false negatives.

### Decision Trees
Makes decisions by asking a series of questions about features.

**When to use:** When you need interpretability, non-linear relationships.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

model = DecisionTreeClassifier(max_depth=5, min_samples_split=20)
model.fit(X_train, y_train)

# Visualize tree
plt.figure(figsize=(20, 10))
tree.plot_tree(model, feature_names=X.columns, filled=True)
plt.show()

# Feature importance
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
```

**Context:** Decision trees are intuitive but prone to overfitting. They create axis-aligned splits (rectangles), which can be inefficient for diagonal boundaries.

### Random Forest
Ensemble of many decision trees voting together.

**When to use:** Generally strong performer, handles non-linearity, less prone to overfitting than single trees.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,
    min_samples_split=20,
    random_state=42
)
model.fit(X_train, y_train)

# Feature importance
importances = model.feature_importances_
```

**Context:** Random Forests are "ensemble" models - they combine predictions from many trees. Each tree sees a random subset of data and features, making the overall model more robust.

### Gradient Boosting
Builds trees sequentially, each correcting errors of previous ones.

**When to use:** Often wins Kaggle competitions, excellent performance but slower to train.

```python
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb

# Scikit-learn version
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)

# XGBoost (popular, optimized implementation)
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    use_label_encoder=False
)

# LightGBM (even faster, Microsoft's version)
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
```

**Context:** Boosting builds models iteratively. Each new model focuses on examples the previous models got wrong. Learning rate controls how much each tree contributes.

### Support Vector Machines (SVM)
Finds optimal boundary (hyperplane) between classes.

**When to use:** Binary classification, especially with clear margin between classes.

```python
from sklearn.svm import SVC

model = SVC(
    kernel='rbf',          # 'linear', 'poly', 'rbf', 'sigmoid'
    C=1.0,                 # Regularization (smaller = more regularization)
    gamma='scale'          # Kernel coefficient
)
model.fit(X_train, y_train)
```

**Context:** SVMs work by finding the decision boundary that maximizes the margin between classes. Kernel trick allows them to handle non-linear boundaries. Can be slow on large datasets.

### K-Nearest Neighbors (KNN)
Classifies based on majority vote of k nearest neighbors.

**When to use:** Small datasets, when decision boundary is irregular, as baseline.

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance'     # 'uniform' or 'distance'
)
model.fit(X_train, y_train)
```

**Context:** KNN is "lazy learning" - it doesn't really train, just memorizes training data. At prediction time, it finds k most similar training examples. Simple but can be slow and memory-intensive.

---

## Unsupervised Learning

### K-Means Clustering
Groups data into k clusters based on similarity.

**When to use:** Customer segmentation, image compression, organizing documents.

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Train model
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Get cluster centers
centers = kmeans.cluster_centers_

# Evaluate using elbow method (find optimal k)
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

**Context:** K-means minimizes within-cluster variance. The "elbow" in the plot suggests optimal k - where adding more clusters doesn't significantly reduce inertia. It assumes clusters are spherical and equal-sized.

### Hierarchical Clustering
Creates tree of clusters (dendrogram).

**When to use:** When you want to see cluster hierarchy, don't know k in advance.

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Create dendrogram
linkage_matrix = linkage(X, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.show()

# Fit model
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
clusters = model.fit_predict(X)
```

**Context:** Hierarchical clustering doesn't require specifying k upfront. Dendrogram shows how clusters merge at different similarity levels. You can "cut" the tree at any level to get desired number of clusters.

### DBSCAN
Density-based clustering that can find arbitrary-shaped clusters and outliers.

**When to use:** Non-spherical clusters, outlier detection, noisy data.

```python
from sklearn.cluster import DBSCAN

model = DBSCAN(
    eps=0.5,               # Maximum distance between neighbors
    min_samples=5          # Minimum points to form cluster
)
clusters = model.fit_predict(X)

# -1 indicates outliers/noise
outliers = X[clusters == -1]
```

**Context:** Unlike K-means, DBSCAN doesn't require specifying number of clusters and can identify outliers (labeled as -1). It groups points that are closely packed together.

### Principal Component Analysis (PCA)
Reduces dimensionality while preserving variance.

**When to use:** Data visualization, feature reduction, noise filtering.

```python
from sklearn.decomposition import PCA

# Reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Explained variance
print(f'Explained variance: {pca.explained_variance_ratio_}')

# Find optimal number of components
pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumsum)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

# Keep components explaining 95% of variance
n_components = np.argmax(cumsum >= 0.95) + 1
```

**Context:** PCA finds directions (principal components) of maximum variance in your data. First component captures most variance, second captures second-most, etc. Useful for visualizing high-dimensional data and speeding up models.

### t-SNE
Non-linear dimensionality reduction for visualization.

**When to use:** Visualizing high-dimensional data (images, embeddings).

```python
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30,         # Balance local vs global structure
    random_state=42
)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis')
plt.show()
```

**Context:** t-SNE is great for visualization but slower than PCA and doesn't preserve global structure well. Good for exploring if there are natural clusters. Don't use for feature reduction before modeling.

---

## Deep Learning

### Neural Networks with TensorFlow/Keras
Neural networks learn hierarchical representations through layers.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Build model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    layers.Dropout(0.2),              # Prevent overfitting
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Output layer
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)

# Predict
predictions = model.predict(X_new)
```

**Context:** Neural networks consist of layers of neurons. Each neuron applies weights, bias, and activation function. Network learns by adjusting weights through backpropagation. More layers = "deeper" network.

### Common Activation Functions
```python
# ReLU (Rectified Linear Unit) - most common for hidden layers
# f(x) = max(0, x)
layers.Dense(64, activation='relu')

# Sigmoid - outputs 0 to 1, good for binary classification
# f(x) = 1 / (1 + e^(-x))
layers.Dense(1, activation='sigmoid')

# Softmax - outputs probabilities summing to 1, multi-class classification
# f(x_i) = e^(x_i) / sum(e^(x_j))
layers.Dense(num_classes, activation='softmax')

# Tanh - outputs -1 to 1
# f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
layers.Dense(64, activation='tanh')
```

**Context:** Activation functions introduce non-linearity, allowing networks to learn complex patterns. ReLU is default choice for hidden layers (fast, avoids vanishing gradients). Output activation depends on task.

### Convolutional Neural Networks (CNN)
Specialized for image data, uses convolutional layers to detect patterns.

**When to use:** Image classification, object detection, computer vision.

```python
model = keras.Sequential([
    # Convolutional layers learn spatial features
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),      # Reduce spatial dimensions
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten and classify
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

**Context:** CNNs use filters (kernels) that slide across images detecting features like edges, textures, and patterns. Early layers detect simple features, deeper layers detect complex objects. MaxPooling reduces dimensionality and provides translation invariance.

### Recurrent Neural Networks (RNN) / LSTM
Specialized for sequential data with memory of previous inputs.

**When to use:** Time series, text, speech, any sequential data.

```python
model = keras.Sequential([
    # LSTM maintains long-term memory
    layers.LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
    layers.LSTM(32),
    layers.Dense(1)
])

# For text classification
model = keras.Sequential([
    layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])
```

**Context:** RNNs process sequences one element at a time, maintaining hidden state. LSTMs (Long Short-Term Memory) solve vanishing gradient problem, allowing learning of long-term dependencies. Used for text generation, translation, sentiment analysis.

### Transfer Learning
Use pre-trained models and fine-tune for your task.

**When to use:** Limited data, want to leverage knowledge from large datasets.

```python
# Load pre-trained model
base_model = keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model layers
base_model.trainable = False

# Add custom layers
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Train only new layers
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)

# Optionally, fine-tune some base layers
base_model.trainable = True
# Freeze early layers, unfreeze later layers
for layer in base_model.layers[:-4]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='categorical_crossentropy')
model.fit(X_train, y_train, epochs=10)
```

**Context:** Transfer learning leverages models trained on huge datasets (ImageNet, Wikipedia). You freeze most layers and only train final layers on your data. Much faster and requires less data than training from scratch.

---

## Natural Language Processing

### Text Preprocessing
```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

text = "The cats are running and jumping quickly!"

# Tokenization (split into words)
tokens = word_tokenize(text.lower())
# ['the', 'cats', 'are', 'running', 'and', 'jumping', 'quickly', '!']

# Remove stopwords (common words like 'the', 'is', 'and')
stop_words = set(stopwords.words('english'))
filtered = [w for w in tokens if w not in stop_words]
# ['cats', 'running', 'jumping', 'quickly', '!']

# Stemming (reduce to root form, crude)
stemmer = PorterStemmer()
stemmed = [stemmer.stem(w) for w in filtered]
# ['cat', 'run', 'jump', 'quick', '!']

# Lemmatization (reduce to dictionary form, better)
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w, pos='v') for w in filtered]
# ['cat', 'run', 'jump', 'quickly', '!']
```

**Context:** Tokenization breaks text into units. Stopwords are common words that often don't carry much meaning. Stemming and lemmatization reduce words to base forms (running→run, better→good).

### Bag of Words (BoW)
Represents text as word frequency counts.

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love Python"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# See vocabulary
print(vectorizer.get_feature_names_out())
# ['amazing', 'i', 'is', 'learning', 'love', 'machine', 'python']

# Each document as word count vector
