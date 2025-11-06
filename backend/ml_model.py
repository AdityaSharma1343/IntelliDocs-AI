"""
ML Model for IntelliDocs AI
Implements BM25 ranking and Learning to Rank with evaluation metrics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
import json
import pickle
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class BM25Ranker:
    """BM25 ranking algorithm for document scoring"""
    
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Length normalization parameter
        self.avg_doc_len = 0
        self.doc_lengths = []
        self.doc_count = 0
        self.idf_scores = {}
        
    def fit(self, documents: List[str]):
        """Fit BM25 model on document corpus"""
        self.doc_count = len(documents)
        self.doc_lengths = [len(doc.split()) for doc in documents]
        self.avg_doc_len = np.mean(self.doc_lengths)
        
        # Calculate IDF scores
        word_doc_count = {}
        for doc in documents:
            words_in_doc = set(doc.lower().split())
            for word in words_in_doc:
                word_doc_count[word] = word_doc_count.get(word, 0) + 1
        
        # IDF = log((N - df + 0.5) / (df + 0.5))
        for word, df in word_doc_count.items():
            self.idf_scores[word] = np.log((self.doc_count - df + 0.5) / (df + 0.5))
        
        print(f"✅ BM25 fitted on {self.doc_count} documents")
        return self
    
    def score(self, query: str, document: str, doc_index: int = 0) -> float:
        """Calculate BM25 score for a query-document pair"""
        query_words = query.lower().split()
        doc_words = document.lower().split()
        doc_len = len(doc_words)
        
        score = 0.0
        for word in query_words:
            if word not in self.idf_scores:
                continue
                
            tf = doc_words.count(word)
            idf = self.idf_scores[word]
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
            score += idf * (numerator / denominator)
        
        return score

class LearnToRankModel:
    """Learning to Rank model for search result ranking"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.is_trained = False
        
    def extract_features(self, query: str, document: Dict) -> np.ndarray:
        """Extract features from query-document pair"""
        features = []
        
        query_lower = query.lower()
        title = document.get('title', '').lower()
        content = document.get('content', '').lower()
        
        # 1. Title exact match
        features.append(1.0 if query_lower == title else 0.0)
        
        # 2. Title contains query
        features.append(1.0 if query_lower in title else 0.0)
        
        # 3. Query terms in title
        query_words = query_lower.split()
        title_words = title.split()
        features.append(sum(1 for word in query_words if word in title_words) / max(len(query_words), 1))
        
        # 4. Content contains query
        features.append(1.0 if query_lower in content else 0.0)
        
        # 5. Query terms in content
        content_words = content.split()
        features.append(sum(1 for word in query_words if word in content_words) / max(len(query_words), 1))
        
        # 6. Term frequency in title
        tf_title = sum(title.count(word) for word in query_words) / max(len(title_words), 1)
        features.append(tf_title)
        
        # 7. Term frequency in content
        tf_content = sum(content.count(word) for word in query_words) / max(len(content_words), 1)
        features.append(tf_content)
        
        # 8. Document length (normalized)
        features.append(len(content_words) / 1000.0)
        
        # 9. Title length (normalized)
        features.append(len(title_words) / 20.0)
        
        # 10. Category relevance (simplified)
        category_score = 0.5  # Default score
        if document.get('category', '').lower() in query_lower:
            category_score = 1.0
        features.append(category_score)
        
        return np.array(features)
    
    def generate_training_data(self, documents: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for the model"""
        print("📊 Generating training data...")
        
        # Define sample queries and their relevant documents
        training_queries = [
            ("employee handbook", ["Employee Handbook", "HR", "policies"]),
            ("security policy", ["Security", "IT", "password"]),
            ("remote work", ["Remote", "work from home", "guidelines"]),
            ("benefits", ["Benefits", "insurance", "401k"]),
            ("vacation policy", ["vacation", "PTO", "time off"]),
            ("code of conduct", ["conduct", "ethics", "behavior"]),
            ("it guidelines", ["IT", "technology", "computer"]),
            ("health insurance", ["health", "medical", "insurance"]),
            ("salary", ["compensation", "salary", "pay"]),
            ("training", ["training", "development", "learning"])
        ]
        
        X = []
        y = []
        
        for query, relevant_terms in training_queries:
            for doc in documents:
                features = self.extract_features(query, doc)
                
                # Calculate relevance score (0-1)
                relevance = 0.0
                doc_text = (doc.get('title', '') + ' ' + doc.get('content', '')).lower()
                
                # Check how many relevant terms are in the document
                matches = sum(1 for term in relevant_terms if term.lower() in doc_text)
                relevance = min(matches / len(relevant_terms), 1.0)
                
                # Add some noise to make it more realistic
                relevance += np.random.normal(0, 0.1)
                relevance = np.clip(relevance, 0, 1)
                
                X.append(features)
                y.append(relevance)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Generated {len(X)} training samples with {X.shape[1]} features")
        return X, y
    
    def train(self, documents: List[Dict]):
        """Train the ranking model"""
        if len(documents) < 10:
            print("⚠️ Not enough documents for training. Need at least 10 documents.")
            return self
            
        X, y = self.generate_training_data(documents)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("🎯 Training Random Forest ranking model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"✅ Model trained successfully!")
        print(f"   Training R² Score: {train_score:.3f}")
        print(f"   Testing R² Score: {test_score:.3f}")
        
        self.is_trained = True
        return self
    
    def predict_relevance(self, query: str, document: Dict) -> float:
        """Predict relevance score for a query-document pair"""
        if not self.is_trained:
            # Return simple heuristic score if model not trained
            query_lower = query.lower()
            doc_text = (document.get('title', '') + ' ' + document.get('content', '')).lower()
            return doc_text.count(query_lower) / max(len(doc_text.split()), 1)
        
        features = self.extract_features(query, document)
        features_scaled = self.scaler.transform([features])
        score = self.model.predict(features_scaled)[0]
        return float(np.clip(score, 0, 1))
    
    def save_model(self, filepath: str = "ranking_model.pkl"):
        """Save trained model to file"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str = "ranking_model.pkl"):
        """Load trained model from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_importance = model_data['feature_importance']
            self.is_trained = model_data['is_trained']
            print(f"✅ Model loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"⚠️ Model file not found: {filepath}")
            return False

class SearchEvaluator:
    """Evaluate search performance with various metrics"""
    
    def __init__(self):
        self.results = {}
        
    def evaluate_search_results(self, 
                               queries: List[str], 
                               search_results: List[List[Dict]], 
                               ground_truth: List[List[str]]) -> Dict:
        """
        Evaluate search results against ground truth
        
        Args:
            queries: List of search queries
            search_results: List of search results for each query
            ground_truth: List of relevant document IDs for each query
        """
        
        all_predictions = []
        all_labels = []
        precisions = []
        recalls = []
        
        for query, results, relevant_docs in zip(queries, search_results, ground_truth):
            # Get top-k results
            k = min(10, len(results))
            top_k_ids = [doc['id'] for doc in results[:k]]
            
            # Calculate metrics for this query
            for doc in results:
                doc_id = doc['id']
                # Binary relevance: 1 if document is relevant, 0 otherwise
                is_relevant = 1 if doc_id in relevant_docs else 0
                predicted_relevant = 1 if doc.get('score', 0) > 0.5 else 0
                
                all_labels.append(is_relevant)
                all_predictions.append(predicted_relevant)
            
            # Precision@k and Recall@k
            relevant_in_top_k = sum(1 for doc_id in top_k_ids if doc_id in relevant_docs)
            precision_at_k = relevant_in_top_k / k if k > 0 else 0
            recall_at_k = relevant_in_top_k / len(relevant_docs) if len(relevant_docs) > 0 else 0
            
            precisions.append(precision_at_k)
            recalls.append(recall_at_k)
        
        # Calculate overall metrics
        precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Mean Average Precision
        map_score = np.mean(precisions)
        
        self.results = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'map': map_score,
            'avg_precision_at_k': np.mean(precisions),
            'avg_recall_at_k': np.mean(recalls),
            'all_predictions': all_predictions,
            'all_labels': all_labels
        }
        
        return self.results
    
    def generate_roc_curve(self, scores: List[float], labels: List[int], save_path: str = "roc_curve.png"):
        """Generate and save ROC curve"""
        if len(scores) == 0 or len(labels) == 0:
            print("⚠️ Not enough data to generate ROC curve")
            return
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random baseline')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve for Document Search Ranking', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 ROC curve saved to {save_path}")
        print(f"   AUC Score: {roc_auc:.3f}")
        
        return roc_auc
    
    def generate_confusion_matrix(self, save_path: str = "confusion_matrix.png"):
        """Generate and save confusion matrix"""
        if 'all_predictions' not in self.results:
            print("⚠️ No evaluation results available")
            return
        
        cm = confusion_matrix(self.results['all_labels'], self.results['all_predictions'])
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix for Search Results', fontsize=14, fontweight='bold')
        plt.colorbar()
        
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Not Relevant', 'Relevant'], rotation=45)
        plt.yticks(tick_marks, ['Not Relevant', 'Relevant'])
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Confusion matrix saved to {save_path}")
    
    def print_evaluation_report(self):
        """Print detailed evaluation report"""
        if not self.results:
            print("⚠️ No evaluation results available")
            return
        
        print("\n" + "="*60)
        print("📊 SEARCH MODEL EVALUATION REPORT")
        print("="*60)
        print(f"📌 Precision:           {self.results['precision']:.3f}")
        print(f"📌 Recall:              {self.results['recall']:.3f}")
        print(f"📌 F1-Score:            {self.results['f1_score']:.3f}")
        print(f"📌 Accuracy:            {self.results['accuracy']:.3f}")
        print(f"📌 Mean Avg Precision:  {self.results['map']:.3f}")
        print(f"📌 Avg Precision@10:    {self.results['avg_precision_at_k']:.3f}")
        print(f"📌 Avg Recall@10:       {self.results['avg_recall_at_k']:.3f}")
        print("="*60)

class IntellidocsMLPipeline:
    """Complete ML pipeline for IntelliDocs AI"""
    
    def __init__(self):
        self.bm25 = BM25Ranker()
        self.ranker = LearnToRankModel()
        self.evaluator = SearchEvaluator()
        self.documents = []
        
    def load_documents(self, doc_file: str = "documents_db.json"):
        """Load documents from database"""
        try:
            with open(doc_file, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            print(f"✅ Loaded {len(self.documents)} documents")
            return True
        except FileNotFoundError:
            print(f"⚠️ Document file not found: {doc_file}")
            return False
    
    def train_models(self):
        """Train all ML models"""
        if not self.documents:
            print("⚠️ No documents loaded. Please load documents first.")
            return
        
        print("\n🚀 Starting ML Model Training...")
        print("-" * 40)
        
        # Train BM25
        doc_texts = [doc.get('content', '') for doc in self.documents]
        self.bm25.fit(doc_texts)
        
        # Train Learning to Rank model
        self.ranker.train(self.documents)
        
        print("\n✅ All models trained successfully!")
    
    def rank_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Rank documents for a given query using ML models"""
        ranked_docs = []
        
        for i, doc in enumerate(documents):
            # BM25 score
            bm25_score = self.bm25.score(query, doc.get('content', ''), i)
            
            # ML model score
            ml_score = self.ranker.predict_relevance(query, doc)
            
            # Combine scores (weighted average)
            final_score = 0.6 * ml_score + 0.4 * (bm25_score / 10)  # Normalize BM25
            
            doc_copy = doc.copy()
            doc_copy['ml_score'] = ml_score
            doc_copy['bm25_score'] = bm25_score
            doc_copy['final_score'] = final_score
            
            ranked_docs.append(doc_copy)
        
        # Sort by final score
        ranked_docs.sort(key=lambda x: x['final_score'], reverse=True)
        
        return ranked_docs
    
    def evaluate_system(self):
        """Evaluate the complete search system"""
        print("\n📊 Evaluating Search System...")
        print("-" * 40)
        
        # Sample evaluation queries
        test_queries = [
            "employee handbook",
            "security policy",
            "remote work",
            "benefits",
            "vacation"
        ]
        
        # For demo, we'll create synthetic ground truth
        search_results = []
        ground_truth = []
        all_scores = []
        all_labels = []
        
        for query in test_queries:
            # Get search results
            results = self.rank_documents(query, self.documents[:20])  # Use subset for evaluation
            search_results.append(results)
            
            # Create synthetic ground truth (documents with query terms in title)
            relevant = [doc['id'] for doc in self.documents 
                       if query.lower() in doc.get('title', '').lower()]
            ground_truth.append(relevant)
            
            # Collect scores and labels for ROC curve
            for doc in results:
                all_scores.append(doc['final_score'])
                all_labels.append(1 if doc['id'] in relevant else 0)
        
        # Evaluate
        metrics = self.evaluator.evaluate_search_results(test_queries, search_results, ground_truth)
        
        # Print report
        self.evaluator.print_evaluation_report()
        
        # Generate visualizations
        if all_scores and all_labels:
            self.evaluator.generate_roc_curve(all_scores, all_labels)
            self.evaluator.generate_confusion_matrix()
        
        return metrics
    
    def save_pipeline(self, directory: str = "ml_models"):
        """Save all trained models"""
        Path(directory).mkdir(exist_ok=True)
        
        # Save ranking model
        self.ranker.save_model(f"{directory}/ranking_model.pkl")
        
        # Save BM25 parameters
        bm25_params = {
            'k1': self.bm25.k1,
            'b': self.bm25.b,
            'avg_doc_len': self.bm25.avg_doc_len,
            'doc_lengths': self.bm25.doc_lengths,
            'doc_count': self.bm25.doc_count,
            'idf_scores': self.bm25.idf_scores
        }
        with open(f"{directory}/bm25_params.json", 'w') as f:
            json.dump(bm25_params, f)
        
        print(f"✅ ML pipeline saved to {directory}/")

# Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 INTELLIDOCS AI - ML MODEL TRAINING")
    print("="*60)
    
    # Initialize pipeline
    pipeline = IntellidocsMLPipeline()
    
    # Load documents
    if pipeline.load_documents("documents_db.json"):
        # Train models
        pipeline.train_models()
        
        # Evaluate system
        metrics = pipeline.evaluate_system()
        
        # Save models
        pipeline.save_pipeline()
        
        # Test ranking
        print("\n🔍 Testing Document Ranking...")
        print("-" * 40)
        test_query = "employee benefits"
        print(f"Query: '{test_query}'")
        
        results = pipeline.rank_documents(test_query, pipeline.documents[:5])
        print("\nTop 3 Results:")
        for i, doc in enumerate(results[:3], 1):
            print(f"{i}. {doc.get('title', 'Untitled')}")
            print(f"   ML Score: {doc['ml_score']:.3f}")
            print(f"   BM25 Score: {doc['bm25_score']:.3f}")
            print(f"   Final Score: {doc['final_score']:.3f}")
    else:
        print("⚠️ Please ensure documents_db.json exists in the backend directory")
        print("   Upload some documents first through the web interface")