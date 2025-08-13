import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ML and Stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import pearsonr

# NLP imports
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from collections import Counter
import string

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class SpotifyAnalyzer:
    """
    Advanced Spotify music analysis toolkit
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.df_clean = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load and validate dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Dataset loaded successfully: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def preprocess_data(self):
        """Advanced data preprocessing pipeline"""
        self.df_clean = self.df.copy()
        
        # Handle missing values intelligently
        numeric_cols = self.df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df_clean.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            if self.df_clean[col].isnull().sum() > 0:
                self.df_clean[col].fillna(self.df_clean[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            if self.df_clean[col].isnull().sum() > 0:
                self.df_clean[col].fillna(self.df_clean[col].mode()[0], inplace=True)
        
        # Remove duplicates
        initial_shape = self.df_clean.shape[0]
        self.df_clean.drop_duplicates(inplace=True)
        removed_dupes = initial_shape - self.df_clean.shape[0]
        
        if removed_dupes > 0:
            print(f"Removed {removed_dupes} duplicate records")
        
        # Feature engineering
        self._engineer_features()
        
        print("Data preprocessing completed")
        return self.df_clean
    
    def _engineer_features(self):
        """Create new features from existing data"""
        numeric_cols = self.df_clean.select_dtypes(include=[np.number]).columns
        
        # Create energy-based features if audio features exist
        audio_features = ['danceability', 'energy', 'valence', 'tempo']
        existing_audio = [col for col in audio_features if col in numeric_cols]
        
        if len(existing_audio) >= 2:
            # Composite mood score
            if 'valence' in existing_audio and 'energy' in existing_audio:
                self.df_clean['mood_score'] = (self.df_clean['valence'] + self.df_clean['energy']) / 2
            
            # Danceability categories
            if 'danceability' in existing_audio:
                self.df_clean['dance_category'] = pd.cut(self.df_clean['danceability'], 
                                                       bins=3, labels=['Low', 'Medium', 'High'])
        
        # Duration categories if duration exists
        duration_cols = [col for col in self.df_clean.columns if 'duration' in col.lower()]
        if duration_cols:
            duration_col = duration_cols[0]
            self.df_clean['duration_category'] = pd.cut(self.df_clean[duration_col], 
                                                      bins=4, labels=['Short', 'Medium', 'Long', 'Very Long'])
    
    def exploratory_analysis(self):
        """Comprehensive EDA with advanced visualizations"""
        numeric_cols = self.df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df_clean.select_dtypes(include=['object']).columns
        
        print("=== EXPLORATORY DATA ANALYSIS ===")
        print(f"Dataset shape: {self.df_clean.shape}")
        print(f"Numeric features: {len(numeric_cols)}")
        print(f"Categorical features: {len(categorical_cols)}")
        
        # Advanced statistical summary
        self._statistical_summary()
        
        # Correlation analysis
        self._correlation_analysis()
        
        # Distribution analysis
        self._distribution_analysis()
        
        return True
    
    def _statistical_summary(self):
        """Advanced statistical analysis"""
        numeric_cols = self.df_clean.select_dtypes(include=[np.number]).columns
        
        stats_df = pd.DataFrame()
        for col in numeric_cols:
            stats_df[col] = {
                'mean': self.df_clean[col].mean(),
                'median': self.df_clean[col].median(),
                'std': self.df_clean[col].std(),
                'skewness': self.df_clean[col].skew(),
                'kurtosis': self.df_clean[col].kurtosis(),
                'cv': self.df_clean[col].std() / self.df_clean[col].mean() if self.df_clean[col].mean() != 0 else 0
            }
        
        print("\nAdvanced Statistical Summary:")
        print(stats_df.T.round(3))
    
    def _correlation_analysis(self):
        """Advanced correlation analysis with significance testing"""
        numeric_cols = self.df_clean.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return
        
        correlation_matrix = self.df_clean[numeric_cols].corr()
        
        # Find significant correlations
        significant_corrs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                corr_coef = correlation_matrix.loc[col1, col2]
                
                if abs(corr_coef) > 0.3:  # Threshold for meaningful correlation
                    # Calculate p-value
                    _, p_value = pearsonr(self.df_clean[col1], self.df_clean[col2])
                    significant_corrs.append((col1, col2, corr_coef, p_value))
        
        if significant_corrs:
            print("\nSignificant Correlations (|r| > 0.3):")
            for col1, col2, corr, p_val in significant_corrs:
                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"{col1} - {col2}: {corr:.3f} {significance}")
    
    def _distribution_analysis(self):
        """Analyze distributions and detect outliers"""
        numeric_cols = self.df_clean.select_dtypes(include=[np.number]).columns
        
        print("\nDistribution Analysis:")
        for col in numeric_cols:
            # Normality test
            _, p_value = stats.shapiro(self.df_clean[col].sample(min(5000, len(self.df_clean))))
            is_normal = p_value > 0.05
            
            # Outlier detection
            Q1 = self.df_clean[col].quantile(0.25)
            Q3 = self.df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df_clean[col] < Q1 - 1.5*IQR) | (self.df_clean[col] > Q3 + 1.5*IQR)).sum()
            
            print(f"{col}: {'Normal' if is_normal else 'Non-normal'} distribution, {outliers} outliers")
    
    def create_visualizations(self):
        """Generate comprehensive visualizations"""
        print("\nGenerating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create multiple visualization types
        self._create_distribution_plots()
        self._create_correlation_heatmap()
        self._create_interactive_plots()
        
        print("Visualizations saved successfully")
    
    def _create_distribution_plots(self):
        """Create distribution plots for numeric variables"""
        numeric_cols = self.df_clean.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return
        
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if len(numeric_cols) > 1 else [axes]
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                sns.histplot(data=self.df_clean, x=col, kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for j in range(len(numeric_cols), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('distribution_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_correlation_heatmap(self):
        """Create advanced correlation heatmap"""
        numeric_cols = self.df_clean.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return
        
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.df_clean[numeric_cols].corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_interactive_plots(self):
        """Create interactive plots using Plotly"""
        numeric_cols = self.df_clean.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # Scatter plot matrix
            fig = px.scatter_matrix(self.df_clean[numeric_cols].sample(min(1000, len(self.df_clean))),
                                  title="Interactive Scatter Plot Matrix")
            fig.update_traces(diagonal_visible=False)
            fig.write_html("scatter_matrix.html")
            
            print("Interactive scatter matrix saved as 'scatter_matrix.html'")
    
    def nlp_analysis(self):
        """Natural Language Processing analysis on text columns"""
        text_columns = []
        for col in self.df_clean.columns:
            if self.df_clean[col].dtype == 'object':
                # Check if column contains text (not just categories)
                sample_text = str(self.df_clean[col].iloc[0])
                if len(sample_text.split()) > 1:  # More than one word
                    text_columns.append(col)
        
        if not text_columns:
            print("No suitable text columns found for NLP analysis")
            return
        
        print(f"\nPerforming NLP analysis on: {text_columns}")
        
        for col in text_columns:
            self._analyze_text_column(col)
    
    def _analyze_text_column(self, column):
        """Analyze individual text column"""
        print(f"\n--- NLP Analysis for {column} ---")
        
        # Clean text data
        text_data = self.df_clean[column].astype(str).str.lower()
        text_data = text_data.str.replace(r'[^\w\s]', '', regex=True)
        
        # Sentiment Analysis
        sia = SentimentIntensityAnalyzer()
        sentiments = text_data.apply(lambda x: sia.polarity_scores(x))
        
        # Extract sentiment scores
        self.df_clean[f'{column}_sentiment_positive'] = [s['pos'] for s in sentiments]
        self.df_clean[f'{column}_sentiment_negative'] = [s['neg'] for s in sentiments]
        self.df_clean[f'{column}_sentiment_neutral'] = [s['neu'] for s in sentiments]
        self.df_clean[f'{column}_sentiment_compound'] = [s['compound'] for s in sentiments]
        
        # Text statistics
        self.df_clean[f'{column}_word_count'] = text_data.str.split().str.len()
        self.df_clean[f'{column}_char_count'] = text_data.str.len()
        
        # Word frequency analysis
        all_text = ' '.join(text_data)
        words = all_text.split()
        word_freq = Counter(words)
        
        print(f"Total words: {len(words)}")
        print(f"Unique words: {len(word_freq)}")
        print(f"Most common words: {word_freq.most_common(5)}")
        
        # Create word cloud
        if len(all_text) > 100:
            try:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Word Cloud for {column}')
                plt.savefig(f'wordcloud_{column}.png', dpi=300, bbox_inches='tight')
                plt.show()
            except Exception as e:
                print(f"Could not generate word cloud: {e}")
    
    def machine_learning_analysis(self):
        """Advanced ML analysis including clustering and classification"""
        print("\n=== MACHINE LEARNING ANALYSIS ===")
        
        # Prepare features
        numeric_cols = self.df_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 3:
            print("Insufficient numeric features for ML analysis")
            return
        
        # Feature scaling
        X = self.df_clean[numeric_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Clustering Analysis
        self._clustering_analysis(X_scaled, numeric_cols)
        
        # Dimensionality Reduction
        self._dimensionality_reduction(X_scaled)
        
        # Prediction modeling (if suitable target exists)
        self._predictive_modeling()
    
    def _clustering_analysis(self, X_scaled, feature_names):
        """Perform clustering analysis"""
        print("\nClustering Analysis:")
        
        # Determine optimal number of clusters
        inertias = []
        k_range = range(2, min(11, len(self.df_clean)//10))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Perform clustering with optimal k
        optimal_k = k_range[0]  # Simple heuristic - could be improved with elbow method
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        self.df_clean['cluster'] = clusters
        
        print(f"Created {optimal_k} clusters")
        print(f"Cluster distribution: {pd.Series(clusters).value_counts().sort_index().to_dict()}")
        
        # Cluster profiling
        cluster_profiles = self.df_clean.groupby('cluster')[feature_names].mean()
        print("\nCluster Profiles (mean values):")
        print(cluster_profiles.round(3))
    
    def _dimensionality_reduction(self, X_scaled):
        """Perform PCA for dimensionality reduction"""
        print("\nDimensionality Reduction (PCA):")
        
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Explained variance
        explained_var_ratio = pca.explained_variance_ratio_
        cumulative_var_ratio = np.cumsum(explained_var_ratio)
        
        # Find number of components for 95% variance
        n_components_95 = np.argmax(cumulative_var_ratio >= 0.95) + 1
        
        print(f"Components needed for 95% variance: {n_components_95}")
        print(f"First 3 components explain {cumulative_var_ratio[2]:.1%} of variance")
        
        # Visualize PCA
        if X_scaled.shape[1] >= 2:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(range(1, len(explained_var_ratio)+1), explained_var_ratio, 'bo-')
            plt.xlabel('Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title('PCA Explained Variance by Component')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(cumulative_var_ratio)+1), cumulative_var_ratio, 'ro-')
            plt.axhline(y=0.95, color='k', linestyle='--', alpha=0.7)
            plt.xlabel('Component')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('PCA Cumulative Explained Variance')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def _predictive_modeling(self):
        """Build predictive models if suitable targets exist"""
        # Look for potential target variables
        categorical_cols = self.df_clean.select_dtypes(include=['object']).columns
        potential_targets = []
        
        for col in categorical_cols:
            unique_vals = self.df_clean[col].nunique()
            if 2 <= unique_vals <= 10:  # Suitable for classification
                potential_targets.append(col)
        
        if not potential_targets:
            print("\nNo suitable target variables found for predictive modeling")
            return
        
        print(f"\nPredictive Modeling - Target options: {potential_targets}")
        
        # Use first suitable target
        target_col = potential_targets[0]
        numeric_cols = self.df_clean.select_dtypes(include=[np.number]).columns
        
        # Prepare data
        X = self.df_clean[numeric_cols].fillna(0)
        y = self.df_clean[target_col]
        
        # Encode target if necessary
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Random Forest Classifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = rf_model.predict(X_test_scaled)
        
        # Results
        print(f"\nRandom Forest Classification Results (Target: {target_col}):")
        print(f"Accuracy: {rf_model.score(X_test_scaled, y_test):.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 Important Features:")
        print(feature_importance.head())
        
        # Save feature importance plot
        plt.figure(figsize=(10, 6))
        top_features = feature_importance.head(10)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Feature Importance for {target_col} Prediction')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_insights_report(self):
        """Generate comprehensive insights report"""
        print("\n" + "="*50)
        print("         SPOTIFY DATA ANALYSIS REPORT")
        print("="*50)
        
        # Dataset overview
        print(f"\n📊 DATASET OVERVIEW")
        print(f"   Records: {len(self.df_clean):,}")
        print(f"   Features: {len(self.df_clean.columns)}")
        print(f"   Data Quality: {(1 - self.df_clean.isnull().sum().sum()/(len(self.df_clean)*len(self.df_clean.columns)))*100:.1f}%")
        
        # Key statistics
        numeric_cols = self.df_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\n📈 KEY INSIGHTS")
            for col in numeric_cols[:5]:  # Top 5 numeric columns
                mean_val = self.df_clean[col].mean()
                std_val = self.df_clean[col].std()
                skew_val = self.df_clean[col].skew()
                
                distribution_type = "Normal" if abs(skew_val) < 0.5 else ("Right-skewed" if skew_val > 0.5 else "Left-skewed")
                
                print(f"   • {col}: μ={mean_val:.3f}, σ={std_val:.3f} ({distribution_type})")
        
        # Correlations
        if len(numeric_cols) > 1:
            corr_matrix = self.df_clean[numeric_cols].corr()
            high_corrs = []
            
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corrs.append((numeric_cols[i], numeric_cols[j], corr_val))
            
            if high_corrs:
                print(f"\n🔗 STRONG CORRELATIONS")
                for col1, col2, corr in high_corrs[:3]:  # Top 3
                    print(f"   • {col1} ↔ {col2}: {corr:.3f}")
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"   Generated files: distribution_plots.png, correlation_heatmap.png")
        if hasattr(self, 'df_clean') and 'cluster' in self.df_clean.columns:
            print(f"   Clustering: {self.df_clean['cluster'].nunique()} clusters identified")
        
        return True

def main():
    """Main execution function"""
    # Configuration
    DATA_PATH = '/content/sample_data/SpotifyFeatures.csv'  # Update this path
    
    print("🎵 SPOTIFY MUSIC ANALYSIS PIPELINE")
    print("="*40)
    
    # Initialize analyzer
    analyzer = SpotifyAnalyzer(DATA_PATH)
    
    # Execute analysis pipeline
    try:
        # Step 1: Load data
        if not analyzer.load_data():
            return
        
        # Step 2: Preprocess
        analyzer.preprocess_data()
        
        # Step 3: Exploratory analysis
        analyzer.exploratory_analysis()
        
        # Step 4: Visualizations
        analyzer.create_visualizations()
        
        # Step 5: NLP analysis
        analyzer.nlp_analysis()
        
        # Step 6: Machine learning
        analyzer.machine_learning_analysis()
        
        # Step 7: Final report
        analyzer.generate_insights_report()
        
    except Exception as e:
        print(f"Error in analysis pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

print("\n💡 LEARNING RESOURCES:")
print("   • Practice with different datasets")
print("   • Learn advanced pandas functions")
print("   • Study machine learning algorithms")
print("   • Explore advanced visualization libraries")

print(f"\n✅ ANALYSIS COMPLETE!")
print("=" * 60)

# PART 16: FUNCTION EXAMPLES FOR REUSABILITY
# ==========================================
# What: Create reusable functions for common tasks
# Why: Make code more efficient and maintainable
# When: After learning the basic steps
# How: Wrap common operations in functions

def quick_data_summary(dataframe):
    """
    Quick summary of any dataset
    
    Parameters:
    dataframe (pd.DataFrame): The dataset to analyze
    
    Returns:
    None (prints summary)
    """
    print("📊 QUICK DATA SUMMARY")
    print("-" * 25)
    print(f"Shape: {dataframe.shape}")
    print(f"Missing values: {dataframe.isnull().sum().sum()}")
    print(f"Duplicates: {dataframe.duplicated().sum()}")
    print(f"Memory usage: {dataframe.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column types
    numerical = dataframe.select_dtypes(include=[np.number]).columns
    categorical = dataframe.select_dtypes(include=['object']).columns
    print(f"Numerical columns: {len(numerical)}")
    print(f"Categorical columns: {len(categorical)}")

def detect_outliers(dataframe, column, method='iqr'):
    """
    Detect outliers in a numerical column
    
    Parameters:
    dataframe (pd.DataFrame): The dataset
    column (str): Column name to check for outliers
    method (str): Method to use ('iqr' or 'zscore')
    
    Returns:
    pd.Series: Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (dataframe[column] < Q1 - 1.5*IQR) | (dataframe[column] > Q3 + 1.5*IQR)
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(dataframe[column]))
        outliers = z_scores > 3
    
    return outliers

# Example usage of the functions
print(f"\n🔧 USING CUSTOM FUNCTIONS:")
print("-" * 30)
quick_data_summary(df_clean)

# Example: Detect outliers in first numerical column if available
if len(numerical_cols) > 0:
    col_to_check = numerical_cols[0]
    outliers = detect_outliers(df_clean, col_to_check)
    print(f"\nOutliers in {col_to_check}: {outliers.sum()} ({outliers.sum()/len(df_clean)*100:.1f}%)")

print(f"\n🎓 CONGRATULATIONS! You've completed a comprehensive data analysis!")
print("Keep practicing with different datasets to master these techniques! 🚀")
