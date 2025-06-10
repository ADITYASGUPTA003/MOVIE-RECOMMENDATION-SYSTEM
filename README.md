-----

# Movie Recommendation System

This project involves building a comprehensive movie recommendation system using the MovieLens dataset, incorporating various collaborative filtering and content-based approaches. The system aims to provide personalized movie suggestions to users.

-----

## Table of Contents

1.  [Project Overview](https://www.google.com/search?q=%23project-overview)
2.  [Dataset](https://www.google.com/search?q=%23dataset)
3.  [Installation and Setup](https://www.google.com/search?q=%23installation-and-setup)
4.  [Data Preprocessing and Feature Engineering](https://www.google.com/search?q=%23data-preprocessing-and-feature-engineering)
5.  [Recommendation Algorithms Implemented](https://www.google.com/search?q=%23recommendation-algorithms-implemented)
6.  [Model Evaluation](https://www.google.com/search?q=%23model-evaluation)
7.  [Interactive Visualizations](https://www.google.com/search?q=%23interactive-visualizations)
8.  [Backend Usage](https://www.google.com/search?q=%23backend-usage)
9.  [Future Improvements](https://www.google.com/search?q=%23future-improvements)
10. [Contributing](https://www.google.com/search?q=%23contributing)
11. [License](https://www.google.com/search?q=%23license)

-----

## Project Overview

This project focuses on developing a movie recommendation system that helps users discover content relevant to their interests from a vast array of options. We implemented and evaluated several recommendation algorithms, including:

  * **Collaborative Filtering (User-Based and Item-Based):** Recommends items based on user-item interactions.
  * **Matrix Factorization (SVD):** Identifies latent factors from user-item interactions.
  * **Content-Based Filtering:** Recommends items based on their attributes and a user's past preferences.
  * **Hybrid Recommendation System:** Combines the strengths of multiple approaches.

Our goal is to provide personalized movie recommendations by analyzing user behavior and movie attributes.

-----

## Dataset

This project uses the **MovieLens latest-small dataset**, a widely-recognized benchmark for recommendation systems research.

The dataset consists of these files:

  * `movies.csv`: Contains movie IDs, titles, and genres.
  * `ratings.csv`: Contains user ratings for movies, including UserID, MovieID, Rating, and Timestamp.
  * `links.csv`: Provides MovieID, IMDB ID, and TMDB ID mappings.
  * `tags.csv`: Contains user-generated tags for movies, along with UserID, MovieID, Tag, and Timestamp.

The dataset is downloaded directly within the notebook for convenience.

-----

## Installation and Setup

To run this project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd movie-recommendation-system
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries (for the backend recommendation system):**

    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn scipy
    ```

    These are the libraries explicitly imported in the project's Python code.

4.  **For the interactive dashboard (frontend):**
    Navigate to the frontend directory (e.g., `frontend` or `src/components`, depending on your project structure).

    ```bash
    # Assuming your React app is in a 'frontend' directory at the root
    cd frontend
    npm install # or yarn install
    ```

    Make sure you have **React, Recharts, and D3.js** installed:

    ```bash
    npm install react recharts d3 # or yarn add react recharts d3
    ```

-----

## Data Preprocessing and Feature Engineering

The data undergoes a thorough cleaning and feature engineering process to prepare it for the recommendation algorithms:

  * **Missing Value Handling:** We check for and address missing values (e.g., `tmdbId` in `links.csv` is filled with -1).
  * **Duplicate Removal:** Duplicate entries are removed from both `movies` and `ratings` datasets.
  * **Timestamp Conversion:** Timestamp columns are converted to datetime objects for easier manipulation.
  * **Feature Extraction:**
      * The `year` is extracted from movie titles.
      * One-hot encoded genre features are created.
      * User activity features (`total_ratings`, `avg_rating`, `rating_std` per user) are generated.
      * Movie popularity features (`total_ratings`, `avg_rating`, `rating_std` per movie) are generated.
      * Time-based features (year, month, day, day of week) are extracted from ratings timestamps.
  * **Data Integrity Checks:** We ensure consistent movie IDs, valid rating ranges (0.5-5.0), and handle cold-start issues by identifying users/movies with very few ratings.
  * **Outlier Handling:** Users with extreme rating patterns (e.g., rating everything the same) are identified and addressed.
  * **Data Transformation:** Ratings are normalized by user mean, and Min-Max scaling is applied to dense features.
  * **User-Item Matrix:** A pivoted user-item matrix is created, which is essential for collaborative filtering algorithms.

-----

## Recommendation Algorithms Implemented

The project implements and evaluates five distinct recommendation algorithms:

1.  **User-Based Collaborative Filtering:** This method identifies users with similar tastes and recommends movies liked by those similar users but not yet watched by the target user.
2.  **Item-Based Collaborative Filtering:** This algorithm recommends movies similar to those the user has already liked, based on item-to-item similarity.
3.  **Matrix Factorization (SVD):** We use Singular Value Decomposition to uncover latent factors that explain user-item interactions and predict ratings for unrated movies.
4.  **Content-Based Filtering:** This approach builds a profile of the user's preferences based on the genres of movies they have rated highly, then recommends movies with similar genre attributes.
5.  **Hybrid Recommendation System:** This system combines the predictions from the user-based, item-based, SVD, and content-based methods by normalizing and weighting their scores. This approach provides a more robust and generally more accurate recommendation.

-----

## Model Evaluation

We evaluate the algorithms using a test set of users to accurately assess their performance.

**Metrics Used:**

  * **RMSE (Root Mean Squared Error):** This measures the accuracy of predicted ratings, with a lower RMSE indicating better accuracy.
  * **Precision:** Represents the proportion of recommended items that are actually relevant to the user.
  * **Recall:** Represents the proportion of relevant items that are successfully recommended to the user.
  * **F1-Score:** This is the harmonic mean of Precision and Recall, offering a single metric that balances both.

The evaluation process involves:

1.  Selecting a subset of users for testing.
2.  Splitting their ratings into training and test sets.
3.  Generating recommendations using each algorithm based on the training data.
4.  Calculating RMSE, Precision, and Recall by comparing predicted recommendations against the held-out test set.

We generate comparison plots for each metric to visually represent the performance of different algorithms.

-----

## Interactive Visualizations

A **React component, `MovieRecommendationDashboard.jsx`**, has been integrated to provide a comprehensive and interactive data analysis dashboard. This dashboard offers rich visual insights into:

  * **Dataset Overview:** Explore genre distribution, rating patterns, and temporal trends directly from the MovieLens dataset.
  * **Algorithm Performance:** Visually compare **RMSE, Precision, Recall, and F1-Score** across different recommendation algorithms.
  * **User Preferences:** An interactive **D3.js heatmap** allows for detailed analysis of user preferences across various movie genres.
  * **Movie Similarity:** A **D3.js force-directed network graph** vividly illustrates relationships between movies based on their similarity.

The dashboard leverages **Recharts** for static charts and **D3.js** for dynamic, interactive visualizations, significantly enhancing data storytelling and user understanding.

-----

## Backend Usage

The `MovieRecommendationSystem` class provides a unified Python interface for generating recommendations.

```python
import pandas as pd
# Assuming 'movies' and 'ratings' DataFrames are loaded as per the project
# from your data loading script or a pre-processed file.

# Load the dataset (example, replace with your actual loading logic if different)
# !wget [https://files.grouplens.org/datasets/movielens/ml-latest-small.zip](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)
# !unzip ml-latest-small.zip
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Preprocessing steps for movies and ratings (as done in the project)
# For example, extracting year and one-hot encoding genres for 'movies'
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
movies['title'] = movies['title'].str.replace(r' \(\d{4}\)$', '', regex=True)
genres = movies['genres'].str.get_dummies('|')
movies = pd.concat([movies, genres], axis=1)

# Convert timestamp to datetime
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

# You'd also merge user activity and movie popularity features here
# For demonstration purposes, let's assume `ratings` and `movies` are ready

from movie_recommendation_system_script import MovieRecommendationSystem, user_based_recommendation, item_based_recommendation, svd_recommendation, content_based_recommendation, hybrid_recommendation

# Initialize the recommender system
recommender = MovieRecommendationSystem(ratings, movies)

# Example 1: Get recommendations for a specific user
user_id = 1
recommendations = recommender.recommend_for_user(user_id, method='hybrid')
print(f"Recommendations for user {user_id} using Hybrid method:")
print(recommendations)

# Example 2: Find similar movies to a given movie
movie_id = 1  # Example movie ID (Toy Story)
similar_movies = recommender.recommend_similar_movies(movie_id)
print(f"\nMovies similar to {movies[movies['movieId'] == movie_id]['title'].values[0]}:")
print(similar_movies)

# Example 3: Get popular movies by rating
popular_by_rating = recommender.popular_movies(by='rating')
print("\nPopular movies by rating:")
print(popular_by_rating)

# Example 4: Get recommendations for a new user based on genre preferences
genre_preferences = {
    'Action': 5,
    'Adventure': 4,
    'Sci-Fi': 5,
    'Drama': 2,
    'Comedy': 3
}
new_user_recs = recommender.recommend_for_new_user(genre_preferences)
print("\nRecommendations for new user with genre preferences:")
print(new_user_recs)
```

-----

## Future Improvements

  * Implement more **advanced deep learning-based recommendation models** (e.g., Neural Collaborative Filtering).
  * Integrate **real-time data streaming** for live recommendation updates.
  * Develop a **full-fledged user interface** for interactive recommendations and feedback.
  * Explore **sentiment analysis on movie reviews** for richer content understanding.
  * Optimize algorithms for handling **cold-start problems** more effectively.

-----

## Contributing

Contributions are welcome\! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

-----
