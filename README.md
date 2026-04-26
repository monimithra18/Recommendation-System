# ALS Matrix Factorization — MATLAB

![MATLAB](https://img.shields.io/badge/MATLAB-0076A8?style=for-the-badge&logo=mathworks&logoColor=white)

A **collaborative filtering recommender system** implemented in MATLAB using **Alternating Least Squares (ALS)** matrix factorization. Unlike item-based filtering, this approach learns latent feature vectors for both users and items by minimising a cost function over observed ratings — enabling predictions for unseen user-item pairs.

> **Note:** This is distinct from the Python item-based collaborative filtering in [movie-recommender-system](https://github.com/monimithra18/movie-recommender-system). That project uses cosine similarity between items; this project uses matrix factorisation to learn latent user and item features.

---

## 🧮 Overview

Matrix factorisation decomposes a user-item ratings matrix into two lower-dimensional matrices:
- **X** — Item feature matrix (num_items × num_features)
- **Theta** — User parameter matrix (num_users × num_features)

The predicted rating for user j on item i is: `Theta(j,:) * X(i,:)'`

The ALS algorithm alternately optimises X and Theta while holding the other fixed, minimising the regularised cost function using gradient descent.

---

## 🗂 Project Structure

```
als-matrix-factorization-matlab/
├── data/                   # Dataset files (ratings matrix)
├── movie_recommender.m     # Main script — runs the full recommendation pipeline
├── cofiCostFunc.m          # Collaborative filtering cost function + gradients
├── fmincg.m                # Function minimisation using conjugate gradient
├── loadMovieList.m         # Loads and parses the movie list
├── normalizeRatings.m      # Mean-normalises the ratings matrix
└── README.md
```

---

## 🛠 Tech Stack

- **MATLAB** — Full implementation in MATLAB
- **Collaborative Filtering** — Matrix factorisation with regularisation
- **Conjugate Gradient** — Used via `fmincg` for efficient optimisation

---

## ⚙️ How It Works

### 1. Load Data
Read the ratings matrix Y (num_movies × num_users) where Y(i,j) is user j's rating of movie i (0 if unrated), and R(i,j) = 1 if the rating exists.

### 2. Normalise Ratings
Subtract the mean rating for each movie using `normalizeRatings.m` to improve convergence.

### 3. Initialise Parameters
Randomly initialise X (item features) and Theta (user weights).

### 4. Minimise Cost Function
Use `fmincg` to minimise the collaborative filtering cost function defined in `cofiCostFunc.m`:

```
J = 0.5 * sum((X * Theta' - Y).^2 .* R) + (lambda/2) * (sum(Theta.^2) + sum(X.^2))
```

### 5. Generate Predictions
Compute predicted ratings matrix: `P = X * Theta'`

Recommend the top-rated unrated movies for each user.

---

## 🚀 Getting Started

```matlab
% Open MATLAB and navigate to the project folder
cd als-matrix-factorization-matlab

% Run the main recommender script
movie_recommender
```

---

## 📊 Key Parameters

| Parameter | Description |
|---|---|
| `num_features` | Number of latent features (e.g. 10) |
| `lambda` | Regularisation parameter to prevent overfitting |
| `num_users` | Total number of users in the dataset |
| `num_movies` | Total number of movies in the dataset |

---

## 🔭 Future Improvements

- Implement true Alternating Least Squares (fixed-point update)
- Compare with Python scikit-learn SVD-based factorisation
- Add Bayesian Personalised Ranking for implicit feedback
- Build a web interface using MATLAB Web App Server

---

*Built by [Monish Mithra Kadiyala](https://linkedin.com/in/monishmithra) ·*
