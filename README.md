# Search Algorithms and Multiple Linear Regression
Welcome to two exciting projects. In these projects, we delve into the realms of optimizing flight planning and predicting flight prices, showcasing the application of cutting-edge algorithms and machine learning techniques.

# Project 1: Finding the Best Flight Route ✈️🌍
This project revolves around the optimization of flight planning using two powerful algorithms: Dijkstra and A*. Developed to minimize travel time, cost, and duration, these algorithms are implemented to find the optimal flight routes. Real-world data, including factors like distance, flight duration, and cost between airports, is incorporated to provide comprehensive and practical solutions for flight route optimization.
## Dijkstra Algorithm
- Graph traversal algorithm for finding the shortest path in a weighted graph.
- Guarantees the shortest path if all edge weights are non-negative.
- Time complexity: O((V + E) log V)
## A* Algorithm
- Combines Dijkstra algorithm with a heuristic function for efficient pathfinding.
- Utilizes a priority queue for node selection, guided by heuristic information.
- Time complexity: O((V + E) log V)

## Results and Analysis
- A* algorithm explores fewer nodes compared to Dijkstra, offering potential time efficiency.
- A* algorithm is chosen for its ability to prioritize paths based on heuristic information, making it suitable for flight route optimization.


# Project 2: Flight Price Prediction 📈💰
In the second project, we shift our focus to predicting flight prices using multiple linear regression. Leveraging historical flight data and considering various features such as departure/arrival time, airlines, flight duration, and distance, this project aims to empower travelers with a reliable tool for estimating flight prices.

key concepts:

## Multiple Linear Regression
Statistical modeling technique to predict the relationship between dependent and multiple independent variables. It assumes a linear relationship between the predictor variables and the response variable.

## Gradient Descent
Gradient descent is an optimization algorithm used to minimize the cost function by iteratively updating model parameters. It involves initializing parameters, computing the cost, calculating gradients, and updating parameters until convergence or a predefined number of iterations.

## Results

![result](https://github.com/SheidaAbedpour/Search-Algorithms/blob/main/Predict_Price/Result.PNG)


### Logs:
- `MSE`: 0.10
- `RMSE`: 0.31
- `MAE`: 0.20
- `R2`: 0.90


# Refrences
- [baeldung.com](https://www.baeldung.com/cs/dijkstra-vs-a-pathfinding)
- [medium.com](https://medium.com/@miguell.m/dijkstras-and-a-search-algorithm-2e67029d7749)
- [youtube.com/@LearningOrbis](https://www.youtube.com/watch?v=LGiRB_lByh0&list=LL&index=5&t=650s)
- [youtube.com/@SimplilearnOfficial](https://www.youtube.com/watch?v=Mb1srg1ON60&list=LL&index=6&t=1976s)
- [youtube.com/@codebasics](https://www.youtube.com/watch?v=9yl6-HEY7_s&list=LL&index=2&t=970s)
- [coursera.org/learn/machine-learning](https://www.coursera.org/learn/machine-learning)
