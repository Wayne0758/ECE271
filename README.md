# Feature Selection via ℓ2-Distance to Centroids
Feature selection plays a crucial role in mitigating the challenges posed by high dimensionality, improving model interpretability,
reducing computational costs, and enhancing predictive performance. Traditional feature selection methods—including filter, wrapper,
and embedded approaches—often suffer from high computational complexity and instability, especially when dealing with large-scale
multi-category data. In this work, we explore sparse centroid-based feature selection methods, which aim to minimize the distance
between data samples and their corresponding class centroids while enforcing sparsity constraints. We identify key limitations
in existing approaches, such as non-convex optimization and sensitivity to noise, and propose an improved linear transformation
framework that mitigates these issues through convex optimization techniques. Our method achieves efficient and stable feature
selection while preserving classification accuracy, making it a promising approach for handling high-dimensional datasets in modern
machine learning applications.
