const learningTopics = [
  {
    slug: "bayesian-statistics",
    title: "Bayesian Statistics",
    description:
      "Priors, likelihoods, posterior inference, credible intervals, model comparison, and Bayesian workflows.",
    status: "Populated guide",
    link: "/learning/bayesian-statistics",
    cta: "Open guide",
    tags: ["Priors", "Posteriors", "MCMC"],
    overview:
      "Bayesian statistics treats unknown quantities as probability distributions. The workflow combines prior information with observed data through a likelihood to produce a posterior distribution for inference and decision making.",
    keyIdeas: [
      "Choose priors that encode assumptions clearly and check whether conclusions are prior-sensitive.",
      "Interpret uncertainty through the posterior distribution rather than a single point estimate.",
      "Use posterior predictive checks and model comparison to test whether the model explains the observed data.",
    ],
    methods: [
      {
        name: "Bayes theorem",
        detail: "Updates prior beliefs with observed evidence.",
        formula: "p(theta | y) = p(y | theta) p(theta) / p(y)",
      },
      {
        name: "Posterior predictive distribution",
        detail: "Generates plausible future data under posterior uncertainty.",
        formula: "p(y_new | y) = integral p(y_new | theta) p(theta | y) dtheta",
      },
      {
        name: "Hierarchical modeling",
        detail: "Shares information across groups through population-level parameters.",
        formula: "y_ij ~ p(y | theta_j), theta_j ~ p(theta | phi), phi ~ p(phi)",
      },
      {
        name: "Markov chain Monte Carlo",
        detail: "Approximates posterior expectations with dependent posterior samples.",
        formula: "E[f(theta) | y] ~= (1 / S) sum_{s=1}^S f(theta_s)",
      },
      {
        name: "Variational inference",
        detail: "Approximates a posterior with an optimized simpler distribution.",
        formula: "q*(theta) = argmin_q KL(q(theta) || p(theta | y))",
      },
    ],
  },
  {
    slug: "simulation",
    title: "Simulation",
    description:
      "Monte Carlo experiments, uncertainty propagation, synthetic data generation, and stochastic system testing.",
    status: "Populated guide",
    link: "/learning/simulation",
    cta: "Open guide",
    tags: ["Monte Carlo", "Uncertainty", "Synthetic Data"],
    overview:
      "Simulation lets you study systems that are difficult to solve analytically. In data science, it is useful for stress testing assumptions, propagating uncertainty, validating estimators, and building synthetic benchmarks.",
    keyIdeas: [
      "Write down the data-generating process before trusting simulated results.",
      "Use repeated trials, fixed seeds, and parameter sweeps to make experiments reproducible.",
      "Compare estimated quantities to known simulated truth whenever possible.",
    ],
    methods: [
      {
        name: "Monte Carlo estimation",
        detail: "Approximates expectations with random samples.",
        formula: "E[g(X)] ~= (1 / N) sum_{i=1}^N g(x_i), x_i ~ p(x)",
      },
      {
        name: "Bootstrap simulation",
        detail: "Estimates sampling variability by resampling observed data.",
        formula: "theta_b* = T(x_1*, ..., x_n*), x_i* sampled with replacement from x",
      },
      {
        name: "Uncertainty propagation",
        detail: "Pushes random input uncertainty through a model or transformation.",
        formula: "Y = f(X), X ~ p_X(x), Var(Y) estimated from simulated f(x_i)",
      },
      {
        name: "Discrete-event simulation",
        detail: "Models systems as state changes occurring at event times.",
        formula: "state(t_{k+1}) = update(state(t_k), event_k)",
      },
      {
        name: "Sensitivity analysis",
        detail: "Measures how outputs respond to changes in inputs or parameters.",
        formula: "S_i = Var_{X_i}(E[Y | X_i]) / Var(Y)",
      },
    ],
  },
  {
    slug: "optimization-methods",
    title: "Optimization Methods",
    description:
      "Gradient descent, adaptive optimizers, objective functions, constraints, and convergence diagnostics.",
    status: "Populated guide",
    link: "/learning/optimization-methods",
    cta: "Open guide",
    tags: ["SGD", "Adam", "Convergence"],
    overview:
      "Optimization is the engine behind model fitting. It defines how parameters move through a loss landscape toward solutions that balance fit, constraints, and generalization.",
    keyIdeas: [
      "The loss function encodes what the model is rewarded or penalized for doing.",
      "Learning rate is often the most important first hyperparameter to tune.",
      "Convergence diagnostics should consider gradient behavior, validation metrics, and sensitivity to initialization.",
    ],
    methods: [
      {
        name: "Gradient descent",
        detail: "Updates parameters by moving opposite the gradient.",
        formula: "theta_{t+1} = theta_t - alpha grad L(theta_t)",
      },
      {
        name: "Stochastic gradient descent",
        detail: "Uses mini-batch gradients to scale optimization to large data.",
        formula: "theta_{t+1} = theta_t - alpha grad L_B(theta_t)",
      },
      {
        name: "Momentum",
        detail: "Smooths updates by accumulating velocity across gradients.",
        formula: "v_t = beta v_{t-1} + grad L(theta_t); theta_{t+1} = theta_t - alpha v_t",
      },
      {
        name: "Adam",
        detail: "Combines momentum with adaptive per-parameter learning rates.",
        formula: "theta_{t+1} = theta_t - alpha m_hat_t / (sqrt(v_hat_t) + epsilon)",
      },
      {
        name: "Constrained optimization",
        detail: "Optimizes an objective while satisfying equality or inequality constraints.",
        formula: "min_x f(x) subject to g_i(x) <= 0, h_j(x) = 0",
      },
    ],
  },
  {
    slug: "supervised-learning",
    title: "Supervised Learning",
    description:
      "Regression, classification, model validation, and supervised decision boundaries.",
    status: "Populated guide",
    link: "/learning/supervised-learning",
    cta: "Open guide",
    tags: ["Regression", "Classification", "Validation"],
    overview:
      "Supervised learning maps labeled examples to predictions. The central workflow is to define a target, choose useful features, train a model, and evaluate whether it generalizes beyond the training sample.",
    keyIdeas: [
      "Separate training, validation, and test data before tuning.",
      "Use metrics that match the task: RMSE or MAE for regression, precision/recall/F1/AUC for classification.",
      "Watch for leakage, class imbalance, and overly complex models that memorize the training data.",
    ],
    methods: [
      {
        name: "Linear regression",
        detail: "Models a continuous outcome as a linear function of features.",
        formula: "y = X beta + epsilon, epsilon ~ N(0, sigma^2 I)",
      },
      {
        name: "Logistic regression",
        detail: "Models binary class probability with a logistic link.",
        formula: "P(y = 1 | x) = 1 / (1 + exp(-x^T beta))",
      },
      {
        name: "K-nearest neighbors",
        detail: "Predicts from the labels or values of nearby samples.",
        formula: "y_hat = mode({y_i : x_i in N_k(x)})",
      },
      {
        name: "Support vector machines",
        detail: "Finds a margin-maximizing separating hyperplane.",
        formula: "min_{w,b} 0.5 ||w||^2 + C sum_i xi_i subject to y_i(w^T x_i + b) >= 1 - xi_i",
      },
      {
        name: "Random forests",
        detail: "Averages many decorrelated decision trees.",
        formula: "f_hat(x) = (1 / B) sum_{b=1}^B T_b(x)",
      },
    ],
  },
  {
    slug: "unsupervised-learning",
    title: "Unsupervised Learning",
    description:
      "Clustering, dimensionality reduction, latent structure, and exploratory data analysis.",
    status: "Populated guide",
    link: "/learning/unsupervised-learning",
    cta: "Open guide",
    tags: ["Clustering", "PCA", "UMAP"],
    overview:
      "Unsupervised learning finds structure without labels. It is useful for exploration, compression, visualization, anomaly discovery, and generating hypotheses about hidden groups.",
    keyIdeas: [
      "Scale features before distance-based methods.",
      "Treat clusters as analytical hypotheses, not ground truth.",
      "Use multiple views: PCA for linear structure, UMAP or t-SNE for nonlinear visual exploration.",
    ],
    methods: [
      {
        name: "K-means clustering",
        detail: "Partitions points into clusters by minimizing within-cluster squared distance.",
        formula: "min_{C_1,...,C_k} sum_{j=1}^k sum_{x_i in C_j} ||x_i - mu_j||^2",
      },
      {
        name: "Gaussian mixture models",
        detail: "Models data as a weighted mixture of Gaussian components.",
        formula: "p(x) = sum_{k=1}^K pi_k N(x | mu_k, Sigma_k)",
      },
      {
        name: "Principal component analysis",
        detail: "Projects data onto directions of maximum variance.",
        formula: "w_1 = argmax_{||w||=1} Var(Xw)",
      },
      {
        name: "Singular value decomposition",
        detail: "Factorizes a matrix into orthogonal latent directions and singular values.",
        formula: "X = U Sigma V^T",
      },
      {
        name: "Density-based clustering",
        detail: "Forms clusters from dense regions separated by sparse regions.",
        formula: "N_eps(x) = {y : dist(x, y) <= eps}, core if |N_eps(x)| >= minPts",
      },
    ],
  },
  {
    slug: "deep-learning",
    title: "Deep Learning",
    description:
      "Neural networks, representation learning, architectures, training loops, and regularization.",
    status: "Populated guide",
    link: "/learning/deep-learning",
    cta: "Open guide",
    tags: ["PyTorch", "CNNs", "Transformers"],
    overview:
      "Deep learning uses multi-layer neural networks to learn hierarchical representations from data. It is especially strong for images, audio, text, and other high-dimensional signals.",
    keyIdeas: [
      "Start with a small baseline before scaling architecture size.",
      "Track training and validation loss to diagnose underfitting, overfitting, and optimization issues.",
      "Use regularization, normalization, and learning-rate schedules before adding unnecessary complexity.",
    ],
    methods: [
      {
        name: "Feed-forward neural networks",
        detail: "Composes affine transformations and nonlinear activations.",
        formula: "h_l = phi(W_l h_{l-1} + b_l), y_hat = W_L h_{L-1} + b_L",
      },
      {
        name: "Backpropagation",
        detail: "Computes gradients through the chain rule over network layers.",
        formula: "dL/dW_l = (dL/dh_l)(dh_l/dz_l)(dz_l/dW_l)",
      },
      {
        name: "Convolutional layers",
        detail: "Applies shared filters over local windows in images or spectrograms.",
        formula: "(X * K)(i,j) = sum_m sum_n X(i+m,j+n) K(m,n)",
      },
      {
        name: "Autoencoders",
        detail: "Learns compressed latent representations by reconstructing inputs.",
        formula: "min_{f,g} sum_i ||x_i - g(f(x_i))||^2",
      },
      {
        name: "Attention",
        detail: "Weights contextual interactions between tokens or features.",
        formula: "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V",
      },
    ],
  },
];

export default learningTopics;
