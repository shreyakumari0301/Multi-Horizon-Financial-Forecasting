# DSAI - Deep Learning for Time Series Forecasting

A comprehensive research framework for time series forecasting using deep learning models (LSTM, Transformer, TCN) with Ridge regression as a baseline. This project addresses the challenge of predicting future values in multivariate time series data while maintaining temporal dependencies and avoiding common pitfalls in sequence modeling.

## Problem Statement

Time series forecasting presents unique challenges compared to standard regression problems:

1. **Temporal Dependencies**: Future values depend on historical patterns, not just current features
2. **Non-Stationarity**: Statistical properties change over time
3. **Sequence Alignment**: Every timestep needs a prediction, but initial timesteps lack sufficient historical context
4. **Over-Smoothing**: Deep learning models often produce overly smooth predictions that miss important directional changes
5. **Evaluation Complexity**: Standard train/test splits can leak future information, requiring careful chronological validation

Traditional machine learning approaches treat each observation independently, losing critical temporal information. Deep learning models can capture these dependencies but require careful architectural and training design to avoid common pitfalls.

## Research Objectives

This project aims to:

1. **Develop a unified framework** for comparing deep learning models (LSTM, Transformer, TCN) against a linear baseline (Ridge)
2. **Address over-smoothing** in deep models through delta target prediction and architectural improvements
3. **Ensure proper temporal validation** through chronological splits and left-padding strategies
4. **Provide comprehensive evaluation** including financial metrics (PnL, Sharpe ratio) and residual analysis
5. **Enable reproducible research** through standardized APIs and experiment management

## Methodology

### Core Design Principles

#### 1. Sliding Window Sequences with Left-Padding

**Problem**: Standard sequence models require fixed-length inputs, but early timesteps lack sufficient history.

**Solution**: We implement a sliding window approach with left-padding:
- Create sequences of fixed length L from historical observations
- For the first L-1 timesteps, pad with the first observation to maintain full sequence length
- This ensures every timestep receives a prediction while maintaining temporal order

**Rationale**: Left-padding preserves the sequence structure without introducing future information leakage. The first observation serves as a reasonable proxy for missing history, allowing the model to make predictions from the very first timestep.

#### 2. Delta Target Prediction

**Problem**: Deep learning models, especially LSTM and TCN, tend to produce overly smooth predictions that lag behind actual movements and miss directional changes.

**Solution**: Instead of predicting absolute values y[t], we predict the change Δy[t] = y[t] - y[t-1], then reconstruct the original series.

**Rationale**: 
- Delta targets are typically smaller in magnitude and easier to learn
- Models focus on predicting changes rather than absolute levels, reducing over-smoothing
- Directional accuracy improves as models learn to capture momentum and trends
- Reconstruction is straightforward: y_pred[t] = y_pred[t-1] + Δy_pred[t]

**Trade-off**: This approach works well for stationary differences but may accumulate errors over long horizons.

#### 3. Chronological Validation

**Problem**: Random train/test splits violate temporal order, allowing models to see future information during training.

**Solution**: 
- Maintain strict chronological order: training data always precedes test data
- Validation set is the last fraction of training data (e.g., last 10%)
- No shuffling or random sampling

**Rationale**: Time series data exhibits temporal dependencies. Shuffling breaks these dependencies and creates unrealistic training conditions. Chronological validation ensures models are evaluated on truly unseen future data, providing realistic performance estimates.

#### 4. Best Model Checkpointing

**Problem**: Training for fixed epochs can lead to overfitting, where validation performance degrades while training performance improves.

**Solution**: Save model weights corresponding to the lowest validation loss, not the final epoch.

**Rationale**: Validation loss is a proxy for generalization. By checkpointing the best validation model, we prevent overfitting and ensure the saved model represents the best trade-off between bias and variance.

#### 5. Unified Scikit-learn API

**Problem**: Different models have different interfaces, making comparison and experimentation difficult.

**Solution**: All models implement a consistent `.fit()` and `.predict()` interface, regardless of underlying architecture.

**Rationale**: 
- Enables easy model swapping and comparison
- Simplifies experiment orchestration
- Follows established Python ML conventions
- Facilitates integration with existing tooling

### Model-Specific Innovations

#### LSTM Architecture

**Challenge**: LSTMs can suffer from vanishing gradients and over-smoothing.

**Approach**:
- Multi-layer architecture with residual-like connections through hidden states
- Delta target prediction to focus on changes
- Reduced dropout (0.0) to prevent over-regularization
- Learning rate scheduling to fine-tune convergence
- Gradient clipping to prevent exploding gradients

**Rationale**: LSTMs excel at capturing long-term dependencies but need careful regularization. Delta targets help them focus on momentum rather than absolute levels.

#### Transformer Architecture

**Challenge**: Transformers require positional information and can struggle with directional accuracy.

**Approach**:
- Sinusoidal positional encoding (not learnable) for better generalization
- Multi-head attention (16 heads) to capture diverse temporal patterns
- Deeper architecture (5 layers) for complex pattern recognition
- **Direction Loss Component**: Combines MSE with directional accuracy loss

**Direction Loss Formula**:
```
total_loss = MSE(y_pred, y_true) + λ * direction_loss
direction_loss = -mean(sign(y_pred) * sign(y_true))
```

**Rationale**: 
- Positional encoding provides explicit temporal structure
- Direction loss explicitly encourages correct sign prediction, critical for financial applications
- Multi-head attention allows the model to attend to different temporal scales simultaneously

#### TCN Architecture

**Challenge**: TCNs need careful dilation design to capture both short and long-term patterns.

**Approach**:
- Dilated causal convolutions with exponential dilation rates (2^0, 2^1, 2^2, ...)
- Residual connections to facilitate gradient flow
- Delta target prediction to reduce smoothing
- Channel progression (128 → 256) for hierarchical feature extraction

**Rationale**: 
- Dilated convolutions exponentially increase receptive field without adding parameters
- Causal padding ensures no future information leakage
- Residual connections help with training deep networks

#### Ridge Regression Baseline

**Purpose**: Provides a linear baseline to assess whether deep learning adds value.

**Approach**: 
- Flattens sequences to use all temporal information as features
- L2 regularization to prevent overfitting
- Fast training and inference

**Rationale**: If Ridge performs comparably to deep models, the problem may be primarily linear, or deep models need architectural improvements.

### Training Strategy

#### Optimization

- **Optimizer**: Adam with configurable learning rate and weight decay
- **Learning Rate Scheduling**: ReduceLROnPlateau reduces LR when validation loss plateaus
- **Early Stopping**: Training stops if validation loss doesn't improve, preventing overfitting
- **Gradient Clipping**: Prevents exploding gradients in deep networks

#### Hyperparameter Selection

Models use different hyperparameters optimized for their architectures:
- **LSTM**: Higher learning rate (1e-3), lower dropout (0.0), delta targets
- **Transformer**: Lower learning rate (5e-4), direction loss weight (0.3), deeper network
- **TCN**: Higher learning rate (1e-3), lower dropout (0.0), delta targets
- **Ridge**: Grid search over regularization strength

### Evaluation Framework

#### Metrics

1. **RMSE (Root Mean Squared Error)**: Measures prediction accuracy
2. **MAE (Mean Absolute Error)**: Robust to outliers
3. **Directional Accuracy**: Fraction of predictions with correct sign - critical for trading applications

#### Financial Evaluation

Beyond standard metrics, we evaluate financial performance:

- **Cumulative PnL**: Theoretical profit/loss from sign-based trading strategy
- **Sharpe Ratio**: Risk-adjusted returns (annualized)
- **Hit Ratio**: Fraction of profitable trades
- **Turnover**: Trading frequency (accounts for transaction costs)

**Rationale**: For financial applications, directional accuracy and risk-adjusted returns matter more than raw prediction error.

#### Residual Analysis

Systematic error detection through:
- **Residuals vs Predicted**: Checks for heteroscedasticity (variance changes with prediction)
- **Residual Distribution**: Checks for normality and bias
- **Residual Time Series**: Checks for autocorrelation (unexplained patterns)

**Rationale**: Systematic errors indicate model misspecification or missing features.

#### Cross-Fold Aggregation

Results aggregated across multiple folds provide:
- **Mean Performance**: Average model performance
- **Standard Deviation**: Performance stability across different time periods
- **Min/Max**: Best and worst case performance
- **Robustness Assessment**: Models that perform consistently across folds are more reliable

## Key Contributions

1. **Unified Framework**: Single API for comparing multiple deep learning architectures
2. **Over-Smoothing Mitigation**: Delta target prediction and architectural improvements reduce smoothing in deep models
3. **Temporal Validation**: Proper chronological splits ensure realistic evaluation
4. **Comprehensive Evaluation**: Financial metrics and residual analysis provide deeper insights than standard regression metrics
5. **Reproducible Research**: Standardized experiment management and result storage

## Technical Highlights

- **Model Registry System**: Dynamic model instantiation enables easy experimentation
- **Hybrid Ensemble**: Weighted combination of models (Ridge 40%, Transformer 30%, LSTM 20%, TCN 10%) leverages strengths of each
- **Delta Target Reconstruction**: Automatic handling of delta predictions with proper reconstruction
- **Best Model Checkpointing**: Prevents overfitting by saving validation-optimal models
- **Comprehensive Visualization**: Metrics tables, PnL charts, and residual plots for thorough analysis

## Results Summary

The framework enables systematic comparison of models across multiple folds and horizons:

- **Ridge Baseline**: Provides linear baseline, surprisingly competitive directional accuracy
- **LSTM**: Captures temporal dependencies but benefits from delta targets to reduce smoothing
- **Transformer**: Best directional accuracy through attention mechanism and direction loss
- **TCN**: Efficient temporal modeling with dilated convolutions
- **Hybrid Ensemble**: Combines model strengths for robust predictions

## Research Applications

This framework is suitable for:
- Financial time series forecasting (returns, volatility)
- Economic indicators prediction
- Sensor data analysis
- Any multivariate time series with temporal dependencies

## Project Structure

The project follows a modular architecture:

- **Config**: Hyperparameter grids and experiment configuration
- **Models**: Implementations with unified API
- **Training**: Experiment orchestration and execution
- **Evaluation**: Metrics aggregation and analysis
- **Visualization**: Comprehensive plotting and reporting

For detailed setup instructions and code documentation, see `SETUP.md`.

## Future Directions

1. **Attention Visualization**: Understanding what temporal patterns Transformers attend to
2. **Adaptive Ensembling**: Dynamic weight adjustment based on recent performance
3. **Multi-Horizon Joint Training**: Training models to predict multiple horizons simultaneously
4. **Uncertainty Quantification**: Prediction intervals and confidence estimates
5. **Feature Importance**: Understanding which features drive predictions

## References

- Scikit-learn API design patterns
- PyTorch sequence modeling best practices
- Time series cross-validation methodologies
- Ensemble methods for forecasting
- Financial time series analysis techniques

---

**Note**: This README provides a high-level overview of the research methodology and design rationale. For implementation details, code examples, and setup instructions, please refer to `SETUP.md` and the module-specific README files in each subdirectory.
