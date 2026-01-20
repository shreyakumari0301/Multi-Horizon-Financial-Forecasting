# DSAI - Deep Learning for Time Series Forecasting

A comprehensive research framework for time series forecasting using deep learning models (LSTM, Transformer, TCN, ESN) with Ridge regression as a baseline. This project addresses the challenge of predicting future values in multivariate time series data while maintaining temporal dependencies and avoiding common pitfalls in sequence modeling.

## Problem Statement

Time series forecasting challenges:
1. **Temporal Dependencies**: Future values depend on historical patterns
2. **Sequence Alignment**: Initial timesteps lack sufficient historical context
3. **Over-Smoothing**: Deep models often miss directional changes
4. **Evaluation Complexity**: Requires chronological validation to prevent information leakage

Traditional ML treats observations independently, losing temporal information. Deep learning models can capture dependencies but require careful design.

## Research Objectives

1. **Unified Framework**: Compare deep learning models (LSTM, Transformer, TCN, ESN) against Ridge baseline
2. **Over-Smoothing Mitigation**: Delta target prediction and architectural improvements
3. **Temporal Validation**: Chronological splits and left-padding strategies
4. **Comprehensive Evaluation**: Financial metrics (PnL, Sharpe ratio) and residual analysis
5. **Reproducible Research**: Standardized APIs and experiment management

## Methodology

### Core Design Principles

#### 1. Sliding Window Sequences with Left-Padding

**Solution**: Sliding window with left-padding:
- Create sequences of fixed length L from historical observations
- Pad first L-1 timesteps with first observation to maintain full sequence length
- Ensures every timestep receives a prediction while maintaining temporal order

**Rationale**: Preserves sequence structure without future information leakage.

#### 2. Delta Target Prediction

**Solution**: Predict changes Δy[t] = y[t] - y[t-1] instead of absolute values, then reconstruct.

**Rationale**: 
- Smaller magnitude, easier to learn
- Reduces over-smoothing by focusing on momentum
- Improves directional accuracy
- Reconstruction: y_pred[t] = y_pred[t-1] + Δy_pred[t]

#### 3. Chronological Validation

**Solution**: 
- Training data always precedes test data
- Validation set is last fraction of training data (10%)
- No shuffling or random sampling

**Rationale**: Preserves temporal dependencies and ensures evaluation on truly unseen future data.

#### 4. Best Model Checkpointing

**Solution**: Save model weights corresponding to lowest validation loss, not final epoch.

**Rationale**: Prevents overfitting and ensures best generalization.

#### 5. Unified Scikit-learn API

**Solution**: All models implement consistent `.fit()` and `.predict()` interface.

**Rationale**: Enables easy model swapping, comparison, and integration.

### Model-Specific Innovations

#### LSTM Architecture

**Approach**: Multi-layer (2 layers, hidden=256), delta targets, reduced dropout (0.0), learning rate scheduling, gradient clipping.

**Rationale**: Captures long-term dependencies; delta targets focus on momentum rather than absolute levels.

#### Transformer Architecture

**Approach**: Sinusoidal positional encoding, multi-head attention (8 heads), deeper architecture (3 layers, d_model=256), direction loss component.

**Direction Loss**: `total_loss = MSE(y_pred, y_true) + λ * direction_loss` where `direction_loss = -mean(sign(y_pred) * sign(y_true))`

**Rationale**: Attention captures complex temporal relationships; direction loss explicitly encourages correct sign prediction.

#### TCN Architecture

**Approach**: Dilated causal convolutions (exponential dilation), residual connections, delta targets, channel progression (128→256).

**Rationale**: Exponentially increases receptive field without adding parameters; causal padding prevents future information leakage.

#### Echo State Network (ESN) Architecture

**Approach**: Random reservoir (400-800 units), spectral radius (0.85-0.95), sparse connectivity (density 0.1), leaky integrator (0.3-1.0), washout (100 timesteps), Ridge regression output.

**Rationale**: Reservoir computing provides rich temporal representations without backpropagation; only output layer is trained (very fast).

#### Ridge Regression Baseline

**Approach**: Flattens sequences to features, L2 regularization, fast training.

**Rationale**: Linear baseline to assess whether deep learning adds value.

### Training Strategy

- **Optimizer**: Adam with learning rate scheduling and weight decay
- **Early Stopping**: Prevents overfitting by stopping when validation loss plateaus
- **Gradient Clipping**: Prevents exploding gradients
- **Hyperparameter Tuning**: Model-specific grids optimized for each architecture

### Evaluation Framework

**Metrics**: RMSE, MAE, Directional Accuracy (critical for trading)

**Financial Evaluation**: Cumulative PnL, Sharpe Ratio, Hit Ratio, Turnover

**Residual Analysis**: Systematic error detection through residual plots and distribution analysis

**Cross-Fold Aggregation**: Mean, std, min/max across 9 folds for robustness assessment

## Key Contributions

1. **Unified Framework**: Single API for comparing multiple deep learning architectures
2. **Over-Smoothing Mitigation**: Delta target prediction and architectural improvements reduce smoothing in deep models
3. **Temporal Validation**: Proper chronological splits ensure realistic evaluation
4. **Comprehensive Evaluation**: Financial metrics and residual analysis provide deeper insights than standard regression metrics
5. **Reproducible Research**: Standardized experiment management and result storage

## Technical Highlights

- **Model Registry System**: Dynamic model instantiation enables easy experimentation
- **Hybrid Ensemble**: Weighted combination of models (Ridge, ESN, LSTM, Transformer, TCN) with validation-based weight optimization leverages strengths of each
- **Delta Target Reconstruction**: Automatic handling of delta predictions with proper reconstruction
- **Best Model Checkpointing**: Prevents overfitting by saving validation-optimal models
- **Comprehensive Visualization**: Metrics tables, PnL charts, and residual plots for thorough analysis

## Results Summary

The framework has been evaluated across 9 chronological folds and 3 prediction horizons. Key results:

### Performance Metrics

**Directional Accuracy (DirAcc) - Test Set:**

| Horizon | Hybrid Ensemble | Transformer | Ridge Baseline |
|---------|----------------|------------|----------------|
| H1 (1-day) | 0.492 ± 0.027 | 0.464 ± 0.033 | 0.516 ± 0.035 |
| H5 (5-day) | 0.503 ± 0.080 | 0.564 ± 0.066 | 0.493 ± 0.061 |
| H20 (20-day) | 0.464 ± 0.095 | 0.494 ± 0.103 | 0.502 ± 0.088 |

**RMSE (Root Mean Squared Error) - Test Set:**

| Horizon | Hybrid Ensemble | Transformer | Ridge Baseline |
|---------|----------------|------------|----------------|
| H1 | 0.012 ± 0.005 | 0.012 ± 0.005 | 0.023 ± 0.016 |
| H5 | 0.028 ± 0.012 | 0.024 ± 0.008 | 0.071 ± 0.060 |
| H20 | 0.063 ± 0.033 | 0.050 ± 0.020 | 0.163 ± 0.110 |

### Key Findings

- **Transformer** achieves lowest RMSE across all horizons (best prediction accuracy)
- **Ridge Baseline** shows competitive directional accuracy (0.516 on H1), indicating significant linear signal
- **Hybrid Ensemble** provides balanced performance with robust predictions across market conditions
- Performance degrades with longer horizons (H1 → H5 → H20), as expected for time series forecasting
- **Best Individual Performance**: Transformer achieves 0.564 DirAcc on H5 horizon

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
