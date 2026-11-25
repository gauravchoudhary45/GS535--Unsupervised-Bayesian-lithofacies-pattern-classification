# GS535--Unsupervised-Bayesian-lithofacies-pattern-classification|

**Course:** GS535  
**Project:**Unsupervised Bayesian Lithofacies-Pattern Classification Using Well-Log Data
**Domain:** Machine Learning | Bayesian Statistics | Petrophysical Analysis

---

## 1. Project Overview

This project investigates the identification of distinct lithofacies-related petrophysical patterns from well-log measurements using unsupervised machine learning and Bayesian statistical modelling.

The study combines probabilistic clustering techniques with sequential modelling to identify statistically distinct rock-response classes while incorporating vertical continuity across depth intervals.

The primary modelling approaches include:

- **Gaussian Mixture Model (GMM):** Probabilistic clustering of well-log measurements.
- **Bayesian Gaussian Mixture Model (BGMM):** Bayesian estimation of mixture parameters and uncertainty quantification.
- **Hidden Markov Model (HMM):** Sequential classification incorporating depth-dependent transition probabilities.

The overall objective is to develop a data-driven classification framework that captures petrophysical variability, quantifies classification uncertainty and produces geologically interpretable depth-wise class sequences.

## 2. Project Objectives

- Perform quality control and preprocessing of well-log measurements.
- Investigate statistical relationships between petrophysical parameters.
- Identify distinct log-response patterns using probabilistic clustering.
- Implement and compare maximum-likelihood and Bayesian estimation approaches.
- Incorporate depth continuity through Hidden Markov Models.
- Evaluate model performance using likelihood-based and stability-based metrics.
- Investigate the preservation of thin intervals and sensitivity to modelling assumptions.
- Interpret the resulting classes using their statistical and petrophysical characteristics.

## 3. Dataset Description

The analysis uses LAS-format well-log measurements from the IITB_GS535 well dataset.

**Dataset characteristics:**

| Parameter | Value |
|---|---:|
| Well name | IITB_GS535 |
| Total depth samples | 1,733 |
| Starting depth | 2153.000 m |
| Ending depth | 2416.9568 m |
| Sampling interval | 0.1524 m |
| Logged interval | Approximately 264 m |

**Well-log parameters:**

| Parameter | Description | Unit |
|---|---|---|
| GR | Gamma Ray | API |
| RHOB | Bulk Density | g/cm³ |
| NPHI | Neutron Porosity | m³/m³ |
| LLD | Deep Laterolog Resistivity | ohm·m |
| LLS | Shallow Laterolog Resistivity | ohm·m |
| DTCO | Compressional Sonic Travel Time | µs/ft |
| DTSM | Shear Sonic Travel Time | µs/ft |
| HCAL | Borehole Caliper | inch |

The selected input feature configuration primarily uses GR, RHOB, NPHI and logarithmically transformed LLD.

## 4. Data Preprocessing and Quality Control

The preprocessing stage ensures that the input measurements are suitable for probabilistic modelling.

### 4.1 Data Cleaning
- LAS data extraction and curve identification.
- Missing-value identification and handling.
- Verification of depth continuity and sampling intervals.
- Physical-range checks on well-log measurements.
- Caliper-based borehole quality control.
- Flagging of unreliable bottom-edge observations.

### 4.2 Feature Engineering

Deep resistivity measurements generally exhibit a positively skewed distribution. A logarithmic transformation is therefore applied:

\[
LLD_{\log}=\log_{10}(LLD)
\]

This reduces the influence of extreme values and makes the feature more suitable for Gaussian-based modelling.

### 4.3 Feature Standardization

Features are standardized to ensure comparable numerical scales:

\[
Z_{ij}=\frac{X_{ij}-\mu_j}{\sigma_j}
\]

Where:

- \(X_{ij}\): Original measurement.
- \(\mu_j\): Mean of feature \(j\) in the training data.
- \(\sigma_j\): Standard deviation of feature \(j\) in the training data.
- \(Z_{ij}\): Standardized measurement.

### 4.4 Training and Testing Strategy

- Data are divided into contiguous depth blocks of approximately 15 m.
- Every fourth block is reserved for testing.
- This avoids randomly splitting highly correlated neighbouring depth observations.
- Standardization parameters are estimated using training observations.

### 4.5 Quality Control Summary

| Parameter | Result |
|---|---:|
| Total samples | 1,733 |
| Missing samples | 0 |
| Flagged samples | 10 |
| Flagged percentage | 0.58% |
| Retained samples | 1,723 |

## 5. Exploratory Data Analysis (EDA)

Exploratory analysis is performed to understand the statistical behaviour and relationships between the selected well-log measurements.

**Analyses performed:**

- Distribution and histogram analysis of individual log curves.
- Feature correlation matrix.
- Crossplots between petrophysical parameters.
- Depth-wise variation of well-log measurements.
- Investigation of relationships between GR, RHOB, NPHI and resistivity.

**Important observations:**

- Gamma Ray provides information about radioactive mineral content and shale-related response.
- RHOB represents bulk formation density.
- NPHI provides neutron-derived porosity response.
- Resistivity measurements help characterize formation electrical properties.
- Correlations between these measurements provide useful information for distinguishing petrophysical response patterns.

The EDA stage provides the basis for selecting the input features used in subsequent clustering models.

## 6. Gaussian Mixture Model (GMM)

### 6.1 Model Description

A Gaussian Mixture Model represents the overall data distribution as a weighted combination of multiple Gaussian distributions.

For an observation \(X\):

\[
p(X)=\sum_{k=1}^{K}\pi_k
\mathcal{N}(X\mid\mu_k,\Sigma_k)
\]

Where:

- \(K\): Number of Gaussian components.
- \(\pi_k\): Mixing coefficient of component \(k\).
- \(\mu_k\): Mean vector.
- \(\Sigma_k\): Covariance matrix.
- \(\mathcal{N}\): Multivariate Gaussian distribution.

The mixing coefficients satisfy:

\[
\sum_{k=1}^{K}\pi_k=1
\]

\[
\pi_k\geq 0
\]

Each Gaussian component represents a statistical group of similar petrophysical measurements.

### 6.2 Expectation-Maximization (EM) Algorithm

The GMM parameters are estimated using the Expectation-Maximization algorithm.

**E-step: Responsibility calculation**

The probability that observation \(X_i\) belongs to component \(k\) is:

\[
\gamma_{ik}=
\frac{\pi_k\mathcal{N}(X_i\mid\mu_k,\Sigma_k)}
{\sum_{j=1}^{K}\pi_j\mathcal{N}(X_i\mid\mu_j,\Sigma_j)}
\]

**M-step: Parameter updates**

Effective component membership:

\[
N_k=\sum_{i=1}^{N}\gamma_{ik}
\]

Updated mixing coefficient:

\[
\pi_k=\frac{N_k}{N}
\]

Updated mean:

\[
\mu_k=\frac{1}{N_k}
\sum_{i=1}^{N}\gamma_{ik}X_i
\]

Updated covariance:

\[
\Sigma_k=
\frac{1}{N_k}
\sum_{i=1}^{N}\gamma_{ik}
(X_i-\mu_k)(X_i-\mu_k)^T
\]

The E-step and M-step are repeated until convergence.

### 6.3 Model Selection

Different component counts are evaluated using:

- Akaike Information Criterion (AIC).
- Bayesian Information Criterion (BIC).
- Training log-likelihood.
- Held-out testing log-likelihood.

\[
AIC=2p-2\ln(\hat L)
\]

\[
BIC=p\ln(N)-2\ln(\hat L)
\]

Where \(p\) is the number of estimated parameters and \(\hat L\) is the maximized likelihood.

Lower AIC and BIC indicate a preferable balance between model fit and complexity.

### 6.4 Manual EM Implementation

A manual EM implementation is also developed to examine the underlying parameter estimation process.

- Responsibility estimation.
- Gaussian probability calculations.
- Mean and covariance updates.
- Mixture weight updates.
- Numerical stabilization using log-sum-exp.
- Convergence monitoring through log-likelihood.

The implementation is compared with Scikit-learn's GMM results.

## 7. Bayesian Gaussian Mixture Model

### 7.1 Motivation

Conventional GMM estimates parameters using maximum likelihood. However, it provides limited information about parameter uncertainty.

Bayesian Gaussian Mixture Modelling introduces prior distributions over model parameters and estimates their posterior distributions.

Using Bayes' theorem:

\[
p(\theta\mid X)=
\frac{p(X\mid\theta)p(\theta)}
{p(X)}
\]

Where:

- \(p(\theta)\): Prior distribution.
- \(p(X\mid\theta)\): Likelihood.
- \(p(\theta\mid X)\): Posterior distribution.

### 7.2 Bayesian Formulation

The model incorporates prior distributions for:

- Mixture weights.
- Component means.
- Covariance-related parameters.

A Dirichlet prior is used for mixture weights:

\[
\boldsymbol{\pi}\sim
Dirichlet(\alpha_1,\alpha_2,\ldots,\alpha_K)
\]

This allows prior information to influence the estimated component proportions.

### 7.3 Variational Bayesian Inference

The Bayesian Gaussian Mixture is estimated using variational inference.

The objective is to approximate the posterior distribution with a tractable distribution \(q(\theta)\).

The variational objective is based on maximizing the Evidence Lower Bound (ELBO):

\[
ELBO=
E_q[\ln p(X,\theta)]
-
E_q[\ln q(\theta)]
\]

This provides an approximate posterior over model parameters.

### 7.4 Posterior Analysis

The Bayesian analysis includes:

- Estimated class proportions.
- Posterior parameter estimates.
- Credible intervals.
- Class membership probabilities.
- Classification entropy.
- Sensitivity to prior assumptions.

Classification uncertainty is measured using entropy:

\[
H_i=-\sum_{k=1}^{K}
P(C_k\mid X_i)\ln P(C_k\mid X_i)
\]

Lower entropy indicates more confident classification, whereas higher entropy indicates greater uncertainty.

## 8. Hidden Markov Model (HMM)

### 8.1 Motivation

Independent clustering methods classify each depth observation separately.

However, geological formations generally exhibit vertical continuity. Consecutive depth observations are therefore expected to show some persistence in their petrophysical characteristics.

The HMM introduces sequential dependence between neighbouring class assignments.

### 8.2 Hidden States and Observations

The hidden states represent the unknown statistical classes:

\[
S_t\in\{0,1,2,3,4\}
\]

The observations represent the measured well-log features:

\[
X_t=[GR,RHOB,NPHI,LLD_{\log}]
\]

The actual class is hidden, while the well-log measurements are observable.

### 8.3 Transition Probabilities

The transition matrix describes the probability of moving from one class to another:

\[
A_{ij}=P(S_t=j\mid S_{t-1}=i)
\]

Each row satisfies:

\[
\sum_{j=1}^{K}A_{ij}=1
\]

High diagonal probabilities indicate a strong tendency to remain in the same class between consecutive depth samples.

### 8.4 Emission Probabilities

The observed measurements are modelled using class-dependent Gaussian distributions:

\[
P(X_t\mid S_t=k)
=
\mathcal{N}(X_t\mid\mu_k,\Sigma_k)
\]

This connects the observed well-log measurements with the hidden statistical classes.

### 8.5 Forward-Backward Algorithm

The Forward-Backward algorithm calculates smoothed hidden-state probabilities:

\[
\gamma_t(k)=
P(S_t=k\mid X_{1:T})
\]

It combines information from observations before and after the current depth sample.

### 8.6 Viterbi Algorithm

The Viterbi algorithm identifies the most probable complete hidden-state sequence:

\[
S_{1:T}^{*}
=
\arg\max_{S_{1:T}}
P(S_{1:T}\mid X_{1:T})
\]

This produces the final depth-wise class sequence.

### 8.7 Baum-Welch Estimation

Baum-Welch is an EM-based algorithm used to estimate HMM parameters.

**E-step:**
- Calculate hidden-state probabilities.
- Estimate expected transition counts.

**M-step:**
- Update transition probabilities.
- Update Gaussian emission means.
- Update covariance matrices.
- Update initial-state probabilities.

### 8.8 MLE HMM and Bayesian-Style HMM

Two HMM configurations are investigated:

- **MLE HMM:** Estimates transition probabilities using maximum-likelihood-based expected transition counts.
- **Bayesian-style HMM:** Incorporates Dirichlet prior information into transition probability estimation.

The two approaches are compared using held-out likelihood and depth-continuity characteristics.

### 8.9 Expected Dwell Length

The expected number of consecutive samples spent in a class is:

\[
E(L_k)=\frac{1}{1-A_{kk}}
\]

This provides an estimate of class persistence along the depth axis.

For a sampling interval of 0.1524 m:

\[
Thickness_k=E(L_k)\times0.1524
\]

This represents an approximate statistical persistence length, not necessarily the actual geological bed thickness.

## 9. Model Results and Comparison

### 9.1 Model Selection Results

The models are evaluated for component counts \(K=2,3,4,5,6\).

| K | GMM Test LL | Bayesian GMM Test LL | HMM Test LL |
|---:|---:|---:|---:|
| 2 | -3.827 | -3.879 | -3.339 |
| 3 | -3.560 | -3.583 | -2.992 |
| 4 | -3.587 | -3.587 | -2.642 |
| 5 | -3.530 | -3.524 | -2.431 |
| 6 | -3.794 | -3.744 | -2.759 |

**Observations:**
- Five components provide the highest held-out likelihood for all three model families among the tested configurations.
- Increasing the number of components beyond five does not improve generalization.
- HMM provides better held-out likelihood than independent GMM and Bayesian GMM in the selected configuration.

### 9.2 Final Model Comparison

| Model | Training LL | Testing LL |
|---|---:|---:|
| GMM | -3.131 | -3.502 |
| HMM (MLE) | -2.120 | -2.543 |
| HMM (Bayesian-style) | -2.132 | -2.431 |

Higher log-likelihood values indicate better fit to the evaluated observations.

### 9.3 Bayesian GMM Class Weights

| Class | Estimated Weight |
|---|---:|
| 0 | 0.414 |
| 1 | 0.246 |
| 2 | 0.240 |
| 3 | 0.085 |
| 4 | 0.015 |

The estimated weights indicate the relative contribution of each Gaussian component to the fitted mixture distribution.

### 9.4 Final HMM Class Statistics

| Class | Samples | Share (%) | Mean GR | Mean RHOB | Mean NPHI |
|---|---:|---:|---:|---:|---:|
| 0 | 612 | 35.5 | 42.21 | 2.556 | 0.066 |
| 1 | 405 | 23.5 | 31.90 | 2.609 | 0.035 |
| 2 | 453 | 26.3 | 44.60 | 2.659 | 0.008 |
| 3 | 225 | 13.1 | 33.86 | 2.456 | 0.153 |
| 4 | 28 | 1.6 | 70.98 | 2.677 | 0.208 |

The classes exhibit different petrophysical response characteristics.

These are statistical classes inferred from well-log measurements and should not be interpreted as independently verified lithological categories without geological reference information.

## 10. Model Validation and Sensitivity Analysis

### 10.1 Posterior Predictive Checks

Replicated observations are generated from the fitted model and compared with the held-out observations.

The checks include:
- Feature means.
- Feature standard deviations.
- Feature correlations.

The comparison indicates differences between observed and replicated feature distributions, suggesting limitations in reproducing the characteristics of held-out depth blocks.

### 10.2 Vertical Autocorrelation

Lag-1 autocorrelation measures similarity between neighbouring depth observations:

\[
r_1=Corr(X_t,X_{t+1})
\]

| Feature | Observed | HMM Median |
|---|---:|---:|
| GR | 0.950 | 0.340 |
| RHOB | 0.957 | 0.605 |
| NPHI | 0.974 | 0.640 |
| logLLD | 0.970 | 0.447 |

The observed measurements exhibit strong vertical correlation. Although HMM improves class continuity, it does not fully reproduce the observed autocorrelation.

### 10.3 Block Bootstrap Stability

Block bootstrap is performed using 100 resampling iterations.

The use of depth blocks rather than individual samples helps preserve local depth dependence.

**Adjusted Rand Index:**

\[
ARI=0.47
\]

5th–95th percentile interval:

\[
[0.30,\ 0.70]
\]

**Class-wise Jaccard similarity:**

| Class | Mean Jaccard |
|---|---:|
| 0 | 0.61 |
| 1 | 0.67 |
| 2 | 0.51 |
| 3 | 0.51 |
| 4 | 0.53 |

These results indicate moderate clustering stability, with differences in the stability of individual classes.

### 10.4 Feature Selection Sensitivity

The effect of different feature combinations is investigated by comparing their classifications with the reference feature configuration.

| Feature Configuration | ARI |
|---|---:|
| Set A | 1.000 |
| Set B | 0.399 |
| Without GR | 0.393 |
| Without logLLD | 0.801 |
| Without NPHI | 0.610 |
| Plus DTCO | 0.323 |

The results demonstrate that feature selection has a considerable influence on the inferred class assignments.

### 10.5 Thin-Bed Preservation

The classification behaviour is examined over the selected depth interval of 2315–2319 m.

| Model | Total Runs | Runs Below 10 Samples |
|---|---:|---:|
| GMM | 142 | 84 |
| HMM (MLE) | 95 | 42 |
| HMM (Bayesian-style) | 89 | 39 |

The HMM reduces short class sequences.

The investigated thin-bed interval is preserved in both HMM configurations.

### 10.6 Transition-Prior Sensitivity

The influence of stronger self-transition priors is examined.

| Additional Diagonal Prior | Total Runs | Short Runs |
|---:|---:|---:|
| 0 | 89 | 39 |
| 10 | 89 | 39 |
| 100 | 86 | 37 |
| 1000 | 76 | 27 |

Increasing the diagonal prior strengthens the preference for remaining in the same class, resulting in smoother classifications.

However, excessive smoothing can suppress genuine thin geological intervals.

## 11. Final Classification and Petrophysical Interpretation

The final classification is generated using the Bayesian-style HMM.

The Viterbi algorithm provides the most probable class sequence, while the Forward-Backward algorithm provides smoothed class membership probabilities.

**Final outputs include:**

- Depth-wise HMM classification.
- Individual well-log tracks alongside predicted classes.
- Class membership probability tracks.
- Maximum posterior class probability.
- Petrophysical crossplots coloured by predicted classes.
- Class-wise statistical summaries.
- Continuous class intervals and approximate thicknesses.

### Class Interpretation

- **Class 0:** Represents the largest inferred group, accounting for 35.5% of the retained classified samples.
- **Class 1:** Represents 23.5% of samples, with comparatively lower mean GR and NPHI.
- **Class 2:** Characterized by the highest mean RHOB and lowest mean NPHI among the five classes.
- **Class 3:** Characterized by the lowest mean RHOB and relatively high NPHI.
- **Class 4:** A relatively rare group (1.6%), with high mean GR and NPHI.

These descriptions represent statistical petrophysical characteristics rather than definitive lithological assignments.

## 12. Conclusion

- Probabilistic clustering successfully identifies five distinct statistical petrophysical response groups.
- Bayesian GMM introduces prior information and provides a framework for parameter uncertainty estimation.
- HMM improves depth-wise classification continuity by incorporating transition probabilities.
- Five components provide the highest held-out likelihood among the tested configurations.
- The investigated thin-bed interval is retained under the selected HMM configurations.
- Bootstrap analysis indicates moderate clustering stability.
- Posterior predictive checks and autocorrelation comparisons reveal limitations in reproducing the full statistical and vertical structure of the observed well logs.

The developed workflow demonstrates the application of unsupervised machine learning, Bayesian inference and sequential probabilistic modelling for data-driven well-log classification.

## 13. Technologies and Libraries

| Technology | Application |
|---|---|
| Python | Computational environment |
| NumPy | Numerical computations |
| Pandas | Data handling and analysis |
| SciPy | Statistical distributions and numerical algorithms |
| Scikit-learn | GMM and Bayesian GMM |
| Matplotlib | Data visualization |
| lasio | LAS-format well-log data extraction |
| Jupyter Notebook | Project implementation |

## 14. Repository Structure

```text
Project-4/
│
├── Main.ipynb
│
├── README.md
│
├── tracks.png
├── crossplots.png
├── caliper_qc.png
│
├── val_K_sensitivity.png
├── final_logs_classes.png
├── crossplots_by_class.png
│
├── Other generated result plots
│
└── Supporting result visualizations
```

**Note:** The original LAS dataset used for the analysis is not included in this repository. The notebook and uploaded plots document the project implementation and generated results.

---

