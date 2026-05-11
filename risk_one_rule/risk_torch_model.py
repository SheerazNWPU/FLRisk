from torch.distributions import Normal
from torch.nn.functional import softmax, sigmoid
from torch.nn.modules.loss import _Loss

from risk_one_rule import similarity_based_feature as sbf
from common import config
import numpy as np
from . import torch_learn_weights as torchlearn
from collections import Counter
import torch
import torch.distributed as dist
from torch import nn, optim
from typing import List, Optional, Union
# import torch
# from torch.nn.modules.loss import _Loss
# from torch.distributions.normal import Normal
import os
import torch
from scipy import stats
from scipy.stats import spearmanr
from torch.distributions import MultivariateNormal
#from torch.distributions.multivariate_normal import MultivariateNormal
cfg = config.Configuration(config.global_data_selection, config.global_deep_learning_selection)
class_num = cfg.get_class_num()

device = [0,1]  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device2 = [0,1]


class RiskTorchModel(object):
    def __init__(self):
        self.prob_interval_boundary_pts = sbf.get_equal_intervals(0.0, 1.0, cfg.interval_number_4_continuous_value)
        self.prob_dist_mean = None
        self.prob_dist_variance = None
        # parameters for risk model
        self.learn_weights = None
        self.rule_w = None
        self.func_params = [0.5, 0.5, 1]
        self.rule_var = None
        self.machine_var = None
        self.a = 0.0
        self.b = 1.0
        self.learn_confidence = 0.9
        self.match_value = None
        self.unmatch_value = None
        # train data
        self.train_data = None
        # validation data
        self.validation_data = None
        # test data
        self.test_data = None
        self.model = None
        self.init_rule_mu = None
        self.init_machine_mu = None

    # def train(self, train_machine_probs, valida_machine_probs, train_machine_mul_probs, valida_machine_mul_probs,):
    def train(self, train_machine_probs, valida_machine_probs, test_machine_probs, train_machine_mul_probs,
              valida_machine_mul_probs, test_machine_mul_probs,
              train_labels=None, val_labels=None, test_labels=None, epoch=""):
        # use new classifier output probabilities.

        self.train_data.update_machine_info(train_machine_probs, train_labels)
        self.validation_data.update_machine_info(valida_machine_probs, val_labels)
        self.test_data.update_machine_info(test_machine_probs, test_labels)

        self.train_data.update_machine_mul_info(train_machine_mul_probs)
        self.validation_data.update_machine_mul_info(valida_machine_mul_probs)
        self.test_data.update_machine_mul_info(test_machine_mul_probs)
        '''
        prob_interval_2_ids, class_label = sbf.get_mul_continuous_interval_to_ids(self.train_data.id_2_mul_probs,
                                                                                  class_num,
                                                                                  self.train_data.data_ids,
                                                                                  self.prob_interval_boundary_pts)

        self.prob_dist_mean, self.prob_dist_variance = sbf.calculate_mul_similarity_interval_distributions(
            prob_interval_2_ids,
            class_label,
            self.train_data.id_2_true_labels,
            cfg.minimum_observation_num)
        '''
        
        self.prob_dist_mean = ([1.] * cfg.interval_number_4_continuous_value) * (
          np.linspace(0, 1, cfg.interval_number_4_continuous_value + 1)[1:]) / 2

        # print(self.prob_dist_variance.shape, self.prob_dist_mean.shape)
        # print(self.prob_dist_variance)
        # print('---------mean')
        # print(self.prob_dist_mean)

        init_rule_mu = self.train_data.mu_vector
        init_machine_mu = self.prob_dist_mean
        init_rule_mu[init_rule_mu < 0] = 0
        init_machine_mu[init_machine_mu < 0] = 0
        self.init_rule_mu = init_rule_mu
        self.init_machine_mu = init_machine_mu
        # print('------update')
        # print(init_rule_mu)
        # print(init_machine_mu)

        # update the probability feature of training data
        self.train_data.update_probability_feature(self.prob_interval_boundary_pts, )
        # update the probability feature of validation data
        self.validation_data.update_probability_feature(self.prob_interval_boundary_pts, )
        self.test_data.update_probability_feature(self.prob_interval_boundary_pts)
        self.train_data = None
        
        _feature_activation_matrix = np.concatenate(
            (np.array(self.validation_data.get_rule_activation_matrix()),
             np.array(self.validation_data.get_prob_activation_matrix())), axis=2)

        max_w = 1.0 / np.max(np.sum(_feature_activation_matrix, axis=2))
        class_count = Counter(self.validation_data.machine_labels.reshape(-1))
        # dist.init_process_group(backend='gloo', init_method='env://')
        print('load model')
        #self.risk_weight_learn = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, requires_grad=True))
        #model = torchlearn.RiskModel(self.validation_data.get_risk_mean_X_discrete().shape[2], max_w).to(device[0])
        model = torchlearn.RiskModel(self.validation_data.get_risk_mean_X_discrete().shape[2], max_w, len(init_machine_mu), class_num, self.validation_data.true_labels).to(device[0])
        # model = torch.nn.DataParallel(model)
        self.model = model
        # for i in self.model.parameters():
        #     print(i)

        func_params, rule_w, rule_var, machine_var = torchlearn.train(self.model, self.validation_data, self.test_data,
                                                                      init_rule_mu, init_machine_mu, epoch)
        self.func_params = func_params
        self.rule_w = rule_w
        self.rule_var = rule_var
        self.machine_var = machine_var

    def predict(self, test_machine_probs, test_machine_mul_probs):
        # use new classifier output probabilities.

        # self.test_data.update_machine_info(test_machine_probs)
        # self.test_data.update_machine_mul_info(test_machine_mul_probs)
        # # update the probability feature of training data
        # self.test_data.update_probability_feature(self.prob_interval_boundary_pts,
        #                                           self.prob_dist_mean,
        #                                           self.prob_dist_variance)
        # self.test_data = self.validation_data
        results = torchlearn.predict(self.model, self.test_data, 0, self.init_rule_mu, self.init_machine_mu, True)
        predict_probs = results
        self.test_data.risk_values = predict_probs


def my_truncated_normal_ppf(confidence, a, b, mean, stddev, class_num):
    x = torch.zeros_like(mean)
    mean = torch.reshape(mean, (-1, 1))
    stddev = torch.reshape(stddev, (-1, 1))
    norm = Normal(mean, stddev)
    _nb = norm.cdf(b)
    _na = norm.cdf(a)
    _sb = 1. - norm.cdf(b)
    _sa = 1. - norm.cdf(a)

    y = torch.where(a > 0,
                    -norm.icdf(confidence * _sb + _sa * (1.0 - confidence)),
                    norm.icdf(confidence * _nb + _na * (1.0 - confidence)))
    return torch.reshape(y, (-1, class_num))


def gaussian_function(a, b, c, x):
    _part = (- torch.div((x - b) ** 2, 2.0 * (c ** 2)))
    _f = -torch.exp(_part) + a + 1.0
    return _f

## Updated Sheeraz 7/6/2024
'''
def my_truncated_normal_ppf(confidence, a, b, mean, stddev, class_num):
    # Reshape mean and stddev
    mean = torch.reshape(mean, (-1, 1))
    stddev = torch.reshape(stddev, (-1, 1))
      # Check for NaN or Inf values in mean and stddev
    #print('mean')
    #print(mean)
    #print('stdev')
    #print(stddev)
    norm = Normal(mean, stddev)
    
    # Compute values for CDF
    _nb = norm.cdf(b)
    _na = norm.cdf(a)
    _sb = 1. - norm.cdf(b)
    _sa = 1. - norm.cdf(a)

    # Compute values for icdf input
    icdf_input_1 = confidence * _sb + _sa * (1.0 - confidence)
    icdf_input_2 = confidence * _nb + _na * (1.0 - confidence)

    # Clip the values to avoid numerical instability
    icdf_input_1 = torch.clamp(icdf_input_1, min=1e-6, max=1-1e-6)
    icdf_input_2 = torch.clamp(icdf_input_2, min=1e-6, max=1-1e-6)

    # Compute the final output using icdf
    y = torch.where(a > 0,
                    -norm.icdf(icdf_input_1),
                    norm.icdf(icdf_input_2))

    return torch.reshape(y, (-1, class_num))
'''
'''
def my_truncated_normal_ppf(confidence, a, b, mean, stddev):
    x = torch.zeros_like(mean)
    mean = torch.reshape(mean, (-1, 1))
    stddev = torch.reshape(stddev, (-1, 1))
    norm = Normal(mean, stddev)
    _nb = norm.cdf(b)
    _na = norm.cdf(a)
    _sb = 1. - norm.cdf(b)
    _sa = 1. - norm.cdf(a)

    y = torch.where(a > 0,
                    -norm.icdf(confidence * _sb + _sa * (1.0 - confidence)),
                    norm.icdf(confidence * _nb + _na * (1.0 - confidence)))
    return torch.reshape(y, (-1, class_num))
'''
def ensure_positive_definite(cov_matrix, epsilon=1e-6):
    """
    Ensure that a matrix is positive definite by trying Cholesky decomposition.
    If the matrix is not positive definite, adjust it by increasing the diagonal elements.
    
    Args:
        cov_matrix (torch.Tensor): The covariance matrix.
        epsilon (float): Small value added to the diagonal if the matrix is not positive definite.
    
    Returns:
        torch.Tensor: A positive definite covariance matrix.
    """
    try:
        # Try Cholesky decomposition to check if the matrix is positive definite
        torch.linalg.cholesky(cov_matrix)
    except RuntimeError:
        # If decomposition fails, the matrix is not positive definite
        # Increase diagonal elements by epsilon until it works
        diag_addition = torch.eye(cov_matrix.size(0), device=cov_matrix.device) * epsilon
        cov_matrix += diag_addition
        print(f"Covariance matrix adjusted by adding {epsilon} to diagonal elements for stability.")
    
    return cov_matrix

def calculate_evar(simulations, alpha, max_lambda=10.0, num_steps=1000):
    """
    Calculate the Entropic Value at Risk (EVaR) for a given set of simulations.
    
    :param simulations: A tensor of simulated outcomes.
    :param alpha: Confidence level for EVaR (e.g., 0.95).
    :param max_lambda: Maximum value of lambda for the optimization process.
    :param num_steps: Number of steps for lambda optimization.
    :return: EVaR value.
    """
    # Initialize lambda and the step size
    lambdas = torch.linspace(1e-5, max_lambda, num_steps)
    evars = []

    # Calculate EVaR for each lambda
    for lam in lambdas:
        exp_moment = torch.exp(lam * simulations).mean(dim=0)  # Exponential moment of the losses
        evar_lam = (1 / lam) * (torch.log(exp_moment) - torch.log(torch.tensor(alpha)))
        evars.append(evar_lam)
    
    # Return the minimum EVaR over all lambda values
    evars = torch.stack(evars, dim=0)
    evar = torch.min(evars, dim=0).values
    return evar

'''
def my_truncated_normal_ppf(confidence, a, b, mean, stddev, label_num, machine_mus, num_simulations=10000):
    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(machine_mus)
    device = stddev.device
    mean = mean.to(device)
    corr_matrix = corr_matrix.to(device)
    # Calculate covariance matrix from stddev (assuming diagonal covariance)
    cov_matrix = calculate_covariance_matrix(stddev[0], corr_matrix)
    
    # Define multivariate normal distribution with mean and covariance
    norm = MultivariateNormal(mean, covariance_matrix=cov_matrix)

    ## Ensure num_simulations is an integer
    #if isinstance(num_simulations, torch.Tensor):
    #    num_simulations = num_simulations[0][0].item() 

    # Monte Carlo simulation to generate potential outcomes
    simulations = norm.sample((int(num_simulations),))  # Ensure num_simulations is passed as an integer
    
    # Apply truncation bounds (a and b) to the simulations
    simulations = torch.clamp(simulations, min=a, max=b)
    
    # Ensure confidence is of the same dtype as simulations
    confidence = confidence.to(simulations.dtype)

    # VaR Calculation: find the quantile for the given confidence level
    VaR = torch.quantile(simulations, confidence, dim=0)

    # ES Calculation: average outcomes in the tail beyond VaR
    tail_mask = simulations >= VaR.unsqueeze(0)
    ES = simulations[tail_mask].mean(dim=0)
    #EVaR = calculate_evar(simulations, confidence)
    # Combine VaR and ES: weight them to balance thresholding vs. average risk severity
    hybrid_risk = 0.5 * VaR + 0.5 * ES

    return VaR
'''
'''
def my_truncated_normal_ppf(confidence, a, b, mean, stddev, label_num, machine_mus, num_simulations=10000):
    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(machine_mus)
    #mean = torch.reshape(mean, (-1, 1))
    #stddev = torch.reshape(stddev, (-1, 1))
    device = stddev.device
    mean = mean.to(device)
    corr_matrix = corr_matrix.to(device)

    # Calculate covariance matrix from stddev (assuming diagonal covariance)
    cov_matrix = calculate_covariance_matrix(stddev[0], corr_matrix)
    
    # Define multivariate normal distribution with mean and covariance
    norm = MultivariateNormal(mean, covariance_matrix=cov_matrix)
    
    # Monte Carlo simulation to generate potential outcomes
    simulations = norm.sample((int(num_simulations),))
    
    # Apply truncation bounds (a and b) to the simulations
    simulations = torch.clamp(simulations, min=a, max=b)
    
    # Ensure confidence is of the same dtype as simulations
    confidence = confidence.to(simulations.dtype)

    # Calculate quantiles for truncation bounds
    q_high = torch.quantile(simulations, 1.0 - confidence, dim=0)  # high quantile
    q_low = torch.quantile(simulations, confidence, dim=0)         # low quantile

    # VaR calculation using torch.where
    VaR = torch.where(
        a > 0,
        -q_high,
        q_low
    )
    
    return torch.reshape(VaR, (-1, label_num))
'''

def pairwise_correlation(machine_mus: torch.Tensor) -> torch.Tensor:
    """Calculate the pairwise correlation matrix across labels (or features if needed).
    
    Args:
        machine_mus: Input tensor [num_samples, num_features, num_labels] or [num_samples, num_labels]
    
    Returns:
        torch.Tensor: Pairwise correlation matrix across labels
    """
    # If input is 2D, add an extra dimension for features (i.e., [num_samples, 1, num_labels])
    if machine_mus.dim() == 2:
        machine_mus = machine_mus.unsqueeze(1)  # Add a dimension for features

    # Ensure input is 3D after adjustment
    if machine_mus.dim() != 3:
        raise ValueError(f"Expected input tensor to have 3 dimensions, got {machine_mus.dim()}")

    # Compute correlation across labels
    num_labels = machine_mus.shape[-1]
    
    # Calculate pairwise correlations
    machine_mus_flat = machine_mus.reshape(-1, num_labels)  # Flatten features and samples
    corr_matrix = torch.corrcoef(machine_mus_flat.T)
    
    return corr_matrix


def shrinkage_correlation(machine_mus: torch.Tensor, shrinkage_factor: float = 0.2) -> torch.Tensor:
    """Calculate correlation matrix with shrinkage over labels (not features).
    
    Args:
        machine_mus: Input tensor [num_samples, num_features, num_labels]
        shrinkage_factor: Shrinkage intensity (0-1)
        
    Returns:
        torch.Tensor: Shrunk correlation matrix across labels
    """
    # Ensure input is 3D: [num_samples, num_features, num_labels]
    if machine_mus.dim() != 3:
        raise ValueError(f"Expected input tensor to have 3 dimensions, got {machine_mus.dim()}")
    
    # Compute the pairwise correlation across labels
    pairwise_corr = pairwise_correlation(machine_mus)
    
    # Apply shrinkage to the correlation matrix
    target = torch.eye(pairwise_corr.shape[0], device=pairwise_corr.device)
    return (1 - shrinkage_factor) * pairwise_corr + shrinkage_factor * target


def adaptive_correlation(machine_mus: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """Calculate correlation matrix with thresholding across labels (not features).
    
    Args:
        machine_mus: Input tensor [num_samples, num_features, num_labels]
        threshold: Correlation threshold to apply
        
    Returns:
        torch.Tensor: Thresholded correlation matrix across labels
    """
    # Ensure input is 3D: [num_samples, num_features, num_labels]
    if machine_mus.dim() != 3:
        raise ValueError(f"Expected input tensor to have 3 dimensions, got {machine_mus.dim()}")
    
    # Compute pairwise correlation across labels
    pairwise_corr = pairwise_correlation(machine_mus)
    
    # Apply threshold to the correlation matrix
    mask = torch.abs(pairwise_corr) < threshold
    pairwise_corr[mask] = 0
    return pairwise_corr



def adaptive_correlation(machine_mus: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """Calculate correlation matrix with thresholding
    
    Args:
        machine_mus: Input tensor
        threshold: Correlation threshold
        
    Returns:
        torch.Tensor: Thresholded correlation matrix
    """
    pairwise_corr = pairwise_correlation(machine_mus)
    mask = torch.abs(pairwise_corr) < threshold
    pairwise_corr[mask] = 0
    return pairwise_corr

def check_normality(data, min_samples=8):
    """
    Check normality of data using D'Agostino and Pearson's test.
    """
    num_labels = data.shape[-1]
    is_normal = []
    p_values = []
    
    for i in range(num_labels):
        # Convert data to CPU if it's on GPU
        label_data = data[..., i].cpu().numpy() if data.is_cuda else data[..., i].numpy()
        
        # Check if we have enough samples
        if label_data.shape[0] < min_samples:
            # If not enough samples, assume normal distribution
            print(f"Warning: Label {i} has {label_data.shape[0]} samples (less than {min_samples}). Assuming normal distribution.")
            is_normal.append(True)
            p_values.append(1.0)
        else:
            try:
                # Perform normality test
                _, p_value = stats.normaltest(label_data, axis=0)
                is_normal.append(all(p > 0.05 for p in p_value))
                p_values.append(p_value)
            except Exception as e:
                print(f"Warning: Normality test failed for label {i}: {str(e)}. Assuming normal distribution.")
                is_normal.append(True)
                p_values.append(1.0)
    
    return is_normal, p_values


def get_best_ppf_method(data):
    """
    Determine the best PPF method based on data characteristics
    """
    num_labels = data.shape[-1]
    methods = []
    
    try:
        is_normal_per_label, p_values_per_label = check_normality(data)
    except Exception as e:
        print(f"Warning: Normality test failed: {str(e)}. Using default methods.")
        # Default to a mix of methods
        return ['univariate'] * num_labels
    
    for label_idx in range(num_labels):
        if is_normal_per_label[label_idx]:
            # If data appears normal, use univariate method
            methods.append('univariate')
        else:
            # If non-normal, use empirical or mixture method
            # Could add more sophisticated selection logic here
            methods.append('empirical')
    
    return methods

def calculate_covariance_matrix(stddev_matrix: torch.Tensor, corr_matrix: torch.Tensor) -> torch.Tensor:
    """
    Calculates the covariance matrix from standard deviations and correlations.
    Handles various input dimension cases safely.

    Args:
        stddev_matrix (torch.Tensor): Tensor of standard deviations [batch_size, num_features] or [batch_size, num_features, 1]
        corr_matrix (torch.Tensor): Tensor of correlations [batch_size, num_features, num_features] or [num_windows, num_features, num_features]

    Returns:
        torch.Tensor: Covariance matrix [batch_size, num_features, num_features]
    """
    # First ensure stddev_matrix is at least 2D
    if stddev_matrix.dim() == 1:
        stddev_matrix = stddev_matrix.unsqueeze(0)
    
    # Remove any trailing singleton dimensions from stddev_matrix
    while stddev_matrix.dim() > 2 and stddev_matrix.size(-1) == 1:
        stddev_matrix = stddev_matrix.squeeze(-1)
    
    # Ensure corr_matrix is 3D
    if corr_matrix.dim() == 2:
        corr_matrix = corr_matrix.unsqueeze(0)
    
    # Get the correct dimensions
    batch_size = stddev_matrix.size(0)
    stddev_features = stddev_matrix.size(1)
    corr_features = corr_matrix.size(-1)
    
    # Verify feature dimensions match
    if stddev_features != corr_features:
        raise ValueError(f"Feature dimension mismatch: stddev has {stddev_features} features, "
                        f"correlation has {corr_features} features")
    
    # Handle case where correlation matrix has more entries (from windows)
    if corr_matrix.size(0) != batch_size:
        # Calculate windows per batch
        windows_per_batch = corr_matrix.size(0) // batch_size
        if windows_per_batch > 0:
            # Reshape and average windows
            corr_matrix = corr_matrix.view(batch_size, windows_per_batch, corr_features, corr_features)
            corr_matrix = corr_matrix.mean(dim=1)
        else:
            # If we have fewer windows than batch size, repeat the correlation matrix
            corr_matrix = corr_matrix.expand(batch_size, -1, -1)
    
    # Reshape stddev_matrix for matrix multiplication
    stddev_matrix = stddev_matrix.unsqueeze(-1)  # [batch_size, num_features, 1]
    
    # Compute covariance matrix
    stddev_outer = stddev_matrix @ stddev_matrix.transpose(-2, -1)  # [batch_size, num_features, num_features]
    cov_matrix = stddev_outer * corr_matrix
    
    return cov_matrix


'''
def my_truncated_normal_ppf(
    confidence: torch.Tensor,
    a: Union[float, torch.Tensor],
    b: Union[float, torch.Tensor],
    mean: torch.Tensor,
    stddev: torch.Tensor,
    label_num: int,
    machine_mus: torch.Tensor,
    method: 'conditional',  # Ensure method is required
    mixture_weight: float = 0.7,
    num_samples: int = 1000,
    window_size: int = 100
) -> torch.Tensor:
    """
    Unified PPF calculation function with manual method selection for multi-label data.
    """
    device = mean.device
    normal = torch.distributions.Normal(0, 1)
    
    # Handle 2D input tensors by reshaping them to 3D
    if machine_mus.dim() == 2:
        machine_mus = machine_mus.unsqueeze(-1).expand(-1, -1, label_num)
    if mean.dim() == 2:
        mean = mean.unsqueeze(-1).expand(-1, -1, label_num)
    if stddev.dim() == 2:
        stddev = stddev.unsqueeze(-1).expand(-1, -1, label_num)

    # Validate input dimensions
    if machine_mus.dim() != 3:
        raise ValueError(f"Expected machine_mus to be a 2D or 3D tensor, got shape {machine_mus.shape}")
    if machine_mus.shape[-1] != label_num:
        raise ValueError(f"Expected {label_num} labels, got {machine_mus.shape[-1]} labels in machine_mus")
    if mean.shape != stddev.shape:
        raise ValueError(f"Mean and stddev must have the same shape, got mean: {mean.shape}, stddev: {stddev.shape}")

    # Determine the methods to use
    if isinstance(method, str):
        methods = [method] * label_num  # Use the same method for all labels
    elif isinstance(method, list) and len(method) == label_num:
        methods = method  # Use the provided methods for each label
    else:
        raise ValueError(f"Invalid method input: expected a single string or a list of length {label_num}")
    
    # Ensure 'a' and 'b' are tensors and on the same device
    a = torch.tensor(a, device=device).expand_as(mean)
    b = torch.tensor(b, device=device).expand_as(mean)
    
    results = []
    
    for label_idx in range(label_num):
        label_data = machine_mus[..., label_idx]
        label_mean = mean[..., label_idx]
        label_stddev = stddev[..., label_idx]
        current_method = methods[label_idx]  # Select method for the current label
        
        if current_method == 'univariate':
            std = label_stddev
            cdf_a = normal.cdf((a[..., label_idx] - label_mean) / std)
            cdf_b = normal.cdf((b[..., label_idx] - label_mean) / std)
            ppf_input = confidence * cdf_b + cdf_a * (1.0 - confidence)
            ppf_input = torch.clamp(ppf_input, 1e-6, 1 - 1e-6)
            result = normal.icdf(ppf_input)
            
        elif current_method == 'empirical':
            # Generate random samples
            samples = torch.randn(num_samples, mean.size(0), device=device)  # Shape: (num_samples, num_features)
            
            # Compute pairwise correlation matrix (ensure it's square: (num_features, num_features))
            corr_matrix = pairwise_correlation(label_data)  # Should have shape (num_features, num_features)
            
            # Perform Cholesky decomposition
            L = torch.linalg.cholesky(corr_matrix + 1e-6 * torch.eye(corr_matrix.size(0), device=device))  # Shape: (num_features, num_features)
            
            # Matrix multiplication: samples @ L.T -> (num_samples, num_features)
            transformed_samples = (samples @ L.T) * label_stddev + label_mean  # Ensure broadcasting works here
            
            # Sort transformed samples
            sorted_samples, _ = torch.sort(transformed_samples, dim=0)
            
            # Quantile index based on confidence level
            quantile_idx = (confidence * (num_samples - 1)).long()
            
            # Get the final result
            result = sorted_samples[quantile_idx]

            
        elif current_method == 'conditional':
            sorted_indices = torch.argsort(label_mean, dim=0)
            sorted_indices = sorted_indices.view(-1, 1).expand(-1, label_data.size(1))
            sorted_data = torch.gather(label_data, 0, sorted_indices)
            batch_size = sorted_data.size(0)
            num_features = sorted_data.size(1)
            effective_window_size = min(window_size, batch_size)
            
            local_corrs = []
            for i in range(0, max(1, batch_size - effective_window_size + 1)):
                window = sorted_data[i:i + effective_window_size]
                if window.dim() == 2:
                    window = window.unsqueeze(-1)
                try:
                    local_corr = pairwise_correlation(window)
                    if not isinstance(local_corr, torch.Tensor):
                        local_corr = torch.tensor(local_corr, device=device)
                    if local_corr.dim() == 0:
                        local_corr = torch.eye(num_features, device=device)
                    elif local_corr.dim() == 1:
                        local_corr = local_corr.view(num_features, num_features)
                except Exception:
                    local_corr = torch.eye(num_features, device=device)
                
                local_corrs.append(local_corr)
            
            local_corr_matrix = torch.stack(local_corrs)
            window_indices = torch.clamp(
                torch.arange(batch_size, device=device) // effective_window_size,
                0, len(local_corrs) - 1
            )
            selected_corrs = local_corr_matrix[window_indices]
            
            if label_stddev.dim() == 1:
                label_stddev = label_stddev.unsqueeze(0)
            if label_stddev.dim() == 3 and label_stddev.size(-1) == 1:
                label_stddev = label_stddev.squeeze(-1)
            
            cov_matrix = calculate_covariance_matrix(label_stddev, selected_corrs)
            std = torch.sqrt(torch.clamp(torch.diagonal(cov_matrix, dim1=-2, dim2=-1), min=1e-6))
            cdf_a = normal.cdf((a[..., label_idx] - label_mean) / std)
            cdf_b = normal.cdf((b[..., label_idx] - label_mean) / std)
            ppf_input = confidence * cdf_b + cdf_a * (1.0 - confidence)
            ppf_input = torch.clamp(ppf_input, 1e-6, 1 - 1e-6)
            result = normal.icdf(ppf_input)
            
        elif current_method == 'mixture':
            std_uni = label_stddev
            cdf_a_uni = normal.cdf((a[..., label_idx] - label_mean) / std_uni)
            cdf_b_uni = normal.cdf((b[..., label_idx] - label_mean) / std_uni)
            ppf_input_uni = confidence * cdf_b_uni + cdf_a_uni * (1.0 - confidence)
            ppf_input_uni = torch.clamp(ppf_input_uni, 1e-6, 1 - 1e-6)
            y_uni = normal.icdf(ppf_input_uni)
            
            corr_matrix = pairwise_correlation(label_data)
            cov_matrix = calculate_covariance_matrix(label_stddev, corr_matrix)
            std_multi = torch.sqrt(torch.clamp(torch.diagonal(cov_matrix, dim1=-2, dim2=-1), min=1e-6))
            cdf_a_multi = normal.cdf((a[..., label_idx] - label_mean) / std_multi)
            cdf_b_multi = normal.cdf((b[..., label_idx] - label_mean) / std_multi)
            ppf_input_multi = confidence * cdf_b_multi + cdf_a_multi * (1.0 - confidence)
            ppf_input_multi = torch.clamp(ppf_input_multi, 1e-6, 1 - 1e-6)
            y_multi = normal.icdf(ppf_input_multi)
            result = mixture_weight * y_uni + (1 - mixture_weight) * y_multi
        
        else:
            raise ValueError(f"Unknown method: {current_method}")
        
        results.append(result)
    
    # Initialize final_result with the correct shape
    final_result = torch.zeros((mean.size(0), label_num), device=device)  # [batch_size, label_num]
    
    for label_idx in range(label_num):
        final_result[:, label_idx] = results[label_idx][:, label_idx]
    
    return final_result
'''






def gaussian_function(a, b, c, x):
    _part = (- torch.div((x - b) ** 2, 2.0 * (c ** 2)))
    _f = -torch.exp(_part) + a + 1.0
    return _f


class RiskLoss(_Loss):
    def __init__(self, risk_model,  size_average=None, reduce=None, reduction='mean'):
        super(RiskLoss, self).__init__(size_average, reduce, reduction)
        self.LEARN_VARIANCE = cfg.learn_variance
        # self.rm = risk_model
        self.a = torch.tensor(0., dtype=torch.float32).to(device2[0])
        self.b = torch.tensor(1., dtype=torch.float32).to(device2[0])
        self.alpha = torch.tensor(risk_model.learn_confidence, dtype=torch.float32).to(device2[0])
        self.weight_func_a = risk_model.func_params[0].to(device2[0])
        self.weight_func_b = risk_model.func_params[1].to(device2[0])
        self.weight_func_c = risk_model.func_params[2].to(device2[0])
        self.m = -1
        self.continuous_m = cfg.interval_number_4_continuous_value
        self.discrete_m = -1
        self.rule_w = risk_model.rule_w.to(device2[0])
        self.rule_var = risk_model.rule_var.to(device2[0])
        self.machine_var = risk_model.machine_var.to(device2[0])
        self.reduction = 'mean'
        del risk_model

    def forward(self, machine_lables, rule_mus, machine_mus,
                rule_feature_matrix, machine_feature_matrix, machine_one, outputs, labels=None):
        outputs =outputs.clone().detach().requires_grad_(True)
        machine_pro = softmax(outputs.to(device2[0]), dim=-1)
        machine_pro = machine_pro.to(dtype=torch.float32)
        # print(machine_pro.shape)
        # rule_w = self.rule_w.clamp(0, 1)
        # rule_var = self.rule_var.clamp(1e-10, 1)
        # machine_var = self.machine_var.clamp(1e-10, 1)

        machine_mus_vector = torch.reshape(torch.sum(machine_mus, 2), (-1, class_num)).to(device2[0])
        # machine_mus_vector = torch.reshape(machine_pro, (-1, class_num)).to(device2[0])
        # machine_w = gaussian_function(self.weight_func_a.to(torch.float32), self.weight_func_b.to(torch.float32), self.weight_func_c.to(torch.float32), machine_mus_vector.reshape(-1, class_num))
        machine_w = gaussian_function(self.weight_func_a.to(torch.float32), self.weight_func_b.to(torch.float32),
                                      self.weight_func_c.to(torch.float32), machine_mus_vector.reshape(-1, class_num))
        machine_w = machine_w.reshape((-1, class_num))

        big_mu = torch.sum(rule_mus * self.rule_w, 2) + machine_mus_vector * machine_w + 1e-10
        # big_mu = torch.sum(rule_mus * self.rule_w, 2) + machine_mus_vector * torch.sum(rule_feature_matrix * self.rule_w, 2) + 1e-10

        # new_machine_feature_matrix = torch.zeros_like(machine_feature_matrix, requires_grad=False)
        # for i in range(len(new_machine_feature_matrix)):
        #     for j in range(len(new_machine_feature_matrix[0])):
        #         x = int(99) if int(machine_pro[i][j] * 100) == 100 else int(machine_pro[i][j] * 100)
        #         new_machine_feature_matrix[i][j][x] = 1
        # machine_sigma = new_machine_feature_matrix * self.machine_var
        # machine_sigma = machine_feature_matrix * self.machine_var
        # machine_sigma_vector = torch.sum(machine_sigma, 2).reshape((-1, class_num))

        # RSD
        # new_rule_mus = rule_mus.clone()
        # new_mac_mus = machine_mus.clone()
        # new_rule_mus[torch.where(new_rule_mus < 1e-10)] = 0.1
        # new_mac_mus[torch.where(new_mac_mus < 1e-10)] = 0.1
        # rule_standard_deviation = new_rule_mus * self.rule_var
        # mac_standard_deviation = new_mac_mus * self.machine_var
        # rule_var = rule_standard_deviation ** 2
        # machine_var = mac_standard_deviation ** 2
        # print(self.rule_var.shape)
        rule_sigma = rule_feature_matrix * self.rule_var

        machine_sigma = machine_feature_matrix * self.machine_var
        machine_sigma_vector = torch.sum(machine_sigma, 2).reshape((-1, class_num))

        big_sigma = torch.sum(rule_sigma * (self.rule_w ** 2), 2) + machine_sigma_vector * (machine_w ** 2) + 1e-10
        # big_sigma = torch.sum(rule_sigma * (self.rule_w ** 2), 2) + machine_sigma_vector * (torch.sum(rule_feature_matrix * self.rule_w, 2) ** 2) + 1e-10
        weight_vector = torch.sum(rule_feature_matrix * self.rule_w, 2) + machine_w + 1e-10
        label_num = len(big_sigma[0])
        #print(big_sigma)
        big_mu = big_mu / weight_vector
        big_sigma = big_sigma / (weight_vector ** 2)
        #
        #Fr_alpha = my_truncated_normal_ppf(self.alpha, self.a, self.b, big_mu, torch.sqrt(big_sigma), label_num, machine_mus_vector, 'mixture')
        # 一个截断正态分布
        #Fr_alpha_bar = my_truncated_normal_ppf(1 - self.alpha, self.a, self.b, big_mu, torch.sqrt(big_sigma), label_num, machine_mus_vector, 'mixture')
        Fr_alpha = my_truncated_normal_ppf(self.alpha, self.a, self.b, big_mu, torch.sqrt(big_sigma), label_num)
        Fr_alpha_bar = my_truncated_normal_ppf(1 - self.alpha, self.a, self.b, big_mu, torch.sqrt(big_sigma), label_num)
        #Fr_alpha = my_truncated_normal_ppf(self.alpha, self.a, self.b, big_mu, torch.sqrt(big_sigma))
        # 一个截断正态分布
        #Fr_alpha_bar = my_truncated_normal_ppf(1 - self.alpha, self.a, self.b, big_mu, torch.sqrt(big_sigma))
        #print(Fr_alpha_bar)
        # torch.argmax返回指定维度最大值的序号
        #print(Fr_alpha_bar)
        # Assuming Fr_alpha_bar is a 2D tensor
        
        risk_label = torch.zeros_like(Fr_alpha_bar[:, 0], dtype=torch.int64)

        # Values >= 0.95 should result in 1
        mask_high = Fr_alpha_bar[:, 0] >= 0.95
        risk_label[mask_high] = 1
        
        # Values between 0.80 and 0.95 should result in -1
        mask_medium = (Fr_alpha_bar[:, 0] >= 0.80) & (Fr_alpha_bar[:, 0] < 0.95)
        risk_label[mask_medium] = -1
        
        # Values < 0.80 should result in 0 (default case)
        mask_low = Fr_alpha_bar[:, 0] < 0.80
        risk_label[mask_low] = 0
        
        #risk_label = torch.argmax(Fr_alpha_bar, 1)
        return risk_label
        #
        # # 2020.7.2 lfy 修改成软标签
        # machine_one = machine_one.float()
        # Fr_alpha = my_truncated_normal_ppf(self.alpha, self.a, self.b, big_mu, torch.sqrt(big_sigma))
        # # 一个截断正态分布
        # Fr_alpha_bar = my_truncated_normal_ppf(1 - self.alpha, self.a, self.b, big_mu, torch.sqrt(big_sigma))
        # # print(Fr_alpha)
        # prob_mul = Fr_alpha * (torch.ones_like(machine_one) - machine_one) + (torch.ones_like(Fr_alpha_bar) - Fr_alpha_bar) * machine_one
        # prob_mul = 1 - Fr_alpha_bar
        # prob = torch.sum(prob_mul * machine_one, 1)
        # prob = torch.reshape(prob, (-1, 1))
        # prob=prob.squeeze(1)
        # return prob
        '''
        risk_one_label = torch.zeros_like(machine_pro, requires_grad=False)
        for i in range(len(risk_one_label)):
            risk_one_label[i][risk_label[i]] = 1
       
        prob = torch.sum(- risk_one_label * torch.log(machine_pro + 1e-10), 1).reshape(-1, 1)
        # print(1 - Fr_alpha_bar)
        # pred = softmax(Fr_alpha_bar * 100, dim=1)
        # prob = torch.sum(- pred * torch.log(machine_pro + 1e-10), 1)
        # print('-----')
        # print(prob)
        # print('----------mac')
        # print(torch.argmax(machine_pro, 1))
        # print('----label')
        # print(labels)
        # print('max fr')
        # print(torch.argmax(Fr_alpha, 1))
        # print('min Fr_bar')
        # print(torch.argmax(pred, 1))
        # print(torch.max(pred, 1))

        instance_weight = None
        if instance_weight is not None:
            instance_weight = instance_weight.to(device2[0])
            prob = prob * instance_weight

        if self.reduction != 'none':
            if instance_weight is not None:
                ret = torch.sum(prob) / torch.sum(instance_weight) if self.reduction == 'mean' else torch.sum(prob)
            else:
                ret = torch.mean(prob) if self.reduction == 'mean' else torch.sum(prob)

        return ret.type(torch.FloatTensor).to(device2[0]), risk_label
        '''

