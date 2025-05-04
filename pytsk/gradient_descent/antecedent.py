import itertools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from sklearn.cluster import KMeans
from torch.nn import functional

from .utils import check_tensor
import importlib.util

# For illustrative purposes.
package_name = 'pandas'

spec = importlib.util.find_spec(package_name)
if spec is None:
    print(package_name +" is not installed")


def antecedent_init_center(X, y=None, n_rule=2, method="kmean", engine="sklearn", n_init=20):
    """

    This function run KMeans clustering to obtain the :code:`init_center` for :func:`AntecedentGMF() <AntecedentGMF>`.

    Examples
    --------
    >>> init_center = antecedent_init_center(X, n_rule=10, method="kmean", n_init=20)
    >>> antecedent = AntecedentGMF(X.shape[1], n_rule=10, init_center=init_center)


    :param numpy.array X: Feature matrix with the size of :math:`[N,D]`, where :math:`N` is the
        number of samples, :math:`D` is the number of features.
    :param numpy.array y: None, not used.
    :param int n_rule: Number of rules :math:`R`. This function will run a KMeans clustering to
        obtain :math:`R` cluster centers as the initial antecedent center for TSK modeling.
    :param str method: Current version only support "kmean".
    :param str engine: "sklearn" or "faiss". If "sklearn", then the :code:`sklearn.cluster.KMeans()`
        function will be used, otherwise the :code:`faiss.Kmeans()` will be used. Faiss provide a
        faster KMeans clustering algorithm, "faiss" is recommended for large datasets.
    :param int n_init: Number of initialization of the KMeans algorithm. Same as the parameter
        :code:`n_init` in :code:`sklearn.cluster.KMeans()` and the parameter :code:`nredo` in
        :code:`faiss.Kmeans()`.
    """
    def faiss_cluster_center(X, y=None, n_rule=2, n_init=20):
        import faiss
        km = faiss.Kmeans(d=X.shape[1], k=n_rule, nredo=n_init)
        km.train(np.ascontiguousarray(X.astype("float32")))
        centers = km.centroids.T
        return centers

    if method == "kmean":
        if engine == "faiss":
            package_name = "faiss"
            spec = importlib.util.find_spec(package_name)
            if spec is not None:
                center = faiss_cluster_center(X=X, y=y, n_rule=n_rule)
                return center
            else:
                print("Package " + package_name + " is not installed, running scikit-learn KMeans...")
        km = KMeans(n_clusters=n_rule, n_init=n_init)
        km.fit(X)
        return km.cluster_centers_.T


class Antecedent(nn.Module):
    def forward(self, **kwargs):
        raise NotImplementedError

    def init(self, **kwargs):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError


class AntecedentGMF(Antecedent):
    """

    Parent: :code:`torch.nn.Module`

    The antecedent part with Gaussian membership function. Input: data, output the corresponding
    firing levels of each rule. The firing level :math:`f_r(\mathbf{x})` of the
    :math:`r`-th rule are computed by:

    .. math::
        &\mu_{r,d}(x_d) = \exp(-\frac{(x_d - m_{r,d})^2}{2\sigma_{r,d}^2}),\\
        &f_{r}(\mathbf{x})=\prod_{d=1}^{D}\mu_{r,d}(x_d),\\
        &\overline{f}_r(\mathbf{x}) = \frac{f_{r}(\mathbf{x})}{\sum_{i=1}^R f_{i}(\mathbf{x})}.


    :param int in_dim: Number of features :math:`D` of the input.
    :param int n_rule: Number of rules :math:`R` of the TSK model.
    :param bool high_dim: Whether to use the HTSK defuzzification. If :code:`high_dim=True`,
        HTSK is used. Otherwise the original defuzzification is used. More details can be found at [1].
        TSK model tends to fail on high-dimensional problems, so set :code:`high_dim=True` is highly
         recommended for any-dimensional problems.
    :param numpy.array init_center: Initial center of the Gaussian membership function with
        the size of :math:`[D,R]`. A common way is to run a KMeans clustering and set
        :code:`init_center` as the obtained centers. You can simply run
        :func:`pytsk.gradient_descent.antecedent.antecedent_init_center <antecedent_init_center>`
        to obtain the center.
    :param float init_sigma: Initial :math:`\sigma` of the Gaussian membership function.
    :param float eps: A constant to avoid the division zero error.
    """
    def __init__(self, in_dim, n_rule, high_dim=False, init_center=None, init_sigma=1., eps=1e-8):
        super(AntecedentGMF, self).__init__()
        self.in_dim = in_dim
        self.n_rule = n_rule
        self.high_dim = high_dim

        self.init_center = check_tensor(init_center, torch.float32) if init_center is not None else None
        self.init_sigma = init_sigma
        self.zr_op = torch.mean if high_dim else torch.sum
        self.eps = eps

        self.__build_model__()

    def __build_model__(self):
        self.center = nn.Parameter(torch.zeros(size=(self.in_dim, self.n_rule)))
        self.sigma = nn.Parameter(torch.zeros(size=(self.in_dim, self.n_rule)))

        self.reset_parameters()

    def init(self, center, sigma):
        """

        Change the value of :code:`init_center` and :code:`init_sigma`.

        :param numpy.array center: Initial center of the Gaussian membership function with the
            size of :math:`[D,R]`. A common way is to run a KMeans clustering and set
            :code:`init_center` as the obtained centers. You can simply run
            :func:`pytsk.gradient_descent.antecedent.antecedent_init_center <antecedent_init_center>`
            to obtain the center.
        :param float sigma: Initial :math:`\sigma` of the Gaussian membership function.
        """
        center = check_tensor(center, torch.float32)
        self.init_center = center
        self.init_sigma = sigma

        self.reset_parameters()

    def reset_parameters(self):
        """
        Re-initialize all parameters.

        :return:
        """
        init.constant_(self.sigma, self.init_sigma)

        if self.init_center is not None:
            self.center.data[...] = torch.FloatTensor(self.init_center)
        else:
            init.normal_(self.center, 0, 1)

    def forward(self, X):
        """

        Forward method of Pytorch Module.

        :param torch.tensor X: Pytorch tensor with the size of :math:`[N, D]`, where :math:`N` is the number of samples, :math:`D` is the input dimension.
        :return: Firing level matrix :math:`U` with the size of :math:`[N, R]`.
        """
        frs = self.zr_op(
            -(X.unsqueeze(dim=2) - self.center) ** 2 * (0.5 / (self.sigma ** 2 + self.eps)), dim=1
        )
        frs = functional.softmax(frs, dim=1)
        return frs


class AntecedentTriMF2(Antecedent):
    def __init__(self, in_dim, n_rule, high_dim=False, init_param=None, eps=1e-8):
        super().__init__()
        self.in_dim = in_dim
        self.n_rule = n_rule
        self.eps = eps
        self.zr_op = torch.mean if high_dim else torch.prod

        # 使用Buffer代替Parameter维护参数约束
        self.register_buffer('a', torch.zeros(in_dim, n_rule))
        self.b = nn.Parameter(torch.zeros(in_dim, n_rule))
        self.register_buffer('c', torch.zeros(in_dim, n_rule))

        self.reset_parameters(init_param)

    def reset_parameters(self, init_param=None):
        """改进的参数初始化策略"""
        with torch.no_grad():  # 禁用梯度计算
            if init_param:
                a_init, b_init, c_init = init_param
                self.a.copy_(torch.tensor(a_init))
                self.b.copy_(torch.tensor(b_init))
                self.c.copy_(torch.tensor(c_init))
            else:
                # 使用均匀分布初始化
                nn.init.uniform_(self.b, 0.3, 0.7)  # 中心点集中在中间区域
                delta = nn.init.uniform_(torch.empty_like(self.b), 0.1, 0.3)

                # 直接操作buffer数据保证类型一致性
                self.a[:] = self.b - delta
                self.c[:] = self.b + delta

            # 添加约束条件的正确实现方式
            self.a.clamp_(min=0, max=0.999)  # 限制在[0,1)范围内
            self.c.clamp_(min=0.001, max=1)  # 限制在(0,1]范围内

    def forward(self, X):
        X = X.unsqueeze(2)  # [N,D,1]

        # 改进的数值稳定实现
        denominator_b = torch.where(self.b - self.a < self.eps,
                                    self.eps, self.b - self.a)
        denominator_c = torch.where(self.c - self.b < self.eps,
                                    self.eps, self.c - self.b)

        left = (X - self.a) / denominator_b
        right = (self.c - X) / denominator_c
        mu = torch.clamp(torch.min(left, right), min=0.0, max=1.0)

        frs = self.zr_op(mu, dim=1)  # [N,R]
        return functional.softmax(frs, dim=1)

class AntecedentShareGMF(Antecedent):
    def __init__(self, in_dim, n_mf=2, high_dim=False, init_center=None, init_sigma=1., eps=1e-8):
        """
        The antecedent part with Gaussian membership function, rules will share the membership
        functions on each feature [2]. The number of rules will be :math:`M^D`, where :math:`M`
        is :code:`n_mf`, :math:`D` is the number of features (:code:`in_dim`).

        :param int in_dim: Number of features :math:`D` of the input.
        :param int n_mf: Number of membership functions :math:`M` of each feature.
        :param bool high_dim: Whether to use the HTSK defuzzification. If :code:`high_dim=True`,
            HTSK is used. Otherwise the original defuzzification is used. More details can be found
            at [1]. TSK model tends to fail on high-dimensional problems, so set :code:`high_dim=True`
            is highly recommended for any-dimensional problems.
        :param numpy.array init_center: Initial center of the Gaussian membership function with
            the size of :math:`[D,M]`.
        :param float init_sigma: Initial :math:`\sigma` of the Gaussian membership function.
        :param float eps: A constant to avoid the division zero error.

        [1] `Cui Y, Wu D, Xu Y. Curse of dimensionality for tsk fuzzy neural networks:
        Explanation and solutions[C]//2021 International Joint Conference on Neural Networks
        (IJCNN). IEEE, 2021: 1-8. <https://arxiv.org/pdf/2102.04271.pdf>`_
        """
        super(AntecedentShareGMF, self).__init__()

        self.in_dim = in_dim
        self.n_mf = n_mf
        self.n_rule = self.n_mf ** self.in_dim
        self.high_dim = high_dim

        self.init_center = check_tensor(init_center, torch.float32) if init_center is not None else None
        self.init_sigma = init_sigma
        self.zr_op = torch.mean if high_dim else torch.sum
        self.eps = eps

        self.__build_model__()

    def __build_model__(self):
        self.center = nn.Parameter(torch.normal(0, 1, size=(self.in_dim, self.n_mf)))
        self.sigma = nn.Parameter(torch.ones(size=(self.in_dim, self.n_mf)) * self.init_sigma)

        self.rule_index = list(itertools.product(*[range(self.n_mf) for _ in range(self.in_dim)]))

        self.reset_parameters()

    def init(self, center, sigma):
        """
        Change the value of :code:`init_center` and :code:`init_sigma`.

        :param numpy.array center: Initial center of the Gaussian membership function
            with the size of :math:`[D,M]`.
        :param float sigma: Initial :math:`\sigma` of the Gaussian membership function.
        """
        center = check_tensor(center, torch.float32)
        self.init_center = center
        self.init_sigma = sigma

        self.reset_parameters()

    def reset_parameters(self):
        """
        Re-initialize all parameters.
        :return:
        """
        init.constant_(self.sigma, self.init_sigma)

        if self.init_center is not None:
            self.center.data[...] = torch.FloatTensor(self.init_center)
        else:
            init.normal_(self.center, 0, 1)

    def forward(self, X):
        """
        Forward method of Pytorch Module.

        :param torch.tensor X: pytorch tensor with the size of :math:`[N, D]`,
            where :math:`N` is the number of samples, :math:`D` is the input dimension.
        :return: Firing level matrix :math:`U` with the size of :math:`[N, R], R=M^D`:.
        """
        zrs = []
        for r in range(self.n_rule):
            mf_index = torch.tensor(self.rule_index[r], device=X.device, dtype=torch.long).unsqueeze(-1)
            center, sigma = torch.gather(self.center, 1, mf_index), torch.gather(self.sigma, 1, mf_index)

            zr = -0.5 * (X.unsqueeze(2) - center) ** 2 / (sigma ** 2 + self.eps)
            zr = self.zr_op(zr, dim=1)
            zrs.append(zr)
        zrs = torch.cat(zrs, dim=1)
        frs = functional.softmax(zrs, dim=1)
        return frs


class AntecedentTriMF(nn.Module):
    """
    Antecedent with triangle membership function
    """
    def __init__(self, in_dim, n_rule, init_center=None, init_left_dist=3., init_right_dist=3., eps=1e-8):
        super(AntecedentTriMF, self).__init__()
        self.in_dim = in_dim
        self.n_rule = n_rule

        self.init_center = check_tensor(init_center, torch.float32) if init_center is not None else None
        self.init_left_dist = math.sqrt(init_left_dist)
        self.init_right_dist = math.sqrt(init_right_dist)
        self.eps = eps

        self.__build_model__()

    def __build_model__(self):
        self.center = nn.Parameter(torch.zeros(size=(self.in_dim, self.n_rule)))
        self.left_dist = nn.Parameter(torch.ones(size=(self.in_dim, self.n_rule)) * self.init_left_dist)
        self.right_dist = nn.Parameter(torch.ones(size=(self.in_dim, self.n_rule)) * self.init_right_dist)

        self.reset_parameters()

    def forward(self, X):
        X = X.unsqueeze(dim=2)
        left_mf = 1 / (self.left_dist ** 2 + self.eps) * X + 1 - self.center / (self.left_dist ** 2 + self.eps)
        right_mf = - 1 / (self.right_dist ** 2 + self.eps) * X + 1 + self.center / (self.right_dist ** 2 + self.eps)
        mf = torch.maximum(
            torch.zeros(1, device=left_mf.device),
            torch.minimum(left_mf, right_mf)
        )
        frs = torch.prod(mf, dim=1)
        frs = frs / (torch.sum(frs, dim=1, keepdim=True) + self.eps)
        return frs

    def init(self, center, left_dist, right_dist):
        """

        Change the value of :code:`init_center` and :code:`init_sigma`.

        :param numpy.array center: Initial center of the Gaussian membership function with the
            size of :math:`[D,R]`. A common way is to run a KMeans clustering and set
            :code:`init_center` as the obtained centers. You can simply run
            :func:`pytsk.gradient_descent.antecedent.antecedent_init_center <antecedent_init_center>`
            to obtain the center.
        :param float left_dist: Initial :math:`\sigma` of the Gaussian membership function.
        :param float right_dist: Initial :math:`\sigma` of the Gaussian membership function.

        """
        center = check_tensor(center, torch.float32)
        self.init_center = center
        self.init_left_dist = math.sqrt(left_dist)
        self.init_right_dist = math.sqrt(right_dist)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Re-initialize all parameters.

        :return:
        """
        init.constant_(self.left_dist, self.init_left_dist ** 2)
        init.constant_(self.right_dist, self.init_right_dist ** 2)

        if self.init_center is not None:
            self.center.data[...] = torch.FloatTensor(self.init_center)
        else:
            init.normal_(self.center, 0, 1)


class AntecedentShareTriMF(nn.Module):
    def __init__(self, in_dim, n_mf=2, init_center=None, init_left_dist=3., init_right_dist=3., eps=1e-8):
        super(AntecedentShareTriMF, self).__init__()

        self.in_dim = in_dim
        self.n_mf = n_mf
        self.n_rule = self.n_mf ** self.in_dim

        self.init_center = check_tensor(init_center, torch.float32) if init_center is not None else None
        self.init_left_dist = math.sqrt(init_left_dist)
        self.init_right_dist = math.sqrt(init_right_dist)
        self.eps = eps

        self.__build_model__()

    def __build_model__(self):
        self.center = nn.Parameter(torch.zeros(size=(self.in_dim, self.n_mf)))
        self.left_dist = nn.Parameter(torch.ones(size=(self.in_dim, self.n_mf)) * self.init_left_dist)
        self.right_dist = nn.Parameter(torch.ones(size=(self.in_dim, self.n_mf)) * self.init_right_dist)

        self.rule_index = list(itertools.product(*[range(self.n_mf) for _ in range(self.in_dim)]))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Re-initialize all parameters.

        :return:
        """
        init.constant_(self.left_dist, self.init_left_dist)
        init.constant_(self.right_dist, self.init_right_dist)

        if self.init_center is not None:
            self.center.data[...] = torch.FloatTensor(self.init_center)
        else:
            init.normal_(self.center, 0, 1)

    def forward(self, X):
        X = X.unsqueeze(dim=2)

        frs = []
        for r in range(self.n_rule):
            mf_index = torch.tensor(self.rule_index[r], device=X.device, dtype=torch.long).unsqueeze(-1)
            c, ld, rd = torch.gather(self.center, 1, mf_index), \
                        torch.gather(self.left_dist, 1, mf_index), \
                        torch.gather(self.right_dist, 1, mf_index)

            left_mf = 1 / (ld ** 2 + self.eps) * X + 1 - c / (ld ** 2 + self.eps)
            right_mf = - 1 / (rd ** 2 + self.eps) * X + 1 + c / (rd ** 2 + self.eps)

            mf = torch.maximum(
                torch.zeros(1, device=left_mf.device),
                torch.minimum(left_mf, right_mf)
            )
            frs.append(torch.prod(mf, dim=1))
        frs = torch.cat(frs, dim=1)
        frs = frs / (torch.sum(frs, dim=1, keepdim=True) + self.eps)
        return frs

class AntecedentBernoulliMF(Antecedent):
    """
    The antecedent part with Bernoulli membership function. Input: data, output the corresponding
    firing levels of each rule. The firing level :math:`f_r(\mathbf{x})` of the
    :math:`r`-th rule are computed by:

    .. math::
        &\mu_{r,d}(x_d) = \exp\left( x_d \ln(p_{r,d}) + (1 - x_d) \ln(1 - p_{r,d}) \right),\\
        &f_{r}(\mathbf{x}) = \frac{1}{D} \sum_{d=1}^{D} \mu_{r,d}(x_d),\\
        &\overline{f}_r(\mathbf{x}) = \frac{f_{r}(\mathbf{x})}{\sum_{i=1}^R f_{i}(\mathbf{x})}.

    :param int in_dim: Number of features :math:`D` of the input.
    :param int n_rule: Number of rules :math:`R` of the TSK model.
    :param numpy.array init_p: Initial probability :math:`p` of the Bernoulli membership function with
        the size of :math:`[D,R]`. A common way is to initialize :code:`init_p` with random values between 0 and 1.
    :param float eps: A constant to avoid the division zero error.
    """
    def __init__(self, in_dim, n_rule, init_p=None, eps=1e-15):
        super(AntecedentBernoulliMF, self).__init__()
        self.in_dim = in_dim
        self.n_rule = n_rule
        self.eps = eps

        # Initialize the probability parameters p for each dimension and rule
        self.p = nn.Parameter(self._initialize_p(init_p))

    def _initialize_p(self, init_p):
        """
        Helper method to initialize the probability parameters p.
        """
        if init_p is not None:
            return torch.clamp(torch.tensor(init_p, dtype=torch.float32), self.eps, 1 - self.eps)
        else:
            # Initialize p with random values between 0 and 1
            return torch.empty((self.in_dim, self.n_rule)).uniform_(self.eps, 1 - self.eps)

    def forward(self, X):
        """
        Forward method of Pytorch Module.

        :param torch.tensor X: Pytorch tensor with the size of :math:`[N, D]`, where :math:`N` is the number of samples, :math:`D` is the input dimension.
        :return: Firing level matrix :math:`U` with the size of :math:`[N, R]`.
        """
        # Clip p to avoid numerical instability
        p_clipped = torch.clamp(self.p, self.eps, 1 - self.eps)

        # Compute the membership function for each dimension and rule
        mu = torch.exp(X.unsqueeze(dim=2) * torch.log(p_clipped) + (1 - X.unsqueeze(dim=2)) * torch.log(1 - p_clipped))

        # Compute the firing level by averaging the membership values across dimensions
        frs = torch.mean(mu, dim=1)

        # Normalize the firing levels
        frs_normalized = functional.softmax(frs, dim=1)

        return frs_normalized

def antecedent_init_p(X, n_rule=2, method="kmean", engine="sklearn", n_init=20, eps=1e-15):
    def faiss_cluster_center(X, n_rule, n_init):
        import faiss
        km = faiss.Kmeans(d=X.shape[1], k=n_rule, nredo=n_init)
        km.train(np.ascontiguousarray(X.astype("float32")))
        return km.centroids.T

    if method == "kmean":
        if engine == "faiss":
            try:
                centers = faiss_cluster_center(X, n_rule, n_init)
            except ImportError:
                print("Package faiss is not installed, running scikit-learn KMeans...")
                km = KMeans(n_clusters=n_rule, n_init=n_init)
                km.fit(X)
                centers = km.cluster_centers_.T
        else:
            km = KMeans(n_clusters=n_rule, n_init=n_init)
            km.fit(X)
            centers = km.cluster_centers_.T

    # Normalize the cluster centers to [0, 1] range
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    normalized_centers = (centers - min_vals[:, None]) / (max_vals[:, None] - min_vals[:, None] + eps)

    # Clip the normalized centers to [eps, 1-eps] to avoid numerical instability
    init_p = torch.clamp(torch.tensor(normalized_centers, dtype=torch.float32), eps, 1 - eps)

    return init_p

class AntecedentATanMF(nn.Module):
    """
    The antecedent part with arc tangent membership function. Input: data, output the corresponding
    firing levels of each rule. The firing level :math:`f_r(\mathbf{x})` of the
    :math:`r`-th rule are computed by:

    .. math::
        &\mu_{r,d}(x_d) = p + (1 - 2p) \cdot \left( \frac{1}{\pi} \arctan\left( \frac{\pi}{2} \alpha (x_d - m_{r,d}) \right) + \frac{1}{2} \right),\\
        &f_{r}(\mathbf{x})=\begin{cases}
            \prod_{d=1}^{D}\mu_{r,d}(x_d), & \text{if } \text{high_dim} = \text{False}, \\
            \frac{1}{D} \sum_{d=1}^{D}\mu_{r,d}(x_d), & \text{if } \text{high_dim} = \text{True},
        \end{cases}\\
        &\overline{f}_r(\mathbf{x}) = \frac{f_{r}(\mathbf{x})}{\sum_{i=1}^R f_{i}(\mathbf{x})}.

    :param int in_dim: Number of features :math:`D` of the input.
    :param int n_rule: Number of rules :math:`R` of the TSK model.
    :param bool high_dim: Whether to use the mean (True) or product (False) to combine membership values.
    :param float alpha: Fixed :math:`\alpha` of the arc tangent membership function, default is 8.0.
    :param float center: Fixed center of the arc tangent membership function, default is 0.5.
    :param float init_p: Initial value of :math:`p`, which is the only learnable parameter, default is 0.1.
    :param float eps: A constant to avoid the division zero error, default is 1e-8.
    """
    def __init__(self, in_dim: int, n_rule: int, high_dim: bool = False, alpha: float = 8.0,
                 center: float = 0.5, init_p: torch.Tensor = None, eps: float = 1e-8):
        super(AntecedentATanMF, self).__init__()
        self.in_dim = in_dim
        self.n_rule = n_rule
        self.high_dim = high_dim
        self.alpha = alpha
        self.center = center
        self.eps = eps

        # 如果 init_p 未提供，则默认随机初始化，值在 0 到 1 之间
        if init_p is None:
            init_p = torch.rand(size=(in_dim, n_rule), dtype=torch.float32)

        # 确保 init_p 的形状为 [in_dim, n_rule] 并且数据类型为 torch.float32
        if init_p.shape != (in_dim, n_rule):
            raise ValueError(f"init_p must have shape [{in_dim}, {n_rule}], but got {init_p.shape}")
        if init_p.dtype != torch.float32:
            init_p = init_p.to(dtype=torch.float32)

        # 只有 p 是可学习的参数，形状为 [in_dim, n_rule]
        self.p = nn.Parameter(init_p)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward method of Pytorch Module.

        :param torch.tensor X: Pytorch tensor with the size of :math:`[N, D]`, where :math:`N` is the number of samples, :math:`D` is the input dimension.
        :return: Firing level matrix :math:`U` with the size of :math:`[N, R]`.
        """
        # 计算 atan 隶属度函数
        shifted_x = X.unsqueeze(dim=2) - self.center  # [batch_size, in_dim, 1]
        atan_part = (math.pi / 2 * self.alpha * shifted_x).atan() / math.pi + 0.5
        mf = self.p + (1 - 2 * self.p) * atan_part  # [batch_size, in_dim, 1]

        # 根据 high_dim 选择组合方式
        if self.high_dim:
            # 使用均值
            frs = torch.mean(mf, dim=1)  # [batch_size, 1]
        else:
            # 使用乘积
            frs = torch.prod(mf, dim=1)  # [batch_size, 1]

        # 归一化隶属度值
        frs = frs / (torch.sum(frs, dim=1, keepdim=True) + self.eps)

        # 扩展到 n_rule 维度
        frs = frs.expand(-1, self.n_rule)  # [batch_size, n_rule]

        return frs

    def reset_parameters(self):
        """
        Re-initialize the learnable parameter `p`.
        """
        init.constant_(self.p, 0.1)


class ThresholdedATanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, p, center, alpha):
        """
        Forward pass: If X < 0.5, membership value is p; if X >= 0.5, membership value is 1 - p.
        """
        ctx.save_for_backward(X, p)  # 只保存张量
        ctx.center = center  # 保存 center 作为类属性
        ctx.alpha = alpha  # 保存 alpha 作为类属性
        mf = torch.where(X < 0.5, p, 1 - p)  # [batch_size, in_dim, n_rule]
        return mf

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Use the arc tangent membership function to approximate the gradient.
        """
        X, p = ctx.saved_tensors
        center = ctx.center  # 从类属性中获取 center
        alpha = ctx.alpha  # 从类属性中获取 alpha

        # 计算 atan 隶属度函数的梯度
        shifted_x = X - center  # [batch_size, in_dim, n_rule]
        atan_part = (math.pi / 2 * alpha * shifted_x).atan() / math.pi + 0.5
        mf = p + (1 - 2 * p) * atan_part  # [batch_size, in_dim, n_rule]

        # 计算梯度
        grad_mf = (1 - 2 * p) * (1 / (1 + (math.pi / 2 * alpha * shifted_x) ** 2)) * (math.pi / 2 * alpha)

        # 将梯度传递回去
        grad_input = grad_output * grad_mf
        grad_p = grad_output * (1 - 2 * atan_part)

        return grad_input, grad_p, None, None  # 返回四个梯度，最后一个为 None（对应 alpha 和 center）

class ThresholdedATanMF(nn.Module):
    """
    The antecedent part with a threshold-based membership function for forward pass and arc tangent membership function for backward pass.
    Input: data, output the corresponding firing levels of each rule.

    :param int in_dim: Number of features :math:`D` of the input.
    :param int n_rule: Number of rules :math:`R` of the TSK model.
    :param bool high_dim: Whether to use the mean (True) or product (False) to combine membership values.
    :param float alpha: Fixed :math:`\alpha` of the arc tangent membership function, default is 8.0.
    :param float center: Fixed center of the arc tangent membership function, default is 0.5.
    :param float init_p: Initial value of :math:`p`, which is the only learnable parameter, default is 0.1.
    :param float eps: A constant to avoid the division zero error, default is 1e-8.
    """
    def __init__(self, in_dim: int, n_rule: int, high_dim: bool = False, alpha: float = 8.0,
                 center: float = 0.5, init_p: torch.Tensor = None, eps: float = 1e-8):
        super(ThresholdedATanMF, self).__init__()
        self.in_dim = in_dim
        self.n_rule = n_rule
        self.high_dim = high_dim
        self.eps = eps

        # 将 center 和 alpha 转换为张量，并注册为缓冲区
        self.register_buffer('center', torch.tensor(center, dtype=torch.float32))
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))

        # 如果 init_p 未提供，则默认随机初始化，值在 0 到 1 之间
        if init_p is None:
            init_p = torch.rand(size=(in_dim, n_rule), dtype=torch.float32)

        # 确保 init_p 的形状为 [in_dim, n_rule] 并且数据类型为 torch.float32
        if init_p.shape != (in_dim, n_rule):
            raise ValueError(f"init_p must have shape [{in_dim}, {n_rule}], but got {init_p.shape}")
        if init_p.dtype != torch.float32:
            init_p = init_p.to(dtype=torch.float32)

        # 只有 p 是可学习的参数，形状为 [in_dim, n_rule]
        self.p = nn.Parameter(init_p)

        # 定义自定义的前向和反向传播函数
        self.thresholded_atan = ThresholdedATanFunction.apply

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward method of Pytorch Module.

        :param torch.tensor X: Pytorch tensor with the size of :math:`[N, D]`, where :math:`N` is the number of samples, :math:`D` is the input dimension.
        :return: Firing level matrix :math:`U` with the size of :math:`[N, R]`.
        """
        batch_size, in_dim = X.shape
        n_rule = self.p.shape[1]

        # 扩展 X 的形状为 [batch_size, in_dim, 1]
        X_expanded = X.unsqueeze(dim=2)  # [batch_size, in_dim, 1]

        # 扩展 p 的形状为 [1, in_dim, n_rule]
        p_expanded = self.p.unsqueeze(0).expand(batch_size, in_dim, n_rule)

        # 使用自定义的前向和反向传播函数
        mf = self.thresholded_atan(X_expanded, p_expanded, self.center, self.alpha)  # [batch_size, in_dim, n_rule]

        # 根据 high_dim 选择组合方式
        if self.high_dim:
            # 使用均值
            frs = torch.mean(mf, dim=1)  # [batch_size, n_rule]
        else:
            # 使用乘积
            frs = torch.prod(mf, dim=1)  # [batch_size, n_rule]

        # 归一化隶属度值
        frs = frs / (torch.sum(frs, dim=1, keepdim=True) + self.eps)

        return frs

    def reset_parameters(self):
        """
        Re-initialize the learnable parameter `p`.
        """
        init.constant_(self.p, 0.1)


class AntecedentSigmoidMF(nn.Module):
    """
    Antecedent with sigmoid membership function. Implements two modes:
    - Ascending (left-shoulder): μ(x) = 1 / (1 + exp(-a*(x - c)))
    - Descending (right-shoulder): μ(x) = 1 / (1 + exp(a*(x - c)))

    Firing level calculation follows the same logic as AntecedentATanMF

    :param int in_dim: Input feature dimension D
    :param int n_rule: Number of fuzzy rules R
    :param bool high_dim: Use mean (True) or product (False) for membership aggregation
    :param bool ascending: True for ascending S-curve (default), False for descending
    :param float init_a: Initial slope parameter (controls curve steepness)
    :param float init_c: Initial center point (where μ=0.5)
    :param float eps: Epsilon for numerical stability
    """

    def __init__(self, in_dim: int, n_rule: int,
                 high_dim: bool = False, ascending: bool = True,
                 init_a: float = 5.0, init_c: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.in_dim = in_dim
        self.n_rule = n_rule
        self.high_dim = high_dim
        self.ascending = ascending
        self.eps = eps

        # 可学习参数定义 (shape: [in_dim, n_rule])
        self.a = nn.Parameter(torch.full((in_dim, n_rule), init_a))
        self.c = nn.Parameter(torch.full((in_dim, n_rule), init_c))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the firing strengths

        :param X: Input tensor of shape [batch_size, in_dim]
        :return: Normalized firing strengths [batch_size, n_rule]
        """
        X = X.unsqueeze(2)  # [batch, in_dim, 1]

        # 核心S型函数计算
        exponent = self.a * (X - self.c)
        if not self.ascending:
            exponent *= -1

        mf = 1 / (1 + torch.exp(-exponent))  # [batch, in_dim, n_rule]

        # 聚合方式选择
        if self.high_dim:
            frs = torch.mean(mf, dim=1)  # [batch, n_rule]
        else:
            frs = torch.prod(mf, dim=1)  # [batch, n_rule]

        # 归一化处理
        frs = frs / (torch.sum(frs, dim=1, keepdim=True) + self.eps)
        return frs

    def reset_parameters(self):
        """重新初始化可学习参数"""
        nn.init.constant_(self.a, 5.0)
        nn.init.constant_(self.c, 0.5)