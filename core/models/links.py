
import numpy

import chainer
from chainer import configuration
from chainer import cuda
from chainer.functions.normalization import batch_normalization
from chainer import initializers
from chainer import link
from chainer.utils import argument
from chainer import variable
from chainer.links import EmbedID
import chainer.functions as F

import numpy

import chainer
from chainer import configuration
from chainer import cuda
from chainer.functions.normalization import batch_normalization
from chainer import initializers
from chainer import link
from chainer.utils import argument
from chainer import variable
from chainer import initializers
from chainer.links import EmbedID
import chainer.functions as F


# WIP
class ConditionalInstanceNormalization(link.Link):
    def __init__(self, size, decay=0.9, eps=2e-5, dtype=None,
                 valid_test=False):
        super(ConditionalInstanceNormalization, self).__init__()
        self.valid_test = valid_test
        self.dtype = chainer.get_dtype(dtype)
        self.decay = decay
        self.eps = eps

        with self.init_scope():
            self.instance_norm = InstanceNormalization(size, use_gamma=False, use_beta=False, decay=self.decay, eps=self.eps, dtype=self.dtype)
            # class 0
            initial_gamma0 = initializers._get_initializer(1)
            initial_gamma0.dtype = self.dtype
            self.gamma0 = variable.Parameter(initial_gamma0, (1,size,1,1))
            initial_beta0 = initializers._get_initializer(0)
            initial_beta0.dtype = self.dtype
            self.beta0 = variable.Parameter(initial_beta0, (1,size,1,1))
            # class 1
            initial_gamma1 = initializers._get_initializer(1)
            initial_gamma1.dtype = self.dtype
            self.gamma1 = variable.Parameter(initial_gamma1, (1,size,1,1))
            initial_beta1 = initializers._get_initializer(0)
            initial_beta1.dtype = self.dtype
            self.beta1 = variable.Parameter(initial_beta1, (1,size,1,1))

    def __call__(self, x, y):
        with cuda.get_device_from_id(self._device_id):
            one = self.xp.ones(y.shape, dtype=x.dtype)
        return (self.gamma0 * x + self.beta0) * y + (self.gamma1 * x + self.beta1) * (one - y) 

class InstanceNormalization(link.Link):

    def __init__(self, size, decay=0.9, eps=2e-5, dtype=None,
                 valid_test=False, use_gamma=True, use_beta=True,
                 initial_gamma=None, initial_beta=None):
        super(InstanceNormalization, self).__init__()
        self.valid_test = valid_test
        self.avg_mean = numpy.zeros(size, dtype=dtype)
        self.avg_var = numpy.zeros(size, dtype=dtype)
        self.N = 0
        self.register_persistent('avg_mean')
        self.register_persistent('avg_var')
        self.register_persistent('N')
        self.decay = decay
        self.eps = eps
        self.dtype = chainer.get_dtype(dtype)

        with self.init_scope():
            if use_gamma:
                if initial_gamma is None:
                    initial_gamma = 1
                initial_gamma = initializers._get_initializer(initial_gamma)
                initial_gamma.dtype = self.dtype
                self.gamma = variable.Parameter(initial_gamma, size)
            if use_beta:
                if initial_beta is None:
                    initial_beta = 0
                initial_beta = initializers._get_initializer(initial_beta)
                initial_beta.dtype = self.dtype
                self.beta = variable.Parameter(initial_beta, size)

    def __call__(self, x, **kwargs):
        """__call__(self, x, finetune=False)
        Invokes the forward propagation of BatchNormalization.
        In training mode, the BatchNormalization computes moving averages of
        mean and variance for evaluation during training, and normalizes the
        input using batch statistics.
        .. warning::
           ``test`` argument is not supported anymore since v2.
           Instead, use ``chainer.using_config('train', False)``.
           See :func:`chainer.using_config`.
        Args:
            x (Variable): Input variable.
            finetune (bool): If it is in the training mode and ``finetune`` is
                ``True``, BatchNormalization runs in fine-tuning mode; it
                accumulates the input array to compute population statistics
                for normalization, and normalizes the input using batch
                statistics.
        """
        # check argument
        argument.check_unexpected_kwargs(
            kwargs, test='test argument is not supported anymore. '
            'Use chainer.using_config')
        finetune, = argument.parse_kwargs(kwargs, ('finetune', False))

        # reshape input x
        original_shape = x.shape
        batch_size, n_ch = original_shape[:2]
        new_shape = (1, batch_size * n_ch) + original_shape[2:]
        reshaped_x = F.reshape(x, new_shape)

        if hasattr(self, 'gamma'):
            gamma = self.gamma
        else:
            with cuda.get_device_from_id(self._device_id):
                gamma = variable.Variable(self.xp.ones(
                    self.avg_mean.shape, dtype=x.dtype))
        if hasattr(self, 'beta'):
            beta = self.beta
        else:
            with cuda.get_device_from_id(self._device_id):
                beta = variable.Variable(self.xp.zeros(
                    self.avg_mean.shape, dtype=x.dtype))

        mean = self.xp.hstack([self.avg_mean] * batch_size)
        var = self.xp.hstack([self.avg_var] * batch_size)
        gamma = self.xp.hstack([gamma.array] * batch_size)
        beta = self.xp.hstack([beta.array] * batch_size)
        if configuration.config.train:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            ret = F.batch_normalization(
                reshaped_x, gamma, beta, eps=self.eps, running_mean=mean,
                running_var=var, decay=decay)
        else:
            # Use running average statistics or fine-tuned statistics.
            ret = F.fixed_batch_normalization(
                reshaped_x, gamma, beta, mean, var, self.eps)

        # ret is normalized input x
        return F.reshape(ret, original_shape)



class ConditionalBatchNormalization(chainer.Chain):
    """
    Conditional Batch Normalization
    Args:
        size (int or tuple of ints): Size (or shape) of channel
            dimensions.
        n_cat (int): the number of categories of categorical variable.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability.
        dtype (numpy.dtype): Type to use in computing.
        use_gamma (bool): If ``True``, use scaling parameter. Otherwise, use
            unit(1) which makes no effect.
        use_beta (bool): If ``True``, use shifting parameter. Otherwise, use
            unit(0) which makes no effect.
    See: `Batch Normalization: Accelerating Deep Network Training by Reducing\
          Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_
    .. seealso::
       :func:`~chainer.functions.batch_normalization`,
       :func:`~chainer.functions.fixed_batch_normalization`
    Attributes:
        gamma (~chainer.Variable): Scaling parameter.
        beta (~chainer.Variable): Shifting parameter.
        avg_mean (numpy.ndarray or cupy.ndarray): Population mean.
        avg_var (numpy.ndarray or cupy.ndarray): Population variance.
        N (int): Count of batches given for fine-tuning.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability. This value is added
            to the batch variances.
    """

    def __init__(self, size, n_cat, decay=0.9, eps=2e-5, dtype=numpy.float32):
        super(ConditionalBatchNormalization, self).__init__()
        self.avg_mean = numpy.zeros(size, dtype=dtype)
        self.register_persistent('avg_mean')
        self.avg_var = numpy.zeros(size, dtype=dtype)
        self.register_persistent('avg_var')
        self.N = 0
        self.register_persistent('N')
        self.decay = decay
        self.eps = eps
        self.n_cat = n_cat

    def __call__(self, x, gamma, beta, **kwargs):
        """__call__(self, x, c, finetune=False)
        Invokes the forward propagation of BatchNormalization.
        In training mode, the BatchNormalization computes moving averages of
        mean and variance for evaluatino during training, and normalizes the
        input using batch statistics.
        .. warning::
           ``test`` argument is not supported anymore since v2.
           Instead, use ``chainer.using_config('train', train)``.
           See :func:`chainer.using_config`.
        Args:
            x (Variable): Input variable.
            gamma (Variable): Input variable of gamma of shape
            finetune (bool): If it is in the training mode and ``finetune`` is
                ``True``, BatchNormalization runs in fine-tuning mode; it
                accumulates the input array to compute population statistics
                for normalization, and normalizes the input using batch
                statistics.
        """
        argument.check_unexpected_kwargs(
            kwargs, test='test argument is not supported anymore. '
                         'Use chainer.using_config')
        finetune, = argument.parse_kwargs(kwargs, ('finetune', False))
        with cuda.get_device_from_id(self._device_id):
            _gamma = variable.Variable(self.xp.ones(
                self.avg_mean.shape, dtype=x.dtype))
        with cuda.get_device_from_id(self._device_id):
            _beta = variable.Variable(self.xp.zeros(
                self.avg_mean.shape, dtype=x.dtype))
        if configuration.config.train:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay
            ret = chainer.functions.batch_normalization(x, _gamma, _beta, eps=self.eps, running_mean=self.avg_mean,
                                                        running_var=self.avg_var, decay=decay)
        else:
            # Use running average statistics or fine-tuned statistics.
            mean = variable.Variable(self.avg_mean)
            var = variable.Variable(self.avg_var)
            ret = batch_normalization.fixed_batch_normalization(
                x, _gamma, _beta, mean, var, self.eps)
        shape = ret.shape
        ndim = len(shape)
        gamma = F.broadcast_to(F.reshape(gamma, list(gamma.shape) + [1] * (ndim - len(gamma.shape))), shape)
        beta = F.broadcast_to(F.reshape(beta, list(beta.shape) + [1] * (ndim - len(beta.shape))), shape)
        return gamma * ret + beta



class CategoricalConditionalBatchNormalization(ConditionalBatchNormalization):
    """
    Conditional Batch Normalization
    Args:
        size (int or tuple of ints): Size (or shape) of channel
            dimensions.
        n_cat (int): the number of categories of categorical variable.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability.
        dtype (numpy.dtype): Type to use in computing.
        use_gamma (bool): If ``True``, use scaling parameter. Otherwise, use
            unit(1) which makes no effect.
        use_beta (bool): If ``True``, use shifting parameter. Otherwise, use
            unit(0) which makes no effect.
    See: `Batch Normalization: Accelerating Deep Network Training by Reducing\
          Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_
    .. seealso::
       :func:`~chainer.functions.batch_normalization`,
       :func:`~chainer.functions.fixed_batch_normalization`
    Attributes:
        gamma (~chainer.Variable): Scaling parameter.
        beta (~chainer.Variable): Shifting parameter.
        avg_mean (numpy.ndarray or cupy.ndarray): Population mean.
        avg_var (numpy.ndarray or cupy.ndarray): Population variance.
        N (int): Count of batches given for fine-tuning.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability. This value is added
            to the batch variances.
    """

    def __init__(self, size, n_cat, decay=0.9, eps=2e-5, dtype=numpy.float32,
                 initial_gamma=None, initial_beta=None):
        super(CategoricalConditionalBatchNormalization, self).__init__(
            size=size, n_cat=n_cat, decay=decay, eps=eps, dtype=dtype)

        with self.init_scope():
            if initial_gamma is None:
                initial_gamma = 1
            initial_gamma = initializers._get_initializer(initial_gamma)
            initial_gamma.dtype = dtype
            self.gammas = EmbedID(n_cat, size, initialW=initial_gamma)
            if initial_beta is None:
                initial_beta = 0
            initial_beta = initializers._get_initializer(initial_beta)
            initial_beta.dtype = dtype
            self.betas = EmbedID(n_cat, size, initialW=initial_beta)

    def __call__(self, x, c, finetune=False, **kwargs):
        """__call__(self, x, c, finetune=False)
        Invokes the forward propagation of BatchNormalization.
        In training mode, the BatchNormalization computes moving averages of
        mean and variance for evaluatino during training, and normalizes the
        input using batch statistics.
        .. warning::
           ``test`` argument is not supported anymore since v2.
           Instead, use ``chainer.using_config('train', train)``.
           See :func:`chainer.using_config`.
        Args:
            x (Variable): Input variable.
            c (Variable): Input variable for conditioning gamma and beta
            finetune (bool): If it is in the training mode and ``finetune`` is
                ``True``, BatchNormalization runs in fine-tuning mode; it
                accumulates the input array to compute population statistics
                for normalization, and normalizes the input using batch
                statistics.
        """
        weights, = argument.parse_kwargs(kwargs, ('weights', None))
        if c.ndim == 2 and weights is not None:
            _gamma_c = self.gammas(c)
            _beta_c = self.betas(c)
            _gamma_c = F.broadcast_to(F.expand_dims(weights, 2), _gamma_c.shape) * _gamma_c 
            _beta_c = F.broadcast_to(F.expand_dims(weights, 2), _beta_c.shape) * _beta_c
            gamma_c = F.sum(_gamma_c, 1) 
            beta_c = F.sum(_beta_c, 1) 
        else:
            gamma_c = self.gammas(c)
            beta_c = self.betas(c)
        return super(CategoricalConditionalBatchNormalization, self).__call__(x, gamma_c, beta_c, **kwargs)


def start_finetuning(self):
    """Resets the population count for collecting population statistics.
    This method can be skipped if it is the first time to use the
    fine-tuning mode. Otherwise, this method should be called before
    starting the fine-tuning mode again.
    """
    self.N = 0
