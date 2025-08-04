import torch

class CustomAdam:
    def __init__(self,
                params, 
                lr=1e-3, 
                betas=(0.2, 0.6), 
                eps=1e-8, 
                weight_decay=0):
        """
        custom differentiable adam optimizer
        """
        self.params = list(params)
        self.lr = [lr for i in range(len(params))]  if isinstance(lr, float) else lr
        self.lr = self.lr + self.lr # repeat the learning rate for x and y parameters as they are concatenated
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize moment estimates for each parameter
        self.m = [torch.zeros_like(p) for p in self.params]  # First moment
        self.v = [torch.zeros_like(p) for p in self.params]  # Second moment
        self.t = 0  # Timestep

    def step(self, params, grads):
        # Compute gradients with respect to loss
        
        self.t += 1
        lr_t = self.lr 

        new_params = []

        for i, (param, grad) in enumerate(zip(params, grads)):
            if grad is None:
                continue

            # Apply weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate 
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Correct bias in first and second moments
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            
            # Update parameter
            param_t = param - lr_t[i] * m_hat / (v_hat+ self.eps).sqrt() 

            new_params.append(param_t)
        
        return new_params